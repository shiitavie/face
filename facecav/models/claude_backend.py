"""Rate faces with Claude through the Anthropic API.

Three differences from the local open-weight path, all of which must be reported
as methodological deviations rather than buried:

1. **No assistant prefill.** Claude 5 models reject a trailing assistant turn
   (400), so the digit cannot be forced at a known position. A system
   instruction is used instead.
2. **No temperature control.** ``temperature`` is removed on Claude Opus 5 and
   Sonnet 5 and returns a 400. Lowering temperature is what took the local
   readout's reliability from 0.27 to 0.95, so that lever is simply unavailable
   here -- a user of this API cannot tune away the sampling noise.
3. **No log-probabilities.** The exact readout is impossible; ratings are
   estimated from sampled text, which the local comparison validated at
   Spearman 0.877.

Server-side refusal fallbacks are deliberately NOT enabled. Refusal is one of
the quantities under study, and routing around it would destroy the measurement.
"""

from __future__ import annotations

import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image

from .scoring import classify_response

#: Claude counts image tokens at roughly width*height/750. CFD images are 4.2 MP
#: natively (~5,600 tokens); the local runs capped vision at ~1.0 MP, so 1000px
#: on the long edge (~0.70 MP, ~940 tokens) is the closest practical match.
DEFAULT_MAX_DIMENSION = 1000

SYSTEM_INSTRUCTION = (
    "You are participating in a perception study. Answer with a single digit "
    "and nothing else -- no words, no punctuation, no explanation."
)


@dataclass
class ClaudeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def cost(self, input_price: float, output_price: float) -> float:
        """Dollars, given per-million-token prices."""
        return (
            self.input_tokens * input_price
            + self.cache_write_tokens * input_price * 1.25
            + self.cache_read_tokens * input_price * 0.1
            + self.output_tokens * output_price
        ) / 1_000_000


def encode_image(path: str, max_dimension: int = DEFAULT_MAX_DIMENSION) -> tuple[str, str]:
    """Return (base64 data, media type), resized to a comparable token budget."""
    image = Image.open(path).convert("RGB")
    scale = min(max_dimension / image.width, max_dimension / image.height, 1.0)
    if scale < 1.0:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.standard_b64encode(buffer.getvalue()).decode(), "image/jpeg"


#: OAuth access tokens from `ant auth login` go on Authorization: Bearer and
#: need this beta header. They are short-lived, so a long batch has to be able
#: to refresh mid-run.
OAUTH_BETA = "oauth-2025-04-20"


def _oauth_token() -> str | None:
    """Current access token from the `ant` CLI, or None if unavailable."""
    import shutil
    import subprocess

    if shutil.which("ant") is None:
        return None
    result = subprocess.run(
        ["ant", "auth", "print-credentials", "--access-token"],
        capture_output=True, text=True,
    )
    token = result.stdout.strip()
    return token or None


class ClaudeRater:
    """Rates faces through the Anthropic API.

    Resolves credentials in the SDK's own order, then falls back to an OAuth
    access token from `ant auth login` -- which SDK versions before 1.x do not
    read from the profile on disk. The token is refreshed on an authentication
    failure so a long run survives its expiry.
    """

    def __init__(self, model: str = "claude-opus-5", max_dimension: int = DEFAULT_MAX_DIMENSION):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.max_dimension = max_dimension
        self.usage = ClaudeUsage()
        self._using_oauth = False
        self.client = self._build_client()

    def _build_client(self):
        anthropic = self._anthropic
        try:
            client = anthropic.Anthropic()
            client.messages.count_tokens(  # free, but proves the credential works
                model=self.model, messages=[{"role": "user", "content": "x"}]
            )
            return client
        except Exception:
            pass

        token = _oauth_token()
        if token is None:
            raise SystemExit(
                "No Anthropic credential. Either export ANTHROPIC_API_KEY, or run "
                "`ant auth login` (the OAuth profile is used automatically)."
            )
        self._using_oauth = True
        return anthropic.Anthropic(
            auth_token=token, default_headers={"anthropic-beta": OAUTH_BETA}
        )

    def _refresh(self) -> None:
        if self._using_oauth:
            self.client = self._build_client()

    def _message(self, image_b64: str, media_type: str, question: str):
        return self.client.messages.create(
            model=self.model,
            max_tokens=8,
            system=SYSTEM_INSTRUCTION,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                        # The same image is queried many times; caching it makes
                        # every repeat after the first cost ~10%.
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": question},
                ],
            }],
        )

    def sample_ratings(
        self,
        image_path: str,
        question: str,
        n_samples: int = 16,
        concurrency: int = 4,
    ) -> list[dict]:
        """Sample ``n_samples`` responses, classified as answer/hedged/refusal."""
        image_b64, media_type = encode_image(image_path, self.max_dimension)

        def one(_):
            for attempt in range(4):
                try:
                    response = self._message(image_b64, media_type, question)
                    break
                except self._anthropic.AuthenticationError:
                    # OAuth access tokens expire mid-run; refresh and retry.
                    self._refresh()
                except self._anthropic.RateLimitError as error:
                    if attempt == 3:
                        raise
                    delay = float(
                        getattr(error, "response", None)
                        and error.response.headers.get("retry-after", 0) or 0
                    ) or 2.0 ** attempt
                    time.sleep(delay)
            else:
                raise RuntimeError("exhausted retries")
            usage = response.usage
            self.usage.input_tokens += usage.input_tokens
            self.usage.output_tokens += usage.output_tokens
            self.usage.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
            self.usage.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

            if response.stop_reason == "refusal":
                return {"kind": "refusal", "rating": math.nan, "text": "<api refusal>"}
            text = "".join(b.text for b in response.content if b.type == "text")
            return classify_response(text)

        # The first call must land alone so it writes the cache; the rest read it.
        results = [one(0)]
        if n_samples > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results.extend(pool.map(one, range(n_samples - 1)))
        return results
