"""Rate faces with OpenAI models, mirroring the Claude backend.

Two reasons this provider is worth the second slot. It is the one most health
systems have actually approved for images, usually through Azure OpenAI under a
BAA, so it speaks to deployed practice. And unlike Anthropic it exposes
log-probabilities, which gives an EXACT rating distribution from a single call
rather than an estimate from repeated sampling -- the measurement error that
drove the whole reliability analysis on the Claude side simply disappears.

The logprob path only works for scales whose every point is one token. "7" is a
single token; "73" is two, so the first-position distribution on a 1-100 scale
is over first digits, not values. Bounded small scales use logprobs; open
numeric tasks fall back to sampling.
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

#: Matches the Claude backend's cap so image content is comparable across
#: providers; see claude_backend.DEFAULT_MAX_DIMENSION.
DEFAULT_MAX_DIMENSION = 1000

#: Preference order, resolved against the models the key can actually see so a
#: retirement or rename does not break the run.
#:
#: gpt-5.1 leads deliberately rather than the newest flagship. Measured on CFD
#: images: 5.1 answers a single-number question in ~10 output tokens, while 5.5
#: reasons first and spends ~51 for the same answer, at roughly 4x the input
#: price, with no evident benefit on a task this simple.
PREFERRED_MODELS = (
    "gpt-5.1", "gpt-5.2", "gpt-5", "gpt-4.1", "gpt-4o",
)

#: Only this generation exposes log-probabilities. The gpt-5 family rejects the
#: parameter outright, so the exact-distribution readout -- and therefore any
#: check of whether the sampled estimate is faithful -- is available only here.
LOGPROB_MODELS = ("gpt-4.1", "gpt-4o")


@dataclass
class OpenAIUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, input_price: float, output_price: float) -> float:
        return (self.input_tokens * input_price
                + self.output_tokens * output_price) / 1_000_000


def encode_image(path: str, max_dimension: int = DEFAULT_MAX_DIMENSION) -> str:
    """Base64 data URL, resized to the same budget the Claude backend uses."""
    image = Image.open(path).convert("RGB")
    scale = min(max_dimension / image.width, max_dimension / image.height, 1.0)
    if scale < 1.0:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    data = base64.standard_b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}"


class OpenAIRater:
    def __init__(
        self,
        model: str | None = None,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
    ):
        import openai

        self._openai = openai
        self.client = openai.OpenAI(max_retries=8)
        self.max_dimension = max_dimension
        self.usage = OpenAIUsage()
        self.model = model or self._resolve_model()

    def _resolve_model(self) -> str:
        """Pick the most capable available model rather than hardcoding one."""
        available = {m.id for m in self.client.models.list().data}
        for candidate in PREFERRED_MODELS:
            if candidate in available:
                return candidate
            # Dated snapshots: "gpt-5-2026-01-01" satisfies "gpt-5".
            matches = sorted(m for m in available if m.startswith(candidate + "-"))
            if matches:
                return matches[-1]
        raise SystemExit(
            f"none of {PREFERRED_MODELS} are available to this key; "
            f"pass --model explicitly"
        )

    def _request(self, image_url: str, question: str, system: str,
                 max_tokens: int, **extra):
        for attempt in range(8):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": question},
                        ]},
                    ],
                    **extra,
                )
            except self._openai.RateLimitError:
                if attempt == 7:
                    raise
                time.sleep(min(2.0 ** attempt, 60.0))
            except self._openai.APIStatusError as error:
                status = getattr(error, "status_code", 0)
                if status < 500 or attempt == 7:
                    raise
                time.sleep(min(2.0 ** attempt, 60.0))
        raise RuntimeError("exhausted retries")

    def _record(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.usage.output_tokens += getattr(usage, "completion_tokens", 0) or 0

    def sample_ratings(
        self,
        image_path: str,
        question: str,
        n_samples: int = 3,
        concurrency: int = 3,
        system: str = "",
        max_tokens: int = 8,
    ) -> list[dict]:
        """Sampled text responses, classified as answer/hedged/refusal."""
        image_url = encode_image(image_path, self.max_dimension)

        def one(_):
            response = self._request(image_url, question, system, max_tokens)
            self._record(response)
            text = response.choices[0].message.content or ""
            return classify_response(text)

        results = [one(0)]
        if n_samples > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results.extend(pool.map(one, range(n_samples - 1)))
        return results

    def rating_distribution(
        self,
        image_path: str,
        question: str,
        options: list[str],
        system: str = "",
    ) -> dict[str, float] | None:
        """EXACT probability over ``options`` from one call, via logprobs.

        Returns None where the model does not support logprobs -- reasoning
        models often do not -- so the caller can fall back to sampling.

        Only valid when every option is a single token: "7" is, "73" is not.
        """
        image_url = encode_image(image_path, self.max_dimension)
        try:
            response = self._request(
                image_url, question, system, max_tokens=4,
                logprobs=True, top_logprobs=20,
            )
        except self._openai.BadRequestError:
            return None

        self._record(response)
        content = response.choices[0].logprobs
        if content is None or not content.content:
            return None

        first = content.content[0]
        probabilities = {
            entry.token.strip(): math.exp(entry.logprob)
            for entry in first.top_logprobs
        }
        selected = {o: probabilities.get(o, 0.0) for o in options}
        total = sum(selected.values())
        if total <= 0:
            return None
        # Renormalise over the rating tokens, as the Claude/Qwen path does.
        return {o: p / total for o, p in selected.items()}
