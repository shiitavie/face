"""Thin wrapper turning a HuggingFace VLM into an attractiveness rater.

Reads the logits at the *first completion position* -- the token immediately
after ``ASSISTANT_PREFIX`` -- and reduces them with the ordinal task score.
Full rating-token probabilities are persisted alongside the score so Stage 1
never has to be rerun to answer a question about the distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from PIL import Image

from ..analysis.bradley_terry import counterbalanced_preference
from .prompting import (
    ASSISTANT_PREFIX,
    COMPARISON_OPTIONS,
    COMPARISON_PREFIX,
    build_comparison_messages,
    build_messages,
)
from .scoring import (
    RatingTokenError,
    expected_rating,
    refusal_mass,
    resolve_rating_tokens,
)


@dataclass
class Rating:
    expected_rating: float
    refusal_mass: float
    rating_probs: list[float] = field(default_factory=list)


class VLMRater:
    #: Cap on vision tokens per image. Qwen2.5-VL defaults to 12.8M pixels, far
    #: more than a face judgment needs, and cost scales with it. This is a
    #: MEASUREMENT choice as well as a performance one -- resolution changes
    #: what the model can see -- so it is explicit, recorded, and must be held
    #: constant across every model and condition in a study.
    DEFAULT_MAX_PIXELS = 1280 * 28 * 28

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype=torch.float16,
        max_pixels: int | None = None,
    ):
        import transformers
        from transformers import AutoProcessor

        # AutoModelForImageTextToText is the current class; AutoModelForVision2Seq
        # is legacy and does not map every VLM we target.
        auto_model = getattr(
            transformers, "AutoModelForImageTextToText", None
        ) or transformers.AutoModelForVision2Seq

        self.model_id = model_id
        self.device = device
        self.max_pixels = self.DEFAULT_MAX_PIXELS if max_pixels is None else max_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_id, max_pixels=self.max_pixels
        )
        self.model = auto_model.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # Raises RatingTokenError if the scale is not single-token here, which
        # is the spec 5.6 gate for dropping a model.
        self.rating_token_ids = resolve_rating_tokens(tokenizer)
        self.option_token_ids = self._option_token_ids()

    def _render(self, messages, images: Sequence[Image.Image]):
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self.processor(
            text=[text + ASSISTANT_PREFIX], images=list(images), return_tensors="pt"
        ).to(self.device)

    @torch.no_grad()
    def rate(
        self,
        image_path: str,
        demonstrations: Sequence[tuple[str, int]] = (),
    ) -> Rating:
        messages = build_messages(demonstrations=demonstrations, query_image=image_path)
        images = [Image.open(p).convert("RGB") for p, _ in demonstrations]
        images.append(Image.open(image_path).convert("RGB"))

        logits = self.model(**self._render(messages, images)).logits[0, -1, :].float()

        probabilities = torch.softmax(
            logits.index_select(0, torch.tensor(self.rating_token_ids, device=logits.device)),
            dim=-1,
        )
        return Rating(
            expected_rating=expected_rating(logits, self.rating_token_ids).item(),
            refusal_mass=refusal_mass(logits, self.rating_token_ids).item(),
            rating_probs=probabilities.tolist(),
        )

    def _option_token_ids(self) -> list[int]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids = []
        for option in COMPARISON_OPTIONS:
            for candidate in (option, f" {option}"):
                encoded = tokenizer.encode(candidate, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.append(encoded[0])
                    break
            else:
                raise RatingTokenError(f"{option!r} is not a single token")
        return ids

    @torch.no_grad()
    def _probability_first(self, path_a: str, path_b: str) -> float:
        messages = build_comparison_messages(path_a, path_b)
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        images = [Image.open(p).convert("RGB") for p in (path_a, path_b)]
        inputs = self.processor(
            text=[text + COMPARISON_PREFIX], images=images, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits[0, -1, :].float()
        option_ids = torch.tensor(self.option_token_ids, device=logits.device)
        return torch.softmax(logits.index_select(0, option_ids), dim=-1)[0].item()

    def compare(self, path_a: str, path_b: str) -> dict:
        """Counterbalanced preference for ``a`` over ``b``.

        Scores the pair in both presentation orders and averages. This is not
        optional: the model prefers whichever image is second in ~87-98% of
        trials, and a single order gives chance-level accuracy (53%) where the
        counterbalanced average gives 81%.
        """
        p_ab = self._probability_first(path_a, path_b)
        p_ba = self._probability_first(path_b, path_a)
        # Raw probabilities are always persisted: they are the primitive
        # observation, and any change to the counterbalancing can be applied
        # retrospectively without rerunning the model.
        return {
            "p_a_first": p_ab,
            "p_b_first": p_ba,
            **counterbalanced_preference(p_ab, p_ba),
        }
