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

from .prompting import ASSISTANT_PREFIX, build_messages
from .scoring import expected_rating, refusal_mass, resolve_rating_tokens


@dataclass
class Rating:
    expected_rating: float
    refusal_mass: float
    rating_probs: list[float] = field(default_factory=list)


class VLMRater:
    def __init__(self, model_id: str, device: str = "cuda", dtype=torch.float16):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, dtype=dtype, device_map=device
        ).eval()

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # Raises RatingTokenError if the scale is not single-token here, which
        # is the spec 5.6 gate for dropping a model.
        self.rating_token_ids = resolve_rating_tokens(tokenizer)

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
