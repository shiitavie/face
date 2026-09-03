"""Prompt construction (spec sections 5.1 and 5.5).

Messages use the HuggingFace chat format so each model's own chat template can
render them. The assistant turn is left open for the query: the template adds
the generation prompt and the caller appends ``ASSISTANT_PREFIX``, so the rating
token lands at the first completion position where ``expected_rating`` reads it.
"""

from __future__ import annotations

from collections.abc import Sequence

QUESTION = "On a scale of 1 to 7, how attractive is this person?"
ASSISTANT_PREFIX = "The rating is "


def _user_turn(image: str | None = None) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", **({"image": image} if image else {})},
            {"type": "text", "text": QUESTION},
        ],
    }


def build_messages(
    demonstrations: Sequence[tuple[str, int]] = (),
    query_image: str | None = None,
) -> list[dict]:
    """Chat messages for one rating query.

    ``demonstrations`` is an ordered sequence of ``(image_path, rating)`` pairs
    for the ICL condition; empty gives the zero-shot condition. Held fixed and
    identical across models per spec section 5.5.
    """
    messages: list[dict] = []
    for image, rating in demonstrations:
        messages.append(_user_turn(image))
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"{ASSISTANT_PREFIX}{rating}"}],
            }
        )
    messages.append(_user_turn(query_image))
    return messages


def count_images(messages: Sequence[dict]) -> int:
    return sum(
        1
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    )
