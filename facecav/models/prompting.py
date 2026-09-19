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


# --- pairwise comparison ---
#
# The absolute 1-7 scale is not a usable instrument for this model: flipping
# which end means "most attractive", with the digits and their order held fixed,
# left ratings correlated at +0.982. The model reads the digits' form, not the
# scale's meaning.
#
# Counterbalanced pairwise comparison does work. Against CFD human norms it
# reaches 100% accuracy on clearly-different pairs and 81% overall, versus 53%
# (chance) from a single presentation order -- the model carries a strong,
# near-constant preference for whichever image comes second, and averaging the
# two orders cancels it.

COMPARISON_QUESTION = (
    "Which of these two people is more attractive, the first or the second?"
)

#: Deliberately stops before naming either option. Naming one would anchor the
#: answer, which is what destroyed the absolute rating scale.
COMPARISON_PREFIX = "The answer is the "

COMPARISON_OPTIONS = ("first", "second")


def build_comparison_messages(image_a: str, image_b: str) -> list[dict]:
    """Chat messages asking which of two faces is more attractive.

    Ordinal phrasing ("first"/"second") rather than A/B labels: on the
    nose-width positive control, ordinal reached 93.3% order consistency and
    80% accuracy against A/B labels' 76.7% and 75%.

    Always call this twice per pair, with the images swapped, and average.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_a},
            {"type": "image", "image": image_b},
            {"type": "text", "text": COMPARISON_QUESTION},
        ],
    }]
