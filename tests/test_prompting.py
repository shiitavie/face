"""Prompt construction for zero-shot and ICL conditions (spec sections 5.1, 5.5)."""
from facecav.models.prompting import ASSISTANT_PREFIX, build_messages, count_images


def test_zero_shot_prompt_has_a_single_image_and_asks_the_rating_question():
    messages = build_messages()
    assert count_images(messages) == 1
    text = str(messages)
    assert "1 to 7" in text
    assert "attractive" in text


def test_zero_shot_prompt_has_no_assistant_turn():
    # The assistant prefix is appended by the chat template at generation time,
    # so the message list must end on the user's query.
    assert build_messages()[-1]["role"] == "user"


def test_icl_prompt_carries_one_image_per_demonstration_plus_the_query():
    messages = build_messages(demonstrations=[("a.jpg", 2), ("b.jpg", 6)])
    assert count_images(messages) == 3


def test_icl_demonstrations_state_their_ratings_and_precede_the_query():
    messages = build_messages(demonstrations=[("a.jpg", 2), ("b.jpg", 6)])
    answers = [m for m in messages if m["role"] == "assistant"]
    assert len(answers) == 2
    assert f"{ASSISTANT_PREFIX}2" in str(answers[0])
    assert f"{ASSISTANT_PREFIX}6" in str(answers[1])
    assert messages[-1]["role"] == "user"


def test_demonstration_order_is_preserved():
    messages = build_messages(demonstrations=[("a.jpg", 7), ("b.jpg", 1)])
    answers = [str(m) for m in messages if m["role"] == "assistant"]
    assert "7" in answers[0] and "1" in answers[1]


# --- pairwise comparison (the surviving instrument) ---

from facecav.models.prompting import COMPARISON_PREFIX, build_comparison_messages


def test_comparison_prompt_carries_exactly_two_images():
    messages = build_comparison_messages("a.jpg", "b.jpg")
    assert count_images(messages) == 2


def test_comparison_prompt_asks_first_or_second():
    # The A/B-label format measured worse than ordinal on the positive control
    # (76.7% vs 93.3% order consistency), so ordinal is what we use.
    text = str(build_comparison_messages("a.jpg", "b.jpg"))
    assert "first" in text and "second" in text


def test_comparison_images_appear_in_the_order_given():
    messages = build_comparison_messages("left.jpg", "right.jpg")
    images = [p["image"] for m in messages for p in m["content"] if p.get("type") == "image"]
    assert images == ["left.jpg", "right.jpg"]


def test_comparison_prefix_does_not_name_either_option():
    # Naming an option in the prefix would anchor the answer, the same failure
    # that killed the absolute rating scale.
    assert "first" not in COMPARISON_PREFIX
    assert "second" not in COMPARISON_PREFIX
