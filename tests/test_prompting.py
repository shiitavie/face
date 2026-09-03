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
