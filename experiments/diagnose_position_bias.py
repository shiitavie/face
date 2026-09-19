"""Does the model prefer whichever face is shown first?

Pairwise comparison avoids the numeric anchoring that contaminates absolute
ratings, but only if the model is not simply picking by position. This measures
that directly: every pair is scored in both orders.

    P(i wins | i shown first)  = p_ij
    P(i wins | i shown second) = 1 - p_ji

Unbiased and self-consistent means p_ij + p_ji = 1. A position advantage above
0.5 means the first slot wins regardless of content, and pairwise is no better
than the digit priming it was meant to replace.

    python experiments/diagnose_position_bias.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --n-pairs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.rater import VLMRater

QUESTION = "Which of these two people is more attractive, the first or the second?"
PREFIX = "The more attractive person is the "
OPTIONS = ("first", "second")


def resolve_option_tokens(tokenizer) -> list[int]:
    ids = []
    for option in OPTIONS:
        encoded = tokenizer.encode(option, add_special_tokens=False)
        if len(encoded) != 1:
            encoded = tokenizer.encode(f" {option}", add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(f"{option!r} is not a single token for this model")
        ids.append(encoded[0])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Vision-token cap per image; must be held "
                             "constant across models and conditions.")
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[matched.join_status == "matched"]
    left = matched.sample(args.n_pairs, random_state=args.seed)
    right = matched.sample(args.n_pairs, random_state=args.seed + 1)

    rater = VLMRater(args.model, device=args.device, max_pixels=args.max_pixels)
    tokenizer = getattr(rater.processor, "tokenizer", rater.processor)
    option_ids = torch.tensor(resolve_option_tokens(tokenizer), device=args.device)
    print(f"option tokens: {[tokenizer.decode([i]) for i in option_ids.tolist()]}\n")

    def probability_first(path_a: str, path_b: str) -> float:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": path_a},
                        {"type": "image", "image": path_b},
                        {"type": "text", "text": QUESTION}],
        }]
        text = rater.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        images = [Image.open(p).convert("RGB") for p in (path_a, path_b)]
        inputs = rater.processor(
            text=[text + PREFIX], images=images, return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            logits = rater.model(**inputs).logits[0, -1, :].float()
        return torch.softmax(logits.index_select(0, option_ids), dim=-1)[0].item()

    advantages, consistencies, agreements = [], [], []
    for a, b in zip(left.itertuples(), right.itertuples()):
        if a.model_id == b.model_id:
            continue
        p_ab = probability_first(str(a.image_path), str(b.image_path))
        p_ba = probability_first(str(b.image_path), str(a.image_path))

        # First-slot advantage: 0.5 means none.
        advantages.append((p_ab + p_ba) / 2)
        # Self-consistency: p_ab should equal 1 - p_ba.
        consistencies.append(abs(p_ab - (1 - p_ba)))
        # Do the two orders name the same winner?
        agreements.append((p_ab > 0.5) != (p_ba > 0.5))

    mean = lambda xs: sum(xs) / len(xs)
    print("=" * 60)
    print(f"POSITION BIAS over {len(advantages)} pairs, each scored both ways")
    print("=" * 60)
    print(f"  first-slot advantage      {mean(advantages):.3f}   (0.500 = none)")
    print(f"  mean |p_ab - (1 - p_ba)|  {mean(consistencies):.3f}   (0 = perfectly consistent)")
    print(f"  orders agree on winner    {mean(agreements):.1%}   (100% = order-invariant)")
    print()
    print("  advantage near 0.5 and agreement >90%: pairwise is sound, and")
    print("    order-counterbalancing cleans up the remainder.")
    print("  advantage >0.65 or agreement <70%: the model is choosing by")
    print("    position, and pairwise trades digit anchoring for slot anchoring.")


if __name__ == "__main__":
    main()
