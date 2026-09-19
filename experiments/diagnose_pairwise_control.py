"""Can the model do pairwise image comparison at all?

diagnose_position_bias found the model picks the second image ~84% of the time
with only 10% agreement between orders. That is consistent with a position
heuristic -- but equally consistent with the model being unable to map "first"
and "second" onto interleaved images, which is a known VLM weakness and has
nothing to do with faces.

A 2x2 separates them:

  format:   "first/second"  vs  explicit "A"/"B" labels
  question: attractiveness (no ground truth)
            vs nose width (CFD measures it, so accuracy is checkable)

  low accuracy on nose width  -> the instrument is broken, not the construct
  high accuracy on nose width,
    inconsistent on attractiveness -> the model can compare; it simply has no
                                      stable attractiveness preference

    python experiments/diagnose_pairwise_control.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --n-pairs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.rater import VLMRater

FORMATS = {
    "first/second": {
        "question": "Which of these two people {comparative}, the first or the second?",
        "prefix": "The answer is the ",
        "options": ("first", "second"),
        "labelled": False,
    },
    "A/B labels": {
        "question": "Image A and Image B are shown above. "
                    "Which of them {comparative}? Answer A or B.",
        "prefix": "The answer is ",
        "options": ("A", "B"),
        "labelled": True,
    },
}

QUESTIONS = {
    "attractive": ("is more attractive", None),
    "nose width": ("has the wider nose", "NoseWidth"),
}


def resolve(tokenizer, options):
    ids = []
    for option in options:
        for candidate in (option, f" {option}"):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
                break
        else:
            raise SystemExit(f"{option!r} is not a single token")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Vision-token cap per image; must be held "
                             "constant across models and conditions.")
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[matched.join_status == "matched"].dropna(subset=["NoseWidth"])

    # Pair the widest noses against the narrowest so the control question has an
    # unambiguous answer; a marginal difference would not test anything.
    ordered = matched.sort_values("NoseWidth")
    narrow = ordered.head(args.n_pairs).reset_index(drop=True)
    wide = ordered.tail(args.n_pairs).reset_index(drop=True)

    rater = VLMRater(args.model, device=args.device, max_pixels=args.max_pixels)
    tokenizer = getattr(rater.processor, "tokenizer", rater.processor)

    def probability_first(spec, comparative, path_a, path_b):
        content = []
        if spec["labelled"]:
            content.append({"type": "text", "text": "Image A:"})
            content.append({"type": "image", "image": path_a})
            content.append({"type": "text", "text": "Image B:"})
            content.append({"type": "image", "image": path_b})
        else:
            content.append({"type": "image", "image": path_a})
            content.append({"type": "image", "image": path_b})
        content.append({"type": "text", "text": spec["question"].format(comparative=comparative)})

        text = rater.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        )
        images = [Image.open(p).convert("RGB") for p in (path_a, path_b)]
        inputs = rater.processor(
            text=[text + spec["prefix"]], images=images, return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            logits = rater.model(**inputs).logits[0, -1, :].float()
        ids = torch.tensor(resolve(tokenizer, spec["options"]), device=args.device)
        return torch.softmax(logits.index_select(0, ids), dim=-1)[0].item()

    print(f"model: {args.model}   pairs: {args.n_pairs}\n")
    print(f"{'format':<14} {'question':<12} {'slot-1 pref':>12} {'consistent':>11} {'accuracy':>9}")
    print("-" * 62)

    for format_name, spec in FORMATS.items():
        for question_name, (comparative, truth_column) in QUESTIONS.items():
            slot1, consistent, correct = [], [], []
            for a, b in zip(narrow.itertuples(), wide.itertuples()):
                p_ab = probability_first(spec, comparative, str(a.image_path), str(b.image_path))
                p_ba = probability_first(spec, comparative, str(b.image_path), str(a.image_path))
                slot1.append((p_ab + p_ba) / 2)
                consistent.append((p_ab > 0.5) != (p_ba > 0.5))
                if truth_column is not None:
                    # b always has the wider nose, so "b wins" is correct.
                    correct.append(p_ab < 0.5)
                    correct.append(p_ba > 0.5)

            mean = lambda xs: sum(xs) / len(xs)
            accuracy = f"{mean(correct):>8.1%}" if correct else "       --"
            print(f"{format_name:<14} {question_name:<12} {mean(slot1):>12.3f} "
                  f"{mean(consistent):>10.1%} {accuracy}")

    print()
    print("  accuracy ~50% on nose width: the model cannot do this comparison in")
    print("    this format. The position-bias result then says nothing about faces.")
    print("  accuracy >75% on nose width but low consistency on attractiveness:")
    print("    comparison works and there is simply no stable preference.")


if __name__ == "__main__":
    main()
