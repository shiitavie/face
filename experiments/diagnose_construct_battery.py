"""Is attractiveness special, or does the model fail all social judgments?

The pairwise control established that the instrument works: 93.3% order
consistency and 80% accuracy on nose width, versus 6.7% consistency on
attractiveness in the identical format. The failure is the question, not the
format. Two explanations remain:

  (a) no stable aesthetic representation
  (b) the model declines appearance judgments, and forced choice turns that
      refusal into positional noise

CFD norms a battery of judgments spanning physical to social, so the same
comparison can be run across constructs with human ground truth for each.

  consistent on social judgments but not attractiveness -> (b), suppression
    specific to appearance
  inconsistent on all social judgments, fine on physical -> a general limit on
    social inference; attractiveness is not special

    python experiments/diagnose_construct_battery.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --n-pairs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.rater import VLMRater

QUESTION = "Which of these two people {comparative}, the first or the second?"
PREFIX = "The answer is the "
OPTIONS = ("first", "second")

#: (label, comparative phrase, CFD column, kind)
CONSTRUCTS = [
    ("nose width",  "has the wider nose",        "NoseWidth",      "physical"),
    ("face length", "has the longer face",       "FaceLength",     "physical"),
    ("age",         "looks older",               "AgeRated",       "perceptual"),
    ("masculine",   "looks more masculine",      "Masculine",      "perceptual"),
    ("babyfaced",   "looks more babyfaced",      "Babyfaced",      "social"),
    ("trustworthy", "looks more trustworthy",    "Trustworthy",    "social"),
    ("dominant",    "looks more dominant",       "Dominant",       "social"),
    ("attractive",  "is more attractive",        "attractive_rel", "social"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=20)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[matched.join_status == "matched"]

    rater = VLMRater(args.model, device=args.device, max_pixels=args.max_pixels)
    tokenizer = getattr(rater.processor, "tokenizer", rater.processor)
    ids = []
    for option in OPTIONS:
        encoded = tokenizer.encode(option, add_special_tokens=False)
        if len(encoded) != 1:
            encoded = tokenizer.encode(f" {option}", add_special_tokens=False)
        ids.append(encoded[0])
    ids = torch.tensor(ids, device=args.device)

    def probability_first(comparative, path_a, path_b):
        content = [
            {"type": "image", "image": path_a},
            {"type": "image", "image": path_b},
            {"type": "text", "text": QUESTION.format(comparative=comparative)},
        ]
        text = rater.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        )
        images = [Image.open(p).convert("RGB") for p in (path_a, path_b)]
        inputs = rater.processor(
            text=[text + PREFIX], images=images, return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            logits = rater.model(**inputs).logits[0, -1, :].float()
        return torch.softmax(logits.index_select(0, ids), dim=-1)[0].item()

    print(f"model: {args.model}   pairs per construct: {args.n_pairs}\n")
    print(f"{'construct':<13} {'kind':<11} {'slot-1':>8} {'consistent':>11} {'accuracy':>9}")
    print("-" * 56)

    for label, comparative, column, kind in CONSTRUCTS:
        usable = matched.dropna(subset=[column]).sort_values(column)
        if len(usable) < 2 * args.n_pairs:
            print(f"{label:<13} {kind:<11} {'-- too few rows --':>30}")
            continue
        low = usable.head(args.n_pairs).reset_index(drop=True)
        high = usable.tail(args.n_pairs).reset_index(drop=True)

        slot1, consistent, correct = [], [], []
        for a, b in zip(low.itertuples(), high.itertuples()):
            p_ab = probability_first(comparative, str(a.image_path), str(b.image_path))
            p_ba = probability_first(comparative, str(b.image_path), str(a.image_path))
            slot1.append((p_ab + p_ba) / 2)
            consistent.append((p_ab > 0.5) != (p_ba > 0.5))
            # b is always the high-scoring member of the pair.
            correct.append(p_ab < 0.5)
            correct.append(p_ba > 0.5)

        mean = lambda xs: sum(xs) / len(xs)
        print(f"{label:<13} {kind:<11} {mean(slot1):>8.3f} "
              f"{mean(consistent):>10.1%} {mean(correct):>8.1%}")

    print()
    print("  Physical rows are the positive control and should stay high.")
    print("  If social rows hold up and only attractiveness collapses, the model")
    print("  is declining appearance judgments rather than lacking the concept,")
    print("  and forced choice is manufacturing a number out of a refusal.")


if __name__ == "__main__":
    main()
