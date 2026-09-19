"""How does pairwise reliability scale with how different the two faces are?

Two runs gave wildly different answers for attractiveness -- 6.7% order
consistency when pairs were selected on nose width, 65% when selected on
attractiveness itself. The likely cause is pair difficulty: faces that barely
differ have no correct answer, so the model falls back to position. (Resolution
also changed between those runs, so neither is clean.)

This measures the relationship directly, holding everything else fixed. Pairs
are binned by the gap in human attractiveness rating.

  consistency rising with the gap -> a real but noisy preference. Bradley-Terry
    handles exactly this, and the curve says how many comparisons are needed.
  flat and low -> position dominates at every difficulty and the earlier 65%
    was an artifact of sampling only extremes.

    python experiments/diagnose_difficulty_gradient.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --per-bin 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.rater import VLMRater

QUESTION = "Which of these two people is more attractive, the first or the second?"
PREFIX = "The answer is the "
OPTIONS = ("first", "second")
CELL = ["race_code", "gender_code"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--per-bin", type=int, default=15)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/difficulty_gradient.csv"))
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[
        (matched.join_status == "matched") & matched.attractive_rel.notna()
    ]

    # Pair within race x gender cells. R013 is normed within cell, so a gap is
    # only interpretable between faces from the same cell.
    rng = np.random.default_rng(args.seed)
    candidates = []
    for _, cell in matched.groupby(CELL):
        rows = cell.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                candidates.append((
                    abs(rows[i]["attractive_rel"] - rows[j]["attractive_rel"]),
                    rows[i], rows[j],
                ))
    gaps = np.array([c[0] for c in candidates])
    edges = np.quantile(gaps, np.linspace(0, 1, args.bins + 1))
    print(f"{len(candidates)} within-cell pairs available; "
          f"gap range {gaps.min():.2f}-{gaps.max():.2f}\n")

    rater = VLMRater(args.model, device=args.device, max_pixels=args.max_pixels)
    tokenizer = getattr(rater.processor, "tokenizer", rater.processor)
    ids = []
    for option in OPTIONS:
        encoded = tokenizer.encode(option, add_special_tokens=False)
        if len(encoded) != 1:
            encoded = tokenizer.encode(f" {option}", add_special_tokens=False)
        ids.append(encoded[0])
    ids = torch.tensor(ids, device=args.device)

    def probability_first(path_a, path_b):
        text = rater.processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": path_a},
                {"type": "image", "image": path_b},
                {"type": "text", "text": QUESTION},
            ]}],
            add_generation_prompt=True, tokenize=False,
        )
        images = [Image.open(p).convert("RGB") for p in (path_a, path_b)]
        inputs = rater.processor(
            text=[text + PREFIX], images=images, return_tensors="pt"
        ).to(args.device)
        with torch.no_grad():
            logits = rater.model(**inputs).logits[0, -1, :].float()
        return torch.softmax(logits.index_select(0, ids), dim=-1)[0].item()

    records = []
    for b in range(args.bins):
        low, high = edges[b], edges[b + 1]
        pool = [c for c in candidates if low <= c[0] <= high]
        chosen = [pool[k] for k in rng.choice(len(pool),
                                              size=min(args.per_bin, len(pool)),
                                              replace=False)]
        print(f"  bin {b + 1}/{args.bins}  gap {low:.2f}-{high:.2f}  "
              f"n={len(chosen)}")
        for gap, a, z in chosen:
            p_ab = probability_first(str(a["image_path"]), str(z["image_path"]))
            p_ba = probability_first(str(z["image_path"]), str(a["image_path"]))
            # Counterbalanced preference for a, averaging the two orders.
            preference_a = (p_ab + (1 - p_ba)) / 2
            records.append({
                "bin": b + 1, "gap_low": round(low, 3), "gap_high": round(high, 3),
                "gap": gap,
                "consistent": (p_ab > 0.5) != (p_ba > 0.5),
                "slot1": (p_ab + p_ba) / 2,
                # a is correct when a has the higher human rating
                "correct_raw": (p_ab > 0.5) == (a["attractive_rel"] > z["attractive_rel"]),
                "correct_balanced": (preference_a > 0.5) == (a["attractive_rel"] > z["attractive_rel"]),
            })

    results = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    print("\n" + "=" * 72)
    print("RELIABILITY vs PAIR DIFFICULTY (human rating gap)")
    print("=" * 72)
    summary = results.groupby("bin").agg(
        gap_lo=("gap_low", "first"), gap_hi=("gap_high", "first"), n=("gap", "size"),
        slot1=("slot1", "mean"), consistent=("consistent", "mean"),
        acc_one_order=("correct_raw", "mean"),
        acc_counterbalanced=("correct_balanced", "mean"),
    )
    print(summary.round(3).to_string())
    print()
    print(f"  overall counterbalanced accuracy: {results.correct_balanced.mean():.1%}")
    print(f"  overall single-order accuracy:    {results.correct_raw.mean():.1%}")
    print()
    print("  If counterbalanced accuracy exceeds single-order, averaging the two")
    print("  presentations is removing position bias and the preference is real.")
    print("  Rising accuracy with gap = a usable signal; Bradley-Terry over")
    print("  sampled pairs will recover a latent scale per face.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
