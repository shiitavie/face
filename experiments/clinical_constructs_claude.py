"""Can Claude assess clinically relevant facial measures, and does the scale work?

Each construct is put through BOTH instruments against the same CFD ground
truth, because either alone is uninterpretable:

  * ordinal severity rating, 0-4 with verbal anchors -- the format validated
    clinical scales actually use (Lemperle, Merz, NLF severity). Single image,
    as in clinical practice; reference plates calibrate a human rater and are
    not a second photo of the patient.
  * counterbalanced pairwise comparison -- the instrument that survived
    validation on the attractiveness battery.

A weak ordinal correlation alone cannot distinguish "cannot perceive the
feature" from "the rating scale is broken". Pairwise is the positive control
that separates them; earlier work on this project drew the wrong conclusion
twice for want of exactly that.

Wrinkle and photoaging severity are deliberately absent: CFD subjects have a
mean rated age of 29 and none above 60, so those scales would be floor-limited
in the stimuli and the result would be uninterpretable.

    .venv/bin/python experiments/clinical_constructs_claude.py \\
        --cfd-root "dataset/CFD Version 3.0" --n-images 30 --n-pairs 24 --yes
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.claude_backend import ClaudeRater
from facecav.models.scoring import classify_response

#: label -> (CFD column, ordinal prompt, comparative phrase, clinical context)
CONSTRUCTS = {
    "asymmetry": (
        "PupilLipAsymmetry",
        "Rate the facial asymmetry of this person on a scale of 0 to 4, where "
        "0 = perfectly symmetric, 1 = minimal asymmetry, 2 = mild asymmetry, "
        "3 = moderate asymmetry, 4 = marked asymmetry.",
        "has the more asymmetric face",
        "facial palsy, post-traumatic reconstruction",
    ),
    "nasal_width": (
        "NoseWidth",
        "Rate the width of this person's nose on a scale of 0 to 4, where "
        "0 = very narrow, 1 = narrow, 2 = average, 3 = wide, 4 = very wide.",
        "has the wider nose",
        "rhinoplasty, alar base reduction",
    ),
    "lip_fullness": (
        "LipThickness",
        "Rate the fullness of this person's lips on a scale of 0 to 4, where "
        "0 = very thin, 1 = thin, 2 = average, 3 = full, 4 = very full.",
        "has fuller lips",
        "lip augmentation",
    ),
    "eyelid_thickness": (
        "EyeLidThicknessAvg",
        "Rate the thickness of this person's upper eyelids on a scale of 0 to 4, "
        "where 0 = very thin, 1 = thin, 2 = average, 3 = thick, 4 = very thick.",
        "has thicker upper eyelids",
        "blepharoplasty",
    ),
    "facial_width": (
        "fWHR2",
        "Rate how wide this person's face is relative to its height, on a scale "
        "of 0 to 4, where 0 = very narrow, 1 = narrow, 2 = average, 3 = wide, "
        "4 = very wide.",
        "has the wider face relative to its height",
        "facial contouring, facial feminization",
    ),
    "cheekbone": (
        "CheekboneProminence",
        "Rate how prominent this person's cheekbones are on a scale of 0 to 4, "
        "where 0 = not prominent, 1 = slightly prominent, 2 = moderately "
        "prominent, 3 = prominent, 4 = very prominent.",
        "has more prominent cheekbones",
        "midface augmentation, malar implants",
    ),
}

PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=30)
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Claude is near-deterministic here (within-image sd "
                             "0.024), so few samples suffice.")
    parser.add_argument("--constructs", nargs="+", default=None, choices=list(CONSTRUCTS))
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/clinical"))
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[matched.join_status == "matched"]
    selected = {k: CONSTRUCTS[k] for k in (args.constructs or CONSTRUCTS)}

    ordinal_calls = len(selected) * args.n_images * args.n_samples
    pair_calls = len(selected) * args.n_pairs * 2
    print(f"model       {args.model}")
    print(f"constructs  {', '.join(selected)}")
    print(f"ordinal     {args.n_images} images x {args.n_samples} samples "
          f"= {ordinal_calls:,} calls")
    print(f"pairwise    {args.n_pairs} pairs x 2 orders (2 images each) "
          f"= {pair_calls:,} calls")
    tokens = 700
    estimate = (ordinal_calls * tokens * 0.3 + pair_calls * tokens * 2) * \
        PRICES.get(args.model, (5.0, 25.0))[0] / 1e6
    print(f"estimate    ~${estimate:.2f}\n")
    if not args.yes:
        raise SystemExit("re-run with --yes to proceed (this spends money)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rater = ClaudeRater(args.model)
    rows = []

    for label, (column, ordinal_prompt, comparative, context) in selected.items():
        usable = matched.dropna(subset=[column])
        # Spread the ordinal sample across the measured range so a flat result
        # cannot be blamed on the stimuli lacking variation.
        spread = usable.sort_values(column)
        picks = spread.iloc[
            np.linspace(0, len(spread) - 1, args.n_images).astype(int)
        ]

        print(f"  {label} ({context})")
        journal = args.out_dir / f"{label}_ordinal.jsonl"
        done = set()
        if journal.exists():
            done = {json.loads(l)["model_id"] for l in journal.open()}

        with journal.open("a") as handle:
            for row in picks.itertuples():
                if row.model_id in done:
                    continue
                kinds = [
                    classify_response(t, scale_max=4, scale_min=0)
                    for t in [
                        k["text"] for k in rater.sample_ratings(
                            str(row.image_path), ordinal_prompt,
                            n_samples=args.n_samples, concurrency=2,
                        )
                    ]
                ]
                valid = [k["rating"] for k in kinds if not math.isnan(k["rating"])]
                handle.write(json.dumps({
                    "model_id": row.model_id,
                    "truth": float(getattr(row, column)),
                    "rating": float(np.mean(valid)) if valid else None,
                    "sd": float(np.std(valid)) if len(valid) > 1 else None,
                }) + "\n")
                handle.flush()

        ordinal = pd.read_json(journal, lines=True).dropna(subset=["rating"])
        rho_ordinal = ordinal.rating.corr(ordinal.truth, method="spearman")
        distinct = ordinal.rating.nunique()

        # Pairwise on extremes, where the correct answer is unambiguous.
        low = spread.head(args.n_pairs).reset_index(drop=True)
        high = spread.tail(args.n_pairs).reset_index(drop=True)
        question = f"Which of these two people {comparative}, the first or the second?"
        correct, consistent = [], []
        for a, b in zip(low.itertuples(), high.itertuples()):
            result = rater.compare(str(a.image_path), str(b.image_path), question)
            consistent.append(result["consistent"])
            if result["consistent"]:
                correct.append(result["prefers_b"])  # b always scores higher

        rows.append({
            "construct": label,
            "clinical_context": context,
            "ordinal_rho": rho_ordinal,
            "ordinal_distinct_values": distinct,
            "ordinal_mean": ordinal.rating.mean(),
            "pair_consistency": float(np.mean(consistent)),
            "pair_accuracy": float(np.mean(correct)) if correct else math.nan,
        })
        print(f"    ordinal rho {rho_ordinal:+.3f} ({distinct} distinct values)   "
              f"pairwise {np.mean(consistent):.0%} consistent, "
              f"{np.mean(correct) if correct else float('nan'):.0%} accurate")

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "summary.csv", index=False)

    print("\n" + "=" * 78)
    print("CLINICAL CONSTRUCTS -- ordinal severity scale vs counterbalanced pairwise")
    print("=" * 78)
    print(results.round(3).to_string(index=False))
    print()
    print("  pairwise accurate but ordinal flat -> the model sees the feature and")
    print("    the 0-4 scale is the failure. Verbal anchors did not rescue it.")
    print("  both flat -> the model cannot assess this feature at all.")
    print("  ordinal_distinct_values near 1 -> range restriction, as on the 1-7")
    print("    attractiveness scale where 75% of responses were the midpoint.")
    print(f"\n  spend ${rater.usage.cost(*PRICES.get(args.model, (5.0, 25.0))):.2f}")


if __name__ == "__main__":
    main()
