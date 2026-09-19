"""Do prompt phrasings agree on the ORDERING of faces?

diagnose_prompt_priming showed absolute ratings swing ~2 scale points with
wording alone. That is fatal only if it also scrambles the ranking: this audit
compares groups, so a constant offset cancels while a face-dependent one does
not.

Two questions, in order of importance:

1. Rank agreement across phrasings. High Spearman correlation means the prompt
   sets the level but not the ordering, and between-group comparisons survive.
2. Whether the by-race ordering is preserved. This is the one that matters --
   if race means reorder across phrasings, the demographic finding is an
   artifact of wording.

    python experiments/diagnose_prompt_stability.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --per-cell 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.prompting import ASSISTANT_PREFIX
from facecav.models.rater import VLMRater

VARIANTS = {
    "digits": "On a scale of 1 to 7, how attractive is this person?",
    "spelled": "On a scale of one to seven, how attractive is this person?",
    "worded": "Rate this person's attractiveness from one (lowest) to seven (highest).",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-cell", type=int, default=6,
                        help="Images sampled per race x gender cell.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/prompt_stability.csv"))
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    matched = manifest[manifest.join_status == "matched"]
    sample = (
        matched.groupby(["race_code", "gender_code"], group_keys=False)
        .apply(lambda g: g.head(args.per_cell))
        .reset_index(drop=True)
    )

    rater = VLMRater(args.model, device=args.device)
    ids = torch.tensor(rater.rating_token_ids, device=args.device)
    scale = torch.arange(1, 8, device=args.device, dtype=torch.float32)

    print(f"model: {args.model}   images: {len(sample)}   "
          f"cells: {sample.groupby(['race_code', 'gender_code']).ngroups}\n")

    records = []
    for variant, question in VARIANTS.items():
        print(f"  running {variant}...")
        for row in sample.itertuples():
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": str(row.image_path)},
                            {"type": "text", "text": question}],
            }]
            text = rater.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            image = Image.open(row.image_path).convert("RGB")
            inputs = rater.processor(
                text=[text + ASSISTANT_PREFIX], images=[image], return_tensors="pt"
            ).to(args.device)
            with torch.no_grad():
                logits = rater.model(**inputs).logits[0, -1, :].float()
            probabilities = torch.softmax(logits.index_select(0, ids), dim=-1)
            records.append({
                "variant": variant,
                "model_id": row.model_id,
                "race_code": row.race_code,
                "gender_code": row.gender_code,
                "rating": (probabilities * scale).sum().item(),
            })

    ratings = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(args.out, index=False)

    wide = ratings.pivot(index="model_id", columns="variant", values="rating")

    print("\n" + "=" * 64)
    print("1. RANK AGREEMENT BETWEEN PHRASINGS (Spearman)")
    print("=" * 64)
    print(wide.corr(method="spearman").round(3).to_string())
    print("\n  >0.9 : prompt sets the level, not the ordering -- audit survives")
    print("  <0.7 : wording reorders faces; absolute ratings are not a measurement")

    print("\n" + "=" * 64)
    print("2. MEAN RATING BY RACE, PER PHRASING")
    print("=" * 64)
    by_race = ratings.pivot_table(index="race_code", columns="variant",
                                  values="rating", aggfunc="mean")
    print(by_race.round(3).to_string())

    # Ranking group means is worthless when the spread between them is smaller
    # than their standard errors -- the order reshuffles on noise alone. Compare
    # the size of the race effect against the prompt effect and the error bars.
    print("\n" + "=" * 64)
    print("3. IS THE RACE EFFECT LARGER THAN THE NOISE?")
    print("=" * 64)
    spread = (by_race.max() - by_race.min()).rename("race spread")
    errors = (ratings.groupby(["variant", "race_code"]).rating.sem()
              .groupby("variant").mean().rename("mean SE of a race"))
    summary = pd.concat([spread, errors], axis=1)
    summary["spread / SE"] = (summary["race spread"] / summary["mean SE of a race"])
    print(summary.round(3).to_string())

    prompt_spread = ratings.groupby("variant").rating.mean()
    print(f"\n  between-prompt spread: "
          f"{prompt_spread.max() - prompt_spread.min():.3f} scale points")
    print(f"  largest race spread:   {spread.max():.3f} scale points")
    print()
    print("  spread / SE below ~2 means the race differences are inside their own")
    print("  error bars at this sample size; no ordering of them is meaningful.")
    print("  Resolving an effect this small needs the full sample, not 6 per cell.")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
