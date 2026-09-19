"""Does the model follow the scale's meaning, or anchor on its digits?

The earlier "7 to 1" variant confounded two things: it reversed the digit order
AND inverted the scale's meaning. It therefore cannot distinguish anchoring from
correct scale-following, and any priming conclusion drawn from it is unsound.

This holds the digits and their order fixed -- both variants say "1 to 7" -- and
flips only which end means "most attractive":

  follows meaning : ratings anti-correlate, means straddle the midpoint
  anchors on digits: ratings barely move between the two

    python experiments/diagnose_scale_semantics.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --n-images 40
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
    "7=best": "On a scale of 1 to 7, where 7 is most attractive, "
              "how attractive is this person?",
    "1=best": "On a scale of 1 to 7, where 1 is most attractive, "
              "how attractive is this person?",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    sample = matched[matched.join_status == "matched"].sample(
        args.n_images, random_state=args.seed
    )

    rater = VLMRater(args.model, device=args.device)
    ids = torch.tensor(rater.rating_token_ids, device=args.device)
    scale = torch.arange(1, 8, device=args.device, dtype=torch.float32)

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
                "rating": (probabilities * scale).sum().item(),
                "p_floor": probabilities[0].item(),
                "p_ceiling": probabilities[-1].item(),
            })

    ratings = pd.DataFrame(records)
    wide = ratings.pivot(index="model_id", columns="variant", values="rating")
    correlation = wide["7=best"].corr(wide["1=best"], method="spearman")

    print("\n" + "=" * 62)
    print("SCALE SEMANTICS vs DIGIT ANCHORING")
    print("=" * 62)
    print(ratings.groupby("variant")[["rating", "p_floor", "p_ceiling"]]
          .mean().round(3).to_string())
    print(f"\n  Spearman between the two phrasings: {correlation:+.3f}")
    print()
    print("  strongly NEGATIVE  -> follows the stated meaning; the scale is real")
    print("                        and the earlier priming reading was wrong")
    print("  near ZERO / POSITIVE -> output is governed by the digits, not their")
    print("                        meaning, and absolute ratings are not valid")


if __name__ == "__main__":
    main()
