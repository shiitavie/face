"""Stage 1a: rate every CFD image with one model, zero-shot and ICL.

Resumable by design (spec section 3): results append to JSONL keyed by
(model, condition, image), and an interrupted run skips what is already on
disk. A preempted Spot instance therefore costs at most one image.

    python experiments/stage1a_rate_cfd.py --model Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from facecav.data.cfd import build_manifest
from facecav.models.rater import VLMRater

CFD_ROOT = Path("dataset/CFD Version 3.0")
OUT_DIR = Path("artifacts/stage1a")


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            (record["condition"], record["image_path"])
            for record in (json.loads(line) for line in handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", nargs="+", default=["zero_shot"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model.replace('/', '__')}.jsonl"
    done = completed_keys(out)

    manifest = build_manifest(CFD_ROOT)
    if args.limit:
        manifest = manifest.head(args.limit)

    rater = VLMRater(args.model, device=args.device)
    print(f"{args.model}: rating tokens {rater.rating_token_ids}")

    with out.open("a") as handle:
        for condition in args.conditions:
            for row in manifest.itertuples():
                if (condition, row.image_path) in done:
                    continue
                rating = rater.rate(row.image_path)
                handle.write(
                    json.dumps(
                        {
                            "model": args.model,
                            "condition": condition,
                            "image_path": row.image_path,
                            "model_id": row.model_id,
                            "subset": row.subset,
                            "race_code": row.race_code,
                            "gender_code": row.gender_code,
                            "expected_rating": rating.expected_rating,
                            "refusal_mass": rating.refusal_mass,
                            "rating_probs": rating.rating_probs,
                        }
                    )
                    + "\n"
                )
                handle.flush()

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
