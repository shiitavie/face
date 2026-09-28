"""Estimate perceived age for CFD faces with Claude.

Perceived age is a better-founded measurand than attractiveness: it has an
objective referent (the subject's actual age), its errors are interpretable in
years rather than arbitrary scale points, there is a human benchmark to compare
against (the CFD rater panel achieves MAE 5.26 years), and it is the quantity
the facial rejuvenation literature already reports as an outcome.

CFD supplies two criteria:
  * AgeRated -- the human rater panel's mean, for all 826 faces
  * AgeSelf  -- self-reported actual age, for 219 (CFD-INDIA and CFD-MR only)

    .venv/bin/python experiments/age_run_claude.py \\
        --cfd-root "dataset/CFD Version 3.0" --yes
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.claude_backend import ClaudeRater
from facecav.models.scoring import parse_age

QUESTION = "How old is this person? Give your best estimate of their age in years."
SYSTEM = (
    "You are participating in a perception study. Answer with a single number "
    "of years and nothing else -- no words, no range, no explanation."
)
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-with-truth", action="store_true",
                        help="Rate faces having a self-reported actual age "
                             "first, so a partial run still supports the "
                             "objective comparison.")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    faces = manifest[manifest.join_status == "matched"]
    if args.limit:
        faces = faces.head(args.limit)

    out = args.out or Path(f"artifacts/age_{args.model.replace('/', '__')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["model_id"] for line in out.open()}

    todo = [row for row in faces.itertuples() if row.model_id not in done]
    rng = np.random.default_rng(args.seed)
    todo = [todo[i] for i in rng.permutation(len(todo))]
    if args.prefer_with_truth:
        # Objective ground truth exists for only 219 faces; rating those first
        # means an interrupted run still answers the question that needs truth.
        def has_truth(row) -> bool:
            value = getattr(row, "AgeSelf", None)
            try:
                return value is not None and not math.isnan(float(value))
            except (TypeError, ValueError):
                return False

        todo.sort(key=lambda row: 0 if has_truth(row) else 1)

    tokens = 936
    estimate = (len(todo) * tokens * 1.25
                + len(todo) * (args.n_samples - 1) * tokens * 0.1) \
        * PRICES.get(args.model, (5.0, 25.0))[0] / 1e6
    print(f"model     {args.model}")
    print(f"faces     {len(faces)} total, {len(done)} done, {len(todo)} to run")
    print(f"samples   {args.n_samples} each = {len(todo) * args.n_samples:,} calls")
    print(f"estimate  ~${estimate:.2f}\n")
    if not args.yes:
        raise SystemExit("re-run with --yes to proceed (this spends money)")
    if not todo:
        print("nothing to do")
        return

    rater = ClaudeRater(args.model)
    with out.open("a") as handle:
        for n, row in enumerate(todo, start=1):
            responses = rater.sample_ratings(
                str(row.image_path), QUESTION, n_samples=args.n_samples,
                concurrency=args.concurrency, system=SYSTEM,
            )
            ages = [parse_age(r["text"]) for r in responses]
            valid = [a for a in ages if not math.isnan(a)]
            handle.write(json.dumps({
                "model": args.model,
                "model_id": row.model_id,
                "race_code": row.race_code,
                "gender_code": row.gender_code,
                "subset": row.subset,
                "ages": valid,
                "age_estimate": float(np.mean(valid)) if valid else None,
                "sd": float(np.std(valid)) if len(valid) > 1 else None,
                "unparseable_rate": 1.0 - len(valid) / len(ages),
                "raw": responses[0]["text"][:80],
            }) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(todo):
                spent = rater.usage.cost(*PRICES.get(args.model, (5.0, 25.0)))
                print(f"  {n}/{len(todo)}   ${spent:.2f} spent")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
