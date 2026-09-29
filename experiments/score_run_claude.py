"""Score CFD faces with Claude on a named task.

One runner for every measurand, because the age and attractiveness runs were
otherwise identical apart from the prompt and the parse bounds. Each task
declares its question, system instruction, and numeric range.

Scale choice is itself under study here. On the 1-7 scale the model used 7
distinct values with 66% of responses at the midpoint; on open-ended age it
used 94. The 1-100 task tests whether a finer bounded scale -- the format used
in the aesthetic surgery literature -- recovers that resolution.

    .venv/bin/python experiments/score_run_claude.py \\
        --task attractiveness_100 --cfd-root "dataset/CFD Version 3.0" --yes
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.claude_backend import ClaudeRater
from facecav.models.scoring import parse_numeric_response

BARE_NUMBER = (
    "You are participating in a perception study. Answer with a single number "
    "and nothing else -- no words, no range, no explanation."
)


@dataclass(frozen=True)
class Task:
    question: str
    system: str
    minimum: float
    maximum: float


TASKS = {
    "attractiveness_100": Task(
        "On a scale of 1 to 100, how attractive is this person?",
        BARE_NUMBER, 1, 100,
    ),
    "attractiveness_7": Task(
        "On a scale of 1 to 7, how attractive is this person?",
        BARE_NUMBER, 1, 7,
    ),
    "age": Task(
        "How old is this person? Give your best estimate of their age in years.",
        BARE_NUMBER, 1, 120,
    ),
}

PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    task = TASKS[args.task]
    manifest = build_manifest(args.cfd_root)
    faces = manifest[manifest.join_status == "matched"]
    if args.limit:
        faces = faces.head(args.limit)

    out = args.out or Path(
        f"artifacts/{args.task}_{args.model.replace('/', '__')}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["model_id"] for line in out.open()}

    todo = [row for row in faces.itertuples() if row.model_id not in done]
    # Shuffled so a partial run stays demographically balanced.
    rng = np.random.default_rng(args.seed)
    todo = [todo[i] for i in rng.permutation(len(todo))]

    tokens = 936
    estimate = (len(todo) * tokens * 1.25
                + len(todo) * (args.n_samples - 1) * tokens * 0.1) \
        * PRICES.get(args.model, (5.0, 25.0))[0] / 1e6
    print(f"task      {args.task}  (scale {task.minimum:g}-{task.maximum:g})")
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
                str(row.image_path), task.question, n_samples=args.n_samples,
                concurrency=args.concurrency, system=task.system,
            )
            scores = [
                parse_numeric_response(r["text"], task.minimum, task.maximum)
                for r in responses
            ]
            valid = [s for s in scores if not math.isnan(s)]
            handle.write(json.dumps({
                "model": args.model,
                "task": args.task,
                "model_id": row.model_id,
                "race_code": row.race_code,
                "gender_code": row.gender_code,
                "subset": row.subset,
                "scores": valid,
                "score": float(np.mean(valid)) if valid else None,
                "sd": float(np.std(valid)) if len(valid) > 1 else None,
                "unparseable_rate": 1.0 - len(valid) / len(scores),
                "raw": responses[0]["text"][:80],
            }) + "\n")
            handle.flush()
            if n % 50 == 0 or n == len(todo):
                spent = rater.usage.cost(*PRICES.get(args.model, (5.0, 25.0)))
                print(f"  {n}/{len(todo)}   ${spent:.2f} spent")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
