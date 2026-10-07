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
    #: Models reason before answering harder questions and that reasoning
    #: consumes the budget. The two providers fail differently when it runs
    #: out: Claude returns an EMPTY string with stop_reason "max_tokens",
    #: silently recorded as unparseable, while OpenAI raises a hard 400. The
    #: default is therefore generous enough that neither happens; measured
    #: usage is 1-5 tokens for a bare rating and ~166 for mm measurement.
    max_tokens: int = 128


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
        BARE_NUMBER, 1, 120, max_tokens=256,
    ),

    # --- objective measurands, each with a CFD ground truth ---

    "skin_tone": Task(
        "On a scale of 1 to 100, how light is this person's skin, where 1 is "
        "the darkest and 100 the lightest?",
        # The model sometimes reasons on this one. At max_tokens=8 that cost
        # 2.8% of responses as empty strings; raised so the budget never
        # silently truncates an answer.
        BARE_NUMBER, 1, 100, max_tokens=256,
    ),
    # Ratio form: scale-free, so no calibration reference is needed.
    "lip_thickness_pct": Task(
        "What percentage of this face's total height (hairline to chin) is "
        "taken up by the thickness of the lips? Answer as a percentage.",
        BARE_NUMBER, 0.5, 40,
    ),
    "nose_width_pct": Task(
        "What percentage of this face's total width is taken up by the width "
        "of the nose at its widest point? Answer as a percentage.",
        BARE_NUMBER, 5, 80,
    ),
    "cheekbone_prominence_pct": Task(
        "How much wider is this face at the cheekbones than at the mouth, "
        "as a percentage of the face's total height? Answer as a percentage.",
        BARE_NUMBER, 0.0, 40,
    ),
    "eyebrow_thickness_pct": Task(
        "What percentage of this face's total height is taken up by the "
        "thickness of an eyebrow? Answer as a percentage.",
        BARE_NUMBER, 0.2, 25,
    ),

    # Direct measurement, anchored on an anthropometric constant. This is the
    # form a surgeon actually uses, and tests whether the model can measure
    # rather than merely rank.
    "brow_position_pct": Task(
        "What percentage of this face's total height (hairline to chin) is the "
        "vertical distance from the centre of the pupil up to the middle of "
        "the eyebrow directly above it? Answer as a percentage.",
        BARE_NUMBER, 1, 40, max_tokens=256,
    ),
    "brow_position_mm": Task(
        "Assume this person's interpupillary distance (centre of one pupil to "
        "the centre of the other) is 63 mm. Using that as your scale "
        "reference, what is the vertical distance from the centre of the pupil "
        "up to the middle of the eyebrow directly above it, in millimetres?",
        BARE_NUMBER, 3, 50, max_tokens=512,
    ),
    "nose_width_mm": Task(
        "Assume this person's interpupillary distance (centre of one pupil to "
        "the centre of the other) is 63 mm. Using that as your scale "
        "reference, how wide is their nose at its widest point, in "
        "millimetres?",
        BARE_NUMBER, 10, 70, max_tokens=512,
    ),
}

#: task -> how to compute its ground truth from CFD columns. Spearman is
#: scale-invariant so ranking works regardless of units, but calibration
#: (is the VALUE right?) needs the truth in the units the question asked for.
GROUND_TRUTH = {
    "skin_tone": ["LuminanceMedian"],
    "lip_thickness_pct": ["LipThickness", "FaceLength"],
    "nose_width_pct": ["NoseWidth", "FaceWidthBZ"],
    "cheekbone_prominence_pct": ["CheekboneProminence"],
    "eyebrow_thickness_pct": ["EyeBrowThicknessAvg", "FaceLength"],
    "nose_width_mm": ["NoseWidth", "EyeDistance"],
    # midpupil_brow is derived by add_derived_measures from PupilTop and
    # MidbrowHairline, so the filter names those raw columns.
    "brow_position_pct": ["PupilTopR", "MidbrowHairlineR", "FaceLength"],
    "brow_position_mm": ["PupilTopR", "MidbrowHairlineR", "EyeDistance"],
}

PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}
#: OpenAI pricing is not hardcoded -- it changes and varies by tier, so the
#: estimate for those models is reported as unknown rather than wrong.
DEFAULT_PRICE = (5.0, 25.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--model", default="claude-opus-5",
                        help="Any Anthropic model id, or 'openai' / an OpenAI "
                             "model id to use the OpenAI backend.")
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

    # Rate only faces whose ground truth exists. eyebrow_thickness_pct and
    # nose_width_mm depend on columns CFD records for 229 faces, so without
    # this the run would pay for 600 unusable ratings.
    required = GROUND_TRUTH.get(args.task, [])
    missing = [c for c in required if c not in faces.columns]
    if missing:
        raise SystemExit(f"manifest lacks ground-truth column(s): {missing}")
    if required:
        faces = faces.dropna(subset=required)

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
        * PRICES.get(args.model, DEFAULT_PRICE)[0] / 1e6
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

    if args.model == "openai" or args.model.startswith(("gpt-", "o3", "o4")):
        from facecav.models.openai_backend import OpenAIRater

        rater = OpenAIRater(None if args.model == "openai" else args.model)
        print(f"  using OpenAI backend, model {rater.model}\n")
    else:
        rater = ClaudeRater(args.model)
    with out.open("a") as handle:
        for n, row in enumerate(todo, start=1):
            responses = rater.sample_ratings(
                str(row.image_path), task.question, n_samples=args.n_samples,
                concurrency=args.concurrency, system=task.system,
                max_tokens=task.max_tokens,
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
                spent = rater.usage.cost(*PRICES.get(args.model, DEFAULT_PRICE))
                print(f"  {n}/{len(todo)}   ${spent:.2f} spent")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
