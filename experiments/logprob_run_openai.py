"""Read exact rating distributions from OpenAI logprobs, and compare to sampling.

Every Claude number in this project comes from repeated sampling, because
Anthropic exposes no log-probabilities. The gpt-5 family rejects the parameter
too. gpt-4.1 is the one model here where BOTH readouts work, which makes it the
only place the central assumption can be tested: is the sampled estimate
faithful to the distribution it is estimating?

The task must be a scale whose every point is a single token -- "7" is, "73" is
not -- so this runs on the 1-7 attractiveness scale.

A second observation this readout makes visible: gpt-4.1 frequently declines
out loud while still carrying a rating distribution underneath. Sampling sees
~80% refusals and discards the rest; logprobs show the refusal mass AND the
judgment, which is exactly what refusal_mass and expected_rating were built to
separate.

    .venv/bin/python experiments/logprob_run_openai.py \\
        --cfd-root "dataset/CFD Version 3.0" --n-faces 200 --yes
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.openai_backend import OpenAIRater

QUESTION = "On a scale of 1 to 7, how attractive is this person?"
SYSTEM = (
    "You are participating in a perception study. Answer with a single digit "
    "and nothing else -- no words, no explanation."
)
OPTIONS = [str(d) for d in range(1, 8)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--n-faces", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Sampled replicates per face, for the comparison.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    faces = manifest[manifest.join_status == "matched"]
    rng = np.random.default_rng(args.seed)
    faces = faces.iloc[rng.permutation(len(faces))[: args.n_faces]]

    out = args.out or Path(f"artifacts/logprob_{args.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["model_id"] for line in out.open()}
    todo = [row for row in faces.itertuples() if row.model_id not in done]

    calls = len(todo) * (1 + args.n_samples)
    print(f"model   {args.model}")
    print(f"faces   {len(faces)} ({len(done)} done, {len(todo)} to run)")
    print(f"calls   {calls:,}  (1 logprob + {args.n_samples} sampled per face)")
    print(f"tokens  ~{calls * 800 / 1e6:.2f}M input\n")
    if not args.yes:
        raise SystemExit("re-run with --yes to proceed (this spends money)")
    if not todo:
        print("nothing to do")
        return

    rater = OpenAIRater(args.model)
    scale = np.arange(1, 8)

    with out.open("a") as handle:
        for n, row in enumerate(todo, start=1):
            exact = rater.rating_distribution(
                str(row.image_path), QUESTION, OPTIONS, system=SYSTEM
            )
            sampled = rater.sample_ratings(
                str(row.image_path), QUESTION,
                n_samples=args.n_samples, system=SYSTEM, max_tokens=8,
            )
            values = [k["rating"] for k in sampled if not math.isnan(k["rating"])]

            record = {
                "model": args.model,
                "model_id": row.model_id,
                "race_code": row.race_code,
                "gender_code": row.gender_code,
                # Exact expected rating, read rather than estimated.
                "exact_expected": (
                    float(sum(scale[i] * exact[OPTIONS[i]] for i in range(7)))
                    if exact else None
                ),
                "exact_distribution": exact,
                "sampled_mean": float(np.mean(values)) if values else None,
                "sampled_values": values,
                "refusal_rate": float(
                    np.mean([k["kind"] == "refusal" for k in sampled])
                ),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}   "
                      f"{rater.usage.input_tokens / 1e6:.2f}M input tokens used")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
