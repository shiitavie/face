"""Stage 1 pilot: counterbalanced pairwise comparisons over all CFD faces.

Replaces the absolute-rating design, which does not work for this model -- it
reads the digits' form rather than the scale's meaning (ratings correlate at
+0.982 when the scale's direction is reversed).

Every pair is scored in both presentation orders and averaged, which is
mandatory rather than a refinement: a single order gives 53% accuracy against
human norms, the counterbalanced average gives 81%.

Resumable. Results append to JSONL keyed by the unordered pair, and completed
pairs are skipped, so an interrupted Colab session costs one comparison. A long
run can be split across several sessions by rerunning this repeatedly.

    python experiments/stage1_pairwise_pilot.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --comparisons-per-face 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from facecav.analysis.bradley_terry import sample_comparison_pairs
from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.rater import VLMRater

OUT_DIR = Path("artifacts/stage1_pairwise")


def completed_pairs(path: Path) -> set[frozenset]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            frozenset((record["face_a"], record["face_b"]))
            for record in (json.loads(line) for line in handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--comparisons-per-face", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Stop after this many new pairs; for smoke tests "
                             "and for splitting a long run across sessions.")
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    matched = manifest[manifest.join_status == "matched"]
    paths = dict(zip(matched.model_id, matched.image_path))

    pairs = sample_comparison_pairs(
        sorted(paths), args.comparisons_per_face, seed=args.seed
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model.replace('/', '__')}.jsonl"
    done = completed_pairs(out)
    remaining = [p for p in pairs if frozenset(p) not in done]
    if args.max_pairs:
        remaining = remaining[: args.max_pairs]

    print(f"faces      {len(paths)}")
    print(f"pairs      {len(pairs)}  ({len(done)} done, {len(remaining)} to run)")
    print(f"passes     {2 * len(remaining)} forward passes this session\n")

    rater = VLMRater(args.model, device=args.device, max_pixels=args.max_pixels)

    started = time.time()
    with out.open("a") as handle:
        for n, (face_a, face_b) in enumerate(remaining, start=1):
            result = rater.compare(str(paths[face_a]), str(paths[face_b]))
            handle.write(json.dumps({
                "model": args.model,
                "face_a": face_a,
                "face_b": face_b,
                **result,
            }) + "\n")
            handle.flush()

            if n % 50 == 0 or n == len(remaining):
                elapsed = time.time() - started
                rate = n / elapsed
                print(f"  {n}/{len(remaining)}  {rate:.2f} pairs/s  "
                      f"eta {(len(remaining) - n) / rate / 60:.0f} min")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
