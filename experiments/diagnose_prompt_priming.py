"""Test whether the scale digits in the question prime the rating.

The question "On a scale of 1 to 7..." contains the literal tokens '1' and '7'.
If the model's first completion token is drawn toward digits merely because they
appear in the context, the expected-rating score measures priming rather than
judgment -- and since '1' sits at the floor of the scale, the bias is downward.

Compares prompt phrasings that differ only in how the scale is expressed. If the
mass on '1' collapses when the digits are spelled out, the effect is priming.

    python experiments/diagnose_prompt_priming.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.prompting import ASSISTANT_PREFIX
from facecav.models.rater import VLMRater

VARIANTS = {
    "digits (current)": "On a scale of 1 to 7, how attractive is this person?",
    "spelled out": "On a scale of one to seven, how attractive is this person?",
    "no scale stated": "How attractive is this person?",
    "digits reversed": "On a scale of 7 to 1, how attractive is this person?",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-images", type=int, default=5)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    rows = manifest[manifest.join_status == "matched"].head(args.n_images)

    rater = VLMRater(args.model, device=args.device)
    ids = torch.tensor(rater.rating_token_ids, device=args.device)
    scale = torch.arange(1, 8, device=args.device, dtype=torch.float32)

    print(f"model: {args.model}   images: {len(rows)}\n")
    print(f"{'variant':<20} {'E[rating]':>10} {'P(1)':>8} {'P(7)':>8} {'off-scale':>10}")
    print("-" * 60)

    for label, question in VARIANTS.items():
        ratings, floor, ceiling, off = [], [], [], []
        for row in rows.itertuples():
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

            full = torch.softmax(logits, dim=-1)
            on_scale = torch.softmax(logits.index_select(0, ids), dim=-1)
            ratings.append((on_scale * scale).sum().item())
            floor.append(on_scale[0].item())
            ceiling.append(on_scale[-1].item())
            off.append(1.0 - full.index_select(0, ids).sum().item())

        mean = lambda xs: sum(xs) / len(xs)
        print(f"{label:<20} {mean(ratings):>10.3f} {mean(floor):>8.3f} "
              f"{mean(ceiling):>8.3f} {mean(off):>10.4f}")

    print("\nIf P(1) drops sharply when the digits are spelled out or omitted,")
    print("the spike is priming from the prompt, not the model's judgment.")
    print("If 'digits reversed' moves P(1) and P(7) together, that confirms it:")
    print("mere presence in context is what matters, not scale position.")


if __name__ == "__main__":
    main()
