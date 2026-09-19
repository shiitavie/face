"""Diagnose what the model actually emits at the first completion position.

The expected-rating score assumes the token after ``ASSISTANT_PREFIX`` is a bare
scale digit. If that assumption is wrong -- the model answers on a different
scale, writes a word first, or the prefix creates a tokenization seam -- the
score is measuring something other than a rating, and every downstream number
inherits the error.

This prints the evidence rather than guessing:

    python experiments/diagnose_first_token.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.prompting import ASSISTANT_PREFIX, build_messages
from facecav.models.rater import VLMRater


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    row = build_manifest(args.cfd_root).iloc[args.index]
    rater = VLMRater(args.model, device=args.device)
    tokenizer = getattr(rater.processor, "tokenizer", rater.processor)

    print(f"model : {args.model}")
    print(f"image : {row.model_id}\n")

    print("=" * 66)
    print("1. RESOLVED RATING TOKENS")
    print("=" * 66)
    print(f"{'rating':>6}  {'id':>8}  decoded")
    for rating, token_id in enumerate(rater.rating_token_ids, start=1):
        print(f"{rating:>6}  {token_id:>8}  {tokenizer.decode([token_id])!r}")
    leading_space = [tokenizer.decode([i]).startswith(" ") for i in rater.rating_token_ids]
    print(f"\nleading-space variants: {sum(leading_space)}/{len(leading_space)}")
    print(f"ASSISTANT_PREFIX ends with space: {ASSISTANT_PREFIX.endswith(' ')}")
    if all(leading_space) and ASSISTANT_PREFIX.endswith(" "):
        print("  !! SEAM: prefix ends in a space AND tokens carry a leading space")

    messages = build_messages(query_image=str(row.image_path))
    text = rater.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    rendered = text + ASSISTANT_PREFIX

    print("\n" + "=" * 66)
    print("2. RENDERED PROMPT (last 160 chars)")
    print("=" * 66)
    print(repr(rendered[-160:]))

    image = Image.open(row.image_path).convert("RGB")
    inputs = rater.processor(text=[rendered], images=[image], return_tensors="pt").to(args.device)

    with torch.no_grad():
        logits = rater.model(**inputs).logits[0, -1, :].float()
    probabilities = torch.softmax(logits, dim=-1)

    print("\n" + "=" * 66)
    print(f"3. TOP {args.top_k} TOKENS AT THE FIRST COMPLETION POSITION")
    print("=" * 66)
    top = torch.topk(probabilities, args.top_k)
    rating_ids = set(rater.rating_token_ids)
    for probability, token_id in zip(top.values.tolist(), top.indices.tolist()):
        marker = " <-- scale" if token_id in rating_ids else ""
        print(f"  {probability:7.4f}  {token_id:>8}  {tokenizer.decode([token_id])!r}{marker}")

    on_scale = probabilities[list(rater.rating_token_ids)].sum().item()
    print(f"\nmass on the seven scale tokens: {on_scale:.4f}")

    print("\n" + "=" * 66)
    print("4. WHAT THE MODEL ACTUALLY SAYS (greedy, 16 tokens)")
    print("=" * 66)
    with torch.no_grad():
        generated = rater.model.generate(**inputs, max_new_tokens=16, do_sample=False)
    continuation = generated[0][inputs["input_ids"].shape[1]:]
    print(f"  {ASSISTANT_PREFIX}{tokenizer.decode(continuation, skip_special_tokens=True)!r}")
    print("\n  If this reads as a different scale (e.g. '10'), the spike at P(1)")
    print("  is P('1' as the first digit of a two-digit number), not P(rating=1).")


if __name__ == "__main__":
    main()
