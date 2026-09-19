# Auditing MLLM facial attractiveness judgments

**This is a bias-audit research project. It is not, and is not intended to be, a
tool for rating people's appearance.**

Multimodal large language models are increasingly proposed for aesthetic and
clinical image assessment. This repository asks what visual features actually
drive those judgments, and whether demographic information is encoded along the
same internal directions — moving from *"the model is biased"* to *"here is the
representation through which the bias operates."*

The method extends [Visual Concept Ranking](https://arxiv.org/abs/2602.05096)
(Janizek et al., 2026) from binary classification to ordinal rating tasks. VCR
in turn builds on [LG-CAV](https://arxiv.org/abs/2410.10308) and
[MONET](https://pubmed.ncbi.nlm.nih.gov/38627560/).

## Status

Work in progress. Stage 1a (behavioral rating) is implemented and tested; the
concept-activation stages are not yet built. No results have been produced, and
nothing here should be cited as a finding.

Design: [`docs/superpowers/specs/`](docs/superpowers/specs/).

## A note on the construct

Attractiveness has no ground truth. Human ratings are used throughout as a
*comparator*, never as truth, and disagreement with human raters is not error.
The design deliberately separates descriptive findings (the model's rating
depends on lip fullness) from normative ones (the model's rating depends on skin
tone), because only the latter are claims about bias. See §7.4 of the spec.

## Data

**No image data is included in this repository, and none should ever be
committed.** The [Chicago Face Database](https://www.chicagofaces.org/) requires
registration and its own license agreement; obtain it directly from the authors.
`.gitignore` blocks image files as a safeguard.

## Running it

```bash
pip install -e ".[dev]"
pytest tests/ -q

python experiments/stage1a_rate_cfd.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --cfd-root "/path/to/CFD Version 3.0"
```

Torch is deliberately not a declared dependency — install it matched to your
CUDA version. See [`notebooks/stage1a_colab.ipynb`](notebooks/stage1a_colab.ipynb)
to run on Colab.

Do not quantize the models. The dependent variable is the logit distribution
over the rating tokens, and quantization perturbs exactly that.
