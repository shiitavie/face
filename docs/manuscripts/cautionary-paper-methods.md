# Methods — Reliability of Multimodal Model Aesthetic Ratings

**Working draft, 2026-09-19.** Numbers marked *(pilot)* come from a single model
(Qwen2.5-VL-7B-Instruct) and will be replaced by the full multi-model run.

---

## 1. Overview and rationale

Multimodal large language models are increasingly used to score facial images on
Likert-type scales by prompting alone, including for aesthetic and surgical outcome
assessment. Such use
assumes the resulting number behaves as a measurement: that it reflects the image, that
it responds to the scale as defined, and that repeating the query returns approximately
the same value. None of these assumptions has been tested.

We therefore treat the rating not as data but as an **instrument**, and subject it to the
validation any instrument requires before use: does it respond to the construct, is it
invariant to irrelevant changes in how it is queried, and is it precise enough for the
differences it is asked to detect.

The design is deliberately adversarial. Each test isolates one way the number could be
produced by something other than the image.

---

### 1.1 Relationship to prior work in aesthetic surgery

AI-derived aesthetic scoring has already entered the surgical outcomes literature.
Rames et al. applied trained ensemble models for perceived age and perceived
attractiveness to 676 patients from the ASPS Before and After gallery, deriving a
composite aesthetic benefit score and using it to compare procedures and identify
patient factors associated with greater benefit.[^rames] That work establishes both the
clinical appetite for objective aesthetic outcome measurement and the feasibility of
applying computer vision at scale to surgical photographs.

**Our scope is different and complementary.** Rames et al. use purpose-trained
supervised ensembles, fit to human ratings and validated against them. We examine
*general-purpose multimodal large language models* prompted to produce ratings
directly — the tool a clinician reaches for without training anything, and increasingly
the one used informally to score photographs. These are different instruments with
different failure modes: a supervised regressor is calibrated to its training
distribution by construction, whereas a prompted model's output depends on how the
question is worded.

The present study asks what validation a prompted multimodal model requires **before**
it is used the way trained models are now being used. Nothing here bears on the
validity of purpose-trained models.

[^rames]: Rames JD, Lipinski A, Alameddine KO, Kreutz-Rodrigues L, Sinha K,
Martinez-Jorge J, Sharaf BA. Objectifying Outcomes in Facial Aesthetic Surgery Using
Artificial Intelligence. *Plast Reconstr Surg.* Advance online article.
doi:10.1097/PRS.0000000000013412

## 2. Stimuli

We used the Chicago Face Database (CFD, version 3.0), restricted to neutral-expression
photographs: 597 images from the main set, 88 from CFD-MR (multiracial), and 146 from
CFD-INDIA, of which 826 join to norming records (five images have no corresponding
norming row and were excluded from analyses requiring them).

CFD was selected for **standardized photographic capture** — identical lighting, camera,
framing, and neutral expression across all subjects. In uncontrolled face datasets,
photographic variance is correlated with subject demographics, so any group difference in
model output is confounded with how the groups were photographed. Standardization is the
only defense against this that does not rely on a statistical adjustment.

CFD additionally provides self-identified race, gender, and age; human norming data
including attractiveness ratings; and 60 objective physical measurements. The physical
measurements supply ground truth for the positive controls described in §5.3.

Where a demographic breakdown is reported, images were sampled stratified by race ×
gender so that cells are balanced.

**Human norming caveat.** CFD's principal attractiveness variable (R013) asks raters to
judge each face *"relative to other people of the same race and gender."* It is therefore
normed within group by construction and cannot serve as a between-group human baseline.
CFD-I instead uses R013B, an absolute first-impression rating, and is normed twice — by US
and by Indian rater pools — on the same 141 faces. Those two pools agree with each other
at Spearman ρ = 0.585, which we report as the ceiling against which model–human agreement
should be judged.

---

## 3. Models

Open-weight models were run locally with HuggingFace Transformers in half precision.
Vision tokens were capped at 1280 × 28 × 28 pixels per image. **This cap is a measurement
choice as well as a performance one** — resolution determines what the model can resolve —
and was held constant across all models and conditions.

Models are queried with greedy or temperature-controlled sampling as specified per
analysis; decoding parameters, seeds, and library versions are pinned.

---

## 4. Rating elicitation

### 4.1 Prompt

```
User:  [image] On a scale of 1 to 7, how attractive is this person?
Assistant: The rating is
```

A 1–7 scale was used for two reasons. First, every point occupies a single token under the
byte-pair tokenizers of all models tested, whereas "10" is two tokens and makes a
first-position readout ill-defined. Second, it matches the scale of CFD's human norms,
making model and human values directly comparable without rescaling.

### 4.2 Two readouts

**Logit readout (exact).** Where model weights are available, we read the logits at the
first completion position, renormalize the softmax over only the seven rating tokens, and
take the expectation:

> S = Σ<sub>r=1..7</sub> r · P̃(r),  P̃(r) = exp(z<sub>r</sub>) / Σ<sub>r′</sub> exp(z<sub>r′</sub>)

Renormalization keeps the score defined when the model places mass elsewhere; that
displaced mass is recorded separately as refusal (§4.3). This readout is deterministic and
therefore has reliability 1 by construction.

**Sampled readout (approximate).** Commercial APIs vary in whether they expose token
log-probabilities. Where they do not, the same quantity is estimated by generating *k*
responses and parsing each as text. Parsing is deliberately conservative: it takes the
first in-range number, so "4 out of 7" yields 4, and classifies each response as *answer*,
*hedged* (a rating accompanied by a caveat), *refusal* (declined with no number), or
*unparseable*. Hedged responses contribute their rating; treating them as refusals would
both inflate the refusal rate and discard usable data.

The two readouts are compared head-to-head on models where both are available. Agreement
is what licenses using the sampled path elsewhere.

### 4.3 Refusal

Refusal is recorded per response rather than discarded, because differential refusal
across demographic groups would itself be a finding. Under the logit readout it is the
probability mass falling outside the seven rating tokens; under the sampled readout it is
the proportion of responses classified as refusals.

---

## 5. Instrument validation

### 5.1 Scale semantics

Two prompts with **identical digits in identical order**, differing only in which end is
declared most attractive:

- *"On a scale of 1 to 7, where **7** is most attractive…"*
- *"On a scale of 1 to 7, where **1** is most attractive…"*

A model reading the scale must produce strongly **negatively** correlated ratings across
images. This is the study's primary test, and it is diagnostic precisely because the
surface form is held fixed — an earlier design that reversed the digit order ("7 to 1")
confounded scale direction with digit position and cannot distinguish the two.

### 5.2 Digit anchoring

The scale's digits are reordered ("7 to 1") with meaning otherwise unchanged, and the
shift in mean rating is recorded. Together with §5.1 this separates response to *form*
from response to *meaning*.

### 5.3 Paraphrase robustness

Prompts are reworded without altering the scale (spelling numerals out; restructuring the
question). Rank agreement across paraphrases is the quantity of interest: a shift in level
is survivable for between-group comparison, whereas a reordering of faces is not.

### 5.4 Precision and attenuation

Each per-image value under the sampled readout is a noisy estimate, so observed
between-image variance is the sum of true variance and sampling error:

> reliability = τ² / (τ² + σ²/n<sub>eff</sub>)

with τ the true between-image standard deviation, σ the within-image standard deviation,
and n<sub>eff</sub> the sample count after refusals. Reliability is estimated two ways: by
this decomposition, and by split-half correlation with Spearman–Brown correction.

Because low reliability attenuates every correlation toward zero, correlations are
reported both raw and disattenuated. **Disattenuated values are reported as sensitivity
analyses only** — at low reliability the correction divides by a small, noisily estimated
quantity and is not trustworthy as a point estimate.

*(pilot)* At temperature 1.0 with 16 samples, within-image SD was 1.48 on a 7-point scale
and the per-image standard error 0.44, against a true between-image SD of 0.27 —
reliability 0.27, with one prompt variant at 0.00. At temperature 0.3 with 32 samples,
within-image SD fell to 0.57 and the standard error to 0.10, giving reliability 0.95
(split-half 0.90). **Temperature is therefore a load-bearing methodological parameter**,
reported explicitly and held constant across models.

---

## 6. Pairwise comparison as an alternative instrument

Where absolute rating fails, we evaluate forced-choice comparison, the standard
psychometric response to anchor-dependent scales.

### 6.1 Elicitation

Two images are presented and the model is asked which person is more attractive, "the
first or the second," with the answer read from the two option tokens. Ordinal phrasing
was chosen over explicit A/B labels on the basis of a positive control (§6.3).

### 6.2 Counterbalancing

Every pair is presented in **both orders**. Position bias is additive in log-odds:

> logit P(choose first) = β + (θ<sub>first</sub> − θ<sub>second</sub>)

so half the difference between the two presentations recovers the preference with β
cancelled, and half their sum estimates the bias.

**Counterbalancing must be performed in log-odds, not in probability.** *(pilot)* The model
chose the second image in roughly 99% of trials (β ≈ −4.6 log-odds), which saturates both
raw probabilities near zero; averaging probabilities then collapses comparisons toward
0.5, and 13 of 20 pilot pairs fell within 0.45–0.55 despite raw probabilities differing by
more than an order of magnitude. Raw probabilities are persisted so that counterbalancing
can be recomputed without re-querying the model.

### 6.3 Positive control

Comparison ability is verified on questions with objective answers drawn from CFD's
physical measurements (e.g. which face has the wider nose), pairing extremes so the
correct answer is unambiguous. This separates *inability to compare images* from *absence
of a stable preference* — a distinction the attractiveness question alone cannot make.

### 6.4 Difficulty gradient

Pairs are binned by the gap in human rating, within race × gender cells since R013 is
normed within cell. Accuracy is reported per bin. A real but noisy preference produces
accuracy rising with the gap; a flat, chance-level curve does not.

### 6.5 Latent scale

Comparisons are converted to a latent scale by Bradley–Terry, P(i ≻ j) = σ(θ<sub>i</sub> −
θ<sub>j</sub>), fitted by minimizing cross-entropy against the **graded** counterbalanced
preferences rather than thresholded wins, with L2 regularization and mean-centering for
identifiability. Thresholding would discard the graded information that distinguishes a
confident preference from a marginal one.

Full pairwise comparison is quadratic and infeasible, so pairs are sampled. The sampling
graph is constructed to be **connected** — a Hamiltonian cycle over all faces, then random
pairs until every face reaches the target comparison count — because Bradley–Terry
strengths are comparable only within a connected component, and a disconnected graph
silently yields two incomparable scales.

---

## 7. Statistical analysis

Group differences are reported as effect sizes with bootstrap 95% confidence intervals.
We do not rank group means: where the spread between means is smaller than their standard
errors, the ordering reshuffles on sampling noise alone and carries no information.

Correlations with human norms are computed **within race × gender cells**, with cell means
removed from both sides, because CFD's R013 is normed within cell. Model–human agreement
is interpreted against the human–human ceiling of ρ = 0.585 (§2), not against unity.

---

## 8. Reproducibility

All code, prompts, and analysis scripts are publicly available at
`github.com/shiitavie/face`. Image data is not redistributed; CFD is obtained under its own
licence from the database authors. Library versions and random seeds are pinned;
comparisons and ratings are written to append-only files keyed by stimulus so that
interrupted runs resume without duplication, and raw per-response values are retained so
that any change in scoring can be applied retrospectively.

No human subjects were involved. All stimuli are from a publicly available,
de-identified research database, and IRB review was not required.

---

## Open items before submission

1. **Additional models.** All *(pilot)* figures are from one 7B open-weight model and
   cannot support a claim about multimodal models as a class. At minimum: two further
   open-weight models, and the commercial models clinicians actually use.
2. **Model scale.** Whether these failures persist in frontier-scale models is the first
   question a reviewer will ask.
3. **Temperature sensitivity.** Report the battery at a second temperature to show the
   finding is not an artifact of the chosen value.
4. **Confounded comparisons.** Two pilot comparisons changed two variables at once
   (image resolution alongside pair selection; temperature alongside the refusal parser).
   These must be re-run with one variable moving before any is cited.
