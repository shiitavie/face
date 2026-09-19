# Can Multimodal Models Measure Faces, and Do They Impose Neoclassical Canons?

**Status:** Design approved, implementation starting
**Date:** 2026-09-19
**Target venue:** *Aesthetic Surgery Journal* or *Facial Plastic Surgery & Aesthetic
Medicine*; *PRS Global Open* for speed
**Relationship to other work:** Paper 2 of three. Distinct question from the
attractiveness audit ([2026-09-03 spec](2026-09-03-vlm-facial-attractiveness-cav-design.md)),
satisfying ICMJE rules on duplicate publication.

---

## 1. Purpose

Aesthetic surgery is beginning to adopt multimodal models for facial analysis. Two
questions precede any such use, and neither has been answered:

1. **Can these models measure facial features at all?**
2. **When they judge faces, do they apply the neoclassical canons** -- a framework
   derived from Greek and Roman sculpture that is known not to hold outside European
   faces?

The second question only becomes interpretable once the first is settled, which fixes
the paper's structure: establish perceptual capability per measurement, then test canon
adherence only on measurements the model demonstrably perceives.

### 1.1 Why this matters clinically

A facial-analysis tool that silently encodes a Eurocentric template would systematically
describe non-European patients as deviating from an aesthetic norm. That is a concrete
harm in a field where "correction toward ideal proportions" is an operative plan.

### 1.2 Prior evidence motivating Part A

From instrument validation on Qwen2.5-VL-7B (2026-09-19):

- **The numeric channel does not read scales.** Flipping which end of a 1-7 scale means
  "most attractive", with digits and order held fixed, left ratings correlated at
  **+0.982**. Reversing digit order moved the mean **2.1 points on a 7-point scale**.
- **Counterbalanced comparison does work.** Accuracy against CFD human norms rises
  monotonically with pair difficulty to 100%, 81% overall, against 53% (chance) from a
  single presentation order.
- **Confident failure exists and is invisible without ground truth.** `FaceLength`:
  85% order consistency, **52.5% accuracy**. The model reliably picks something that is
  not what CFD measures.

That last result is the template for Part A's most useful finding.

---

## 2. Data

CFD 3.0 as in the companion spec: 831 neutral images, 826 joined to norming rows.
**60 objective physical measurements** ship with it, which is what makes both parts
possible without new annotation.

### 2.1 Measurements under test (Part A)

Chosen to span easy to hard, and to cover the structures aesthetic surgery operates on:

| measurement | CFD | clinical relevance |
|---|---|---|
| Nose width | `NoseWidth` | rhinoplasty, alar base |
| Intercanthal distance | `EyeDistance` | canon anchor |
| Eye width | `EyeWidthAvg` | periorbital |
| Face width | `FaceWidthBZ` | facial proportion |
| Lip thickness | `LipThickness` | lip augmentation |
| Chin length | `ChinLength` | genioplasty |
| Eyebrow thickness | `EyeBrowThicknessAvg` | brow position |
| Facial asymmetry | `PupilLipAsymmetry` | symmetry, a core aesthetic construct |

### 2.2 Canons computable from CFD (Part B)

| canon | operationalization |
|---|---|
| Facial fifths | `FaceWidthBZ / EyeWidthAvg`, ideal 5.0 |
| Intercanthal equals eye width | `EyeDistance / EyeWidthAvg`, ideal 1.0 |
| Nasal width equals intercanthal | `NoseWidth / EyeDistance`, ideal 1.0 |
| Facial thirds | equality of `Forehead`, `MidfaceLength`, `ChinLength` |
| Facial width-to-height | `fWHR2` |
| Symmetry | `PupilTopAsymmetry`, `PupilLipAsymmetry`, ideal 0 |

Per-face **canon deviation** is the summed absolute log-ratio of each index against its
ideal, so over- and under-shooting are penalized symmetrically and the measure is
scale-free.

---

## 3. Part A -- measurement capability

### 3.1 Value-estimation channel

Ask for scale-free **ratios**, not millimetres: CFD measures in image pixels, so absolute
units have no ground truth, and ratios are what the canons are expressed in anyway.

Prompt form: *"What is the ratio of this person's nose width to the distance between
their eyes?"*

Metrics: Spearman correlation against the true ratio; calibration slope; and the spread
of responses, since a model emitting a near-constant value regardless of input is the
expected failure mode.

**Prior expectation: this channel fails.** A model that ignores the stated direction of a
seven-point scale is unlikely to produce calibrated ratios. A clean negative here is a
publishable result, and is more useful to the field than a marginal positive.

### 3.2 Comparison channel

Counterbalanced pairwise comparison, the validated instrument. **Counterbalancing in
log-odds is mandatory** -- position bias is additive in log-odds, and averaging
probabilities collapses every comparison toward 0.5 once the bias saturates (observed:
13 of 20 pairs inside 0.45-0.55 before the fix, 2 of 20 after).

For each measurement: sample pairs stratified across the true gap, score both orders,
report accuracy per gap bin.

### 3.3 Capability classification

Each measurement is labelled from the comparison channel:

- **perceived** -- accuracy rises with gap and exceeds 75% in the top bins
- **confidently wrong** -- high order consistency, accuracy near chance (the `FaceLength`
  pattern)
- **absent** -- neither consistent nor accurate

Only **perceived** measurements carry forward to Part B.

---

## 4. Part B -- canon adherence

1. **Compute canon deviation per face** from CFD measurements.
2. **Test whether deviation differs by race.** This confirms existing literature (Farkas
   and others showed the neoclassical canons rarely hold outside European faces) and must
   be presented as setup, not as a novel finding.
3. **Test whether the model's aesthetic judgment tracks canon adherence**, using the
   Bradley-Terry attractiveness scale from the companion pilot. Then test whether that
   relationship **differs by group** -- the mechanism claim.

Step 3 is the paper's contribution. It is only interpretable for measurements Part A
classifies as perceived; a canon built on a measurement the model cannot see tells you
nothing about the model.

---

## 5. Reuse and new work

Carried over unchanged: CFD manifest, `VLMRater`, log-odds counterbalancing,
Bradley-Terry, pair sampling.

New: canon index computation (pure arithmetic, unit-tested), a value-estimation probe,
and the measurement comparison battery.

**Compute:** ~8 measurements x 200 pairs x 2 orders = ~3,200 forward passes, one to two
hours. Far smaller than the attractiveness pilot.

---

## 6. Dependencies and risks

- **Part B step 3 needs the attractiveness pilot's Bradley-Terry scale.** The two papers
  are not fully independent; this one cites the pilot's instrument validation rather than
  repeating it.
- **If Part A finds only two or three measurements perceived**, Part B narrows and may be
  too thin to carry a canon argument. In that case the paper becomes Part A alone, which
  still stands as a capability-and-caution result.
- **Single model.** Qwen2.5-VL-7B alone cannot support a claim about "multimodal models".
  At least two more open-weight models are required before submission.

---

## 7. Open questions

1. Exact ratio phrasings for the value-estimation probe; several will be piloted and one
   frozen before the main run.
2. Whether to include CFD-MR and CFD-INDIA in Part B's group comparison, or to restrict
   to CFD main's four categories for cleaner cells.
