"""Build the PI-facing Word summary from the current state of the project.

    .venv/bin/python docs/manuscripts/build_summary.py

Kept as a script rather than typed inline, because the findings table changes
every time a battery runs and the document has to be regenerated, not edited.
"""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

NAVY = RGBColor(0x1F, 0x36, 0x4D)
OUT = "docs/manuscripts/Cautionary_Paper_Methods_Summary.docx"

doc = Document()
for section in doc.sections:
    section.top_margin = section.bottom_margin = Inches(1)
    section.left_margin = section.right_margin = Inches(1.1)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.15


def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = NAVY


def para(text, italic=False, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic, run.bold = italic, bold


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def table(rows, widths):
    t = doc.add_table(rows=0, cols=len(widths))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.paragraphs[0].text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(9.5)
            if i == 0:
                run.bold = True
    for row in t.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)
    doc.add_paragraph()


title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Are Prompted Multimodal Model Aesthetic Ratings a Measurement?")
run.bold = True
run.font.size = Pt(16)
run.font.color.rgb = NAVY

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("A reliability audit of Likert-scale facial ratings, with implications "
                  "for aesthetic outcome assessment")
run.italic = True
run.font.size = Pt(12)

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = meta.add_run("Methods summary · working draft · 20 September 2026")
run.font.size = Pt(9.5)
run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

heading("1. The question")
para("Multimodal large language models are increasingly used to score facial photographs "
     "on Likert scales by prompting alone — no model training, no calibration. Reports "
     "already quote figures of the form “the model rated our post-operative results "
     "8.2/10.”")
para("Such use assumes the number behaves as a measurement: that it reflects the image, "
     "that it responds to the scale as defined, and that repeating the query returns "
     "approximately the same value. To our knowledge none of these assumptions has been "
     "tested.")
para("We treat the rating as an instrument rather than as data, and subject it to the "
     "validation any instrument requires before clinical use.", bold=True)

heading("2. Relationship to existing work in aesthetic surgery", 2)
para("AI-derived aesthetic scoring has already entered the surgical outcomes literature. "
     "Rames et al. applied trained ensemble models for perceived age and perceived "
     "attractiveness to 676 patients from the ASPS Before and After gallery, deriving a "
     "composite aesthetic benefit score and using it to compare procedures and identify "
     "patient factors associated with greater benefit (Plast Reconstr Surg, "
     "doi:10.1097/PRS.0000000000013412). That work establishes both the clinical appetite "
     "for objective aesthetic outcome measurement and the feasibility of applying computer "
     "vision at scale to surgical photographs.")
para("Our scope is complementary. Rames et al. use purpose-trained supervised ensembles, "
     "fit to human ratings and validated against them. We examine general-purpose models "
     "prompted to produce ratings directly — the tool a clinician reaches for without "
     "training anything. These are different instruments with different failure modes: a "
     "supervised regressor is calibrated to its training distribution by construction, "
     "whereas a prompted model's output depends on how the question is worded.")
para("This study asks what validation a prompted model requires before it is used the way "
     "trained models are now being used. Nothing here bears on the validity of "
     "purpose-trained models.", bold=True)

heading("3. Preliminary findings")
para("Two models so far — one open-weight (Qwen2.5-VL-7B) and one commercial "
     "(Claude Opus 5) — on the Chicago Face Database. They fail in opposite ways.",
     italic=True)
table([
    ("Test", "Qwen2.5-VL-7B", "Claude Opus 5"),
    ("Scale direction — flipping which end means “most attractive”, digits unchanged",
     "ρ = +0.98 (should be strongly negative)", "pending"),
    ("Digit anchoring — reversing digit order, meaning unchanged",
     "mean shifts 2.2 points on a 7-point scale; ratings become uncorrelated with the "
     "photograph (ρ = 0.00)", "pending"),
    ("Repeatability at a fixed prompt",
     "SE ≈ 0.44 per image, larger than the true spread between faces (SD 0.27)",
     "near-deterministic (SD 0.02)"),
    ("Scale use",
     "uses the full range", "75% of all ratings are the midpoint; only 3 of 7 points "
     "ever used"),
    ("Presentation order in paired comparison",
     "chooses the second image in ~99% of trials", "position bias present; under "
     "measurement"),
], [1.7, 2.3, 2.3])
para("Qwen is imprecise and driven by the surface form of the question. Claude is "
     "perfectly repeatable but barely discriminates. Both are unusable as measurements, "
     "for opposite reasons — which is why validating the specific model and prompt in "
     "use is not optional.", bold=True)

heading("4. Methods in brief")
bullet("Chicago Face Database, neutral expression, 831 photographs. Standardized "
       "lighting, camera and framing, so group differences cannot be confounded with how "
       "groups were photographed. Includes self-identified demographics, human norming "
       "data, and 60 objective facial measurements used as ground truth.")
bullet("Ratings read exactly from token probabilities where model weights allow, and "
       "estimated from repeated sampling where they do not. The two readouts are compared "
       "head-to-head on a model permitting both (ρ = 0.88), which is what licenses the "
       "sampled method on commercial APIs.")
bullet("Four tests, each isolating one way the number could be produced by something "
       "other than the photograph: scale semantics, digit anchoring, paraphrase "
       "robustness, and refusal rate including refusal by demographic group.")
bullet("Precision is decomposed into true between-face variation and sampling error, and "
       "correlations are reported both raw and corrected for attenuation.")
bullet("Where absolute rating fails, counterbalanced forced-choice comparison is "
       "evaluated as the alternative — the standard psychometric remedy for "
       "anchor-dependent scales, and structurally the same judgement the Global Aesthetic "
       "Improvement Scale asks for.")

heading("5. What remains")
bullet("Additional models, including further commercial systems, before any claim about "
       "multimodal models as a class.")
bullet("Whether these failures persist at frontier scale — the first question a reviewer "
       "will ask.")
bullet("A scale-format arm: verbally anchored and categorical scales, to test whether the "
       "formats used by validated clinical instruments avoid the failures seen with bare "
       "numeric scales.")

heading("6. Positioning")
para("The contribution is a validation protocol and a caution, not a benchmark. The paper "
     "reports what fails, why the failure is invisible in ordinary use, and a "
     "counterbalanced comparison procedure that works where direct rating does not.")
para("Candidate venues: Aesthetic Surgery Journal; Facial Plastic Surgery & Aesthetic "
     "Medicine; Plastic and Reconstructive Surgery – Global Open. No patient data and no "
     "IRB requirement; all stimuli are from a public de-identified research database. "
     "Code and analysis are public at github.com/shiitavie/face.")

doc.save(OUT)
print(f"wrote {OUT}")
