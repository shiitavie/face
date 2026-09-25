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
OUT = "docs/manuscripts/MLLM_Facial_Rating_Validation_Summary.docx"

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
run = title.add_run("Validating Multimodal Models as Raters of Facial Aesthetics")
run.bold = True
run.font.size = Pt(16)
run.font.color.rgb = NAVY

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("Reliability, validity, and demographic fairness of prompted "
                  "attractiveness ratings")
run.italic = True
run.font.size = Pt(12)

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = meta.add_run("Preliminary results · working draft · 25 September 2026")
run.font.size = Pt(9.5)
run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

heading("1. The question")
para("Multimodal large language models are increasingly used to score facial photographs "
     "on Likert scales by prompting alone — no model training, no calibration. Before such "
     "a score can inform clinical judgement, three things must be true: it must repeat, it "
     "must respond to the scale as defined, and it must agree with human judgement. A "
     "fourth follows immediately in a surgical context — it must do so equally for every "
     "patient group.")
para("We treat the rating as an instrument and validate it accordingly. The finding is not "
     "that these models fail, but that they differ enormously: one of the two tested is "
     "unusable, and the other performs at the level of human rater agreement.", bold=True)

heading("2. Relationship to existing work in aesthetic surgery", 2)
para("AI-derived aesthetic scoring has already entered the surgical outcomes literature. "
     "Rames et al. applied trained ensemble models for perceived age and attractiveness to "
     "676 patients from the ASPS Before and After gallery (Plast Reconstr Surg, "
     "doi:10.1097/PRS.0000000000013412). Varghaei et al. built a landmark-geometric "
     "pipeline over 7,160 photographs from 1,259 patients (arXiv:2508.13363), reporting "
     "nasal ratios — including alar width to intercanthal distance — as outcome measures.")
para("Those are purpose-built instruments: a supervised regressor is calibrated to its "
     "training distribution by construction, and a geometric pipeline measures what it is "
     "told to measure. We examine general-purpose models prompted to produce ratings "
     "directly — the tool a clinician reaches for without training anything. Our scope is "
     "complementary, and nothing here bears on the validity of purpose-built models.")
para("Notably, neither prior work could assess demographic fairness: Varghaei et al. state "
     "explicitly that their dataset carries no subject-level demographic annotations. The "
     "Chicago Face Database does, which is what makes the present analysis possible.")

heading("3. Preliminary findings")
para("Two models on the Chicago Face Database: one open-weight (Qwen2.5-VL-7B) and one "
     "commercial (Claude Opus 5). Fairness figures are from 276 of 826 faces; the "
     "remainder are pending.", italic=True)

heading("3.1 The two models are not comparable instruments", 2)
table([
    ("", "Qwen2.5-VL-7B", "Claude Opus 5"),
    ("Repeats on re-query?",
     "No — SE ≈ 0.44 per face, larger than the true spread between faces (SD 0.27)",
     "Yes — SD 0.02, effectively deterministic"),
    ("Responds to the scale's meaning?",
     "No — flipping which end means “most attractive” leaves ratings almost unchanged "
     "(ρ = +0.98)", "Yes"),
    ("Sensitive to irrelevant wording?",
     "Severely — reversing digit order moves the mean 2.2 points and decorrelates ratings "
     "from the photograph entirely (ρ = 0.00)", "No"),
    ("Agrees with human raters?",
     "Not assessed — the instrument fails first",
     "ρ = +0.57 to +0.63 within demographic cells"),
], [1.55, 2.4, 2.35])
para("An investigator using the open-weight model would obtain confident numbers driven by "
     "the wording of the question rather than by the photograph, with no indication from "
     "the output that anything was wrong. This is why the validation step is not optional.",
     bold=True)

heading("3.2 The commercial model performs at human level", 2)
para("Agreement with CFD's human attractiveness norms, computed within race × gender cells "
     "because CFD's principal item asks raters to judge each face relative to others of "
     "the same race and gender:")
table([
    ("Group", "n", "Agreement with human norms (95% CI)"),
    ("Black", "55", "+0.63  [+0.44, +0.77]"),
    ("White", "48", "+0.62  [+0.37, +0.79]"),
    ("Multiracial", "19", "+0.62  [+0.19, +0.88]"),
    ("Asian", "97", "+0.60  [+0.46, +0.71]"),
    ("Indian", "35", "+0.44  [+0.12, +0.68]"),
    ("Latino", "22", "+0.28  [−0.19, +0.68]"),
], [1.5, 0.7, 4.1])
para("For reference, two human rater pools (US and Indian) agree with each other at ρ = "
     "0.585 on the same construct. The model's agreement with human raters is of the same "
     "order as human agreement with one another.")

heading("3.3 No demographic bias detected — with the power caveat", 2)
bullet("Differential validity: all six confidence intervals overlap. There is no evidence "
       "the model tracks human judgement better for one race group than another. The two "
       "lowest estimates are also the two smallest groups (n = 22 and 19), so this is a "
       "null with limited power rather than a demonstration of fairness.")
bullet("Colorism: with race and gender held fixed, lighter skin is associated with higher "
       "ratings at ρ = +0.10 for the model and ρ = +0.12 for human raters on the identical "
       "test. The model does not amplify the human effect.")
bullet("Scale use: rating diversity ranges 0.38–0.59 across groups — no group is collapsed "
       "to a single value, so all remain internally rankable.")
bullet("Absolute mean ratings differ across groups by up to 0.6 scale points, but CFD "
       "supplies no absolute human baseline for the main set. That difference cannot be "
       "attributed to the model rather than to the faces or the photographs, and is not "
       "reported as bias.")

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
bullet("Completing the fairness run: 550 of 826 faces remain, roughly $4 of API credit.")
bullet("Additional models, including a second commercial system, before any claim about "
       "multimodal models as a class.")
bullet("A scale-format arm: verbally anchored and categorical scales of the kind validated "
       "clinical instruments use, to test whether those formats avoid the failures seen "
       "with bare numeric scales.")
bullet("Replication on surgical photographs, which requires IRB and institutional data.")

heading("6. Positioning")
para("The contribution is a validation protocol, a demonstration that models differ "
     "enough for it to matter, and the first demographic fairness audit of a prompted "
     "multimodal model on standardized facial photographs. The message to clinicians is "
     "not that these tools cannot be used, but that the specific model and prompt must be "
     "validated before they are.")
para("Candidate venues: Aesthetic Surgery Journal; Facial Plastic Surgery & Aesthetic "
     "Medicine; Plastic and Reconstructive Surgery – Global Open. No patient data and no "
     "IRB requirement; all stimuli are from a public de-identified research database. "
     "Code and analysis are public at github.com/shiitavie/face.")

doc.save(OUT)
print(f"wrote {OUT}")
