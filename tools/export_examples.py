"""Regenerate explorer-data.js (the embedded examples for the "Explore the data" panel).

Usage:  pip install datasets pillow && huggingface-cli login
        python tools/export_examples.py > explorer-data.js

Edit MOD / TASK / TEST below to change which rows are shown. Row indices refer to the
current HF revision of neginb/OpenMedReason; labels are assigned by hand.
"""
import base64, io, json, re, sys
from datasets import load_dataset
from PIL import Image

THUMB_PX, QUALITY = 520, 78
MOD = [('CT', 5009), ('MRI', 5012), ('Ultrasound', 20002), ('X-ray', 5077), ('Endoscopy', 20049), ('Dermatology', 75), ('OCT / OCTA', 60035), ('Intraoperative photo', 5090), ('Histology', 37004), ('Immunofluorescence', 80030), ('Electron microscopy', 80036)]
TASK = [("Perception", "Findings / description", 74067, "Identify the main visible finding or visual pattern."),
        ("Perception", "Anatomy / localization", 56, "Identify the depicted structure, tissue, or organ."),
        ("Perception", "Normal vs. abnormal", 75, "Decide whether the appearance is normal or abnormal."),
        ("Perception", "Annotation / marker", 140075, "Interpret a visual marker (arrow, box, label, etc.)."),
        ("Perception", "Spatial location", 9, "Locate a finding within the image or anatomy."),
        ("Perception", "Counting", 69, "Count visible structures, lesions, or cells."),
        ("Diagnosis", "Diagnosis", 111066, "Infer the most likely diagnosis from visual and clinical cues."),
        ("Diagnosis", "Mechanism / pathophysiology", 37019, "Explain the mechanism underlying the finding."),
        ("Diagnosis", "Differential diagnosis", 5057, "Distinguish the correct diagnosis from plausible alternatives."),
        ("Diagnosis", "Severity grading", 20039, "Assess the grade, stage, or extent of the condition."),
        ("Management", "Next-step management", 111028, "Choose the appropriate management or intervention."),
        ("Management", "Surgical management", 5070, "Reason about a surgical approach or intraoperative decision."),
        ("Management", "Drug therapy", 125098, "Identify the relevant medication or therapeutic class."),
        ("Risk", "Complications", 111087, "Identify an associated complication or adverse event."),
        ("Risk", "Prognosis", 90056, "Infer the expected outcome or disease course."),
        ("Risk", "Symptoms / signs", 90088, "Link the finding to associated symptoms or signs."),
        ("Risk", "Safety / contraindications", 125024, "Flag safety concerns or contraindications."),
        ("Risk", "Hereditary risk", 74043, "Reason about inherited risk or genetic association."),
        ("Workup", "Next-step diagnostic test", 5005, "Choose the appropriate follow-up test or evaluation.")]
TEST = [(2, "Diagnosis · pelvic radiograph"), (3, "Long stem · chest radiograph"),
        (7, "Annotation / marker · chest radiograph"), (85, "Hereditary risk · hair microscopy")]

ds = load_dataset("neginb/OpenMedReason")

def parse_q(q):
    q = q.strip(); m = re.search(r"\n\s*A[\.\)]\s", q)
    stem = q[:m.start()].strip() if m else q
    opts = re.findall(r"(?m)^\s*([A-F])[\.\)]\s*(.+?)\s*$", q[m.start():]) if m else []
    return stem, [{"k": k, "t": t} for k, t in opts]

def clean_trace(t):
    t = re.sub(r"</?think>", "", t); t = re.sub(r"<answer>.*?</answer>", "", t, flags=re.S)
    return [p.strip() for p in t.strip().split("\n") if p.strip()]

def thumb(img):
    img = img.convert("RGB"); w, h = img.size; img.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=QUALITY)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode(), w, h

ex = {}
def add(split, i):
    key = f"{split}-{i}"
    if key in ex: return key
    r = ds[split][i]; stem, opts = parse_q(r["question"]); img, w, h = thumb(r["image"])
    ex[key] = {"split": split, "idx": i, "stem": stem, "options": opts, "answer": r["answer"],
               "trace": clean_trace(r["reasoning"]), "img": img, "w": w, "h": h}
    if split == "test":
        ex[key]["rubric"] = {a: r[a] for a in ("perception", "knowledge", "rationale")}
    return key

data = {"modalities": [{"label": l, "key": add("train", i)} for l, i in MOD],
        "tasks": [{"family": f, "label": l, "objective": o, "key": add("train", i)} for f, l, i, o in TASK],
        "tests": [{"label": l, "key": add("test", i)} for i, l in TEST],
        "examples": ex}
sys.stdout.write("// OpenMedReason project page: hand-picked examples from huggingface.co/datasets/neginb/OpenMedReason (train + test splits).\n"
                 "// Images are downscaled WebP thumbnails embedded as data URIs. Regenerate with tools/export_examples.py.\n"
                 "window.OMR_EXAMPLES = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
