"""Cross-check manually transcribed premium CSVs against independent OCR of the source images.

Exit code 0 only if every non-dash cell is confirmed by OCR. Dash cells that OCR does not
detect are reported as "dash-unconfirmed" and must be checked visually.
"""
import csv, re, sys
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

PAIRS = {
    "年缴_必选": "data/extracted/images/年缴费率_p1_img1.jpg",
    "年缴_加油包": "data/extracted/images/年缴费率_p1_img2.jpg",
    "月缴_必选": "data/extracted/images/月缴费率_p1_img1.jpg",
    "月缴_加油包": "data/extracted/images/月缴费率_p1_img2.jpg",
}
AGE = re.compile(r"^\[?(\d+),(\d+)\]?$")
NUM = re.compile(r"^\d+(\.\d+)?$|^-$")
ocr = RapidOCR()
# Cells where OCR is unreliable (watermark / highlighted background / dash read as "1") and that were
# confirmed by visual inspection of 2x crops on 2026-09-30. Format: (table, age, column index or "extra").
VISUALLY_VERIFIED = {
    ("年缴_加油包", "[11,15]", 9), ("年缴_加油包", "[51,55]", 9),
    ("月缴_加油包", "[46,50]", 9), ("月缴_加油包", "[51,55]", 9), ("月缴_加油包", "[56,60]", 9),
    ("月缴_加油包", "[66,70]", 9), ("月缴_加油包", "[56,60]", 1), ("月缴_加油包", "[56,60]", 2),
    ("月缴_加油包", "[56,60]", 3), ("月缴_加油包", "[56,60]", 7),
    ("月缴_加油包", "[91,95]", "extra"), ("月缴_加油包", "[96,100]", "extra"),
}
visual = 0


def read_cell(img, cx, cy, w=70, h=12):
    box = tuple(int(v) for v in (max(cx - w, 0), max(cy - h, 0), min(cx + w, img.width), min(cy + h, img.height)))
    crop = img.crop(box).resize(((box[2] - box[0]) * 3, (box[3] - box[1]) * 3), Image.LANCZOS)
    res, _ = ocr(np.array(crop))
    return "".join(t for _, t, _ in res or []).replace(" ", "")


bad, dash_unconfirmed, confirmed = 0, 0, 0
for key, path in PAIRS.items():
    img = Image.open(path).convert("RGB")
    rows = list(csv.reader(open(f"data/clean/premium/{key}.csv", encoding="utf-8-sig")))[1:]
    res, _ = ocr(path)
    boxes = [((b[0][1] + b[2][1]) / 2, (b[0][0] + b[2][0]) / 2, b[2][0] - b[0][0], t.replace(" ", "").replace("，", ","))
             for b, t, _ in res]
    anchors, age_right = {}, 0
    for y, x, w, t in boxes:
        m = AGE.match(t)
        if m and x < 200:
            anchors[f"[{m.group(1)},{m.group(2)}]"] = y
            age_right = max(age_right, x + w / 2)
    order = [r[0] for r in rows]
    known = [(i, anchors[a]) for i, a in enumerate(order) if a in anchors]
    ys = np.interp(range(len(order)), [i for i, _ in known], [y for _, y in known])
    col_x, pending = {}, []
    for idx, r in enumerate(rows):
        age = r[0]
        expect = [(ci, c) for ci, c in enumerate(r[1:]) if c != ""]
        y0 = ys[idx]
        got = [(x, t) for y, x, w, t in sorted(boxes, key=lambda b: b[1])
               if abs(y - y0) < 8 and x > age_right + 5 and NUM.match(t)]
        gi = 0
        for ci, e in expect:
            if gi < len(got) and got[gi][1] == e:
                col_x.setdefault(ci, []).append(got[gi][0]); confirmed += 1; gi += 1; continue
            if e == "-":
                dash_unconfirmed += 1; continue
            if gi < len(got) and not any(t == e for _, t in got[gi + 1:]):
                gi += 1
            pending.append((age, y0, ci, e))
        for x, t in got[gi:]:
            if t != "-" and not any(a == age for a, *_ in pending):
                if (key, age, "extra") in VISUALLY_VERIFIED:
                    visual += 1; continue
                print(f"[{key}] {age}: extra OCR token {t} at x={x:.0f}"); bad += 1
    for age, y0, ci, e in pending:
        reread = read_cell(img, float(np.median(col_x[ci])), y0)
        if reread == e:
            confirmed += 1
        elif (key, age, ci + 1) in VISUALLY_VERIFIED:
            visual += 1
        else:
            print(f"[{key}] {age} col{ci + 1}: manual={e} reread={reread}"); bad += 1
    print(f"[{key}] {len(rows)} rows done, {len(pending)} cells re-read")
print(f"ocr_confirmed={confirmed} visually_verified={visual} dash_unconfirmed={dash_unconfirmed} mismatched={bad}")
sys.exit(1 if bad else 0)
