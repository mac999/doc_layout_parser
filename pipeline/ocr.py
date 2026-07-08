"""Text detection with EasyOCR plus text / dimension / annotation classification.

- OCR: EasyOCR (GPU). If a PDF page provides enough native text, the native
  words are used instead of running OCR.
- Classification: regex/heuristic rules distinguish dimensions (numbers,
  diameter/radius marks, tolerances, rebar callouts), annotations (grid
  labels, section marks, title keywords) and plain text.
"""
import re

import numpy as np

_reader = None


def get_reader(cfg: dict):
    """Create the EasyOCR reader once and reuse it (model load is expensive)."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(cfg["ocr"]["languages"], gpu=cfg["ocr"]["gpu"], verbose=False)
    return _reader


def run_ocr(page_img: np.ndarray, cfg: dict) -> list:
    """OCR the whole page. Returns [{bbox:[x0,y0,x1,y1], text, confidence}]."""
    reader = get_reader(cfg)
    o = cfg["ocr"]
    results = reader.readtext(
        page_img,
        paragraph=o["paragraph"],
        mag_ratio=o.get("mag_ratio", 1.0),      # internal magnification for small text
        canvas_size=o.get("canvas_size", 2560),
        min_size=o.get("min_size", 10),
        text_threshold=o.get("text_threshold", 0.7),
        low_text=o.get("low_text", 0.4),
    )
    min_conf = cfg["ocr"]["min_confidence"]
    items = []
    for box, text, conf in results:
        if conf < min_conf or not text.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "bbox": [round(float(min(xs)), 1), round(float(min(ys)), 1),
                     round(float(max(xs)), 1), round(float(max(ys)), 1)],
            "text": text.strip(),
            "confidence": round(float(conf), 3),
        })
    return items


# Dimension patterns: plain numbers, diameter/radius symbols, tolerances,
# spacing callouts, units, thickness, quantities, scale ratios, formulas.
_DIM_PATTERNS = [
    r"^[~≈±]?\d{1,6}([.,]\d+)?$",      # 1200, 350.5, ~300
    r"^\d+([.,]\d+)?\s*[xX×*]\s*\d+",        # 300x600
    r"[ØøΦφ⌀]",          # diameter symbols
    r"^R\s?\d+([.,]\d+)?$",                        # R25 (radius)
    r"±",                                     # tolerance (plus-minus)
    r"^@\s?\d+",                                   # @200 (rebar spacing)
    r"\d+\s*(mm|cm|m|MM)\b",                       # values with units
    r"^(H?D|HD|UHD)\d{1,2}([-@]\d+)?",             # D13, HD16@200 (rebar sizes)
    r"^L\s*=",                                     # L=A+B (length formula)
    r"^\d+\s*[-~]\s*[A-Z]?D?\d+",                  # 4-D22 (count-size)
    r"THK|t\s*=\s*\d+",                            # thickness
    r"^\d+\s*(EA|ea)$",                            # quantity
    r"^1\s*[/:]\s*\d+$",                           # scale 1/100, 1:100
]
_DIM_RE = [re.compile(p) for p in _DIM_PATTERNS]

# Annotation patterns: grid labels, section marks, drawing title keywords
# (English and Korean), mark numbers.
_ANNOT_PATTERNS = [
    r"^[A-Z]{1,3}\d{0,3}$",                        # X1, DDA, A (grid/member labels)
    r"^[A-Z]\s*-\s*[A-Z]$",                        # A-A (section mark)
    r"\b(DETAIL|SCALE|NOTE|NOTES|TYP|TYPE|SECTION|PLAN|ELEVATION|VIEW|KEYPLAN|KEY\s*PLAN|LIST|SCHEDULE|LEGEND)\b",
    r"(상세|축척|단면|평면|입면|주기|범례|일람표|배근도|기초|보|기둥|슬래브)",
    r"^#\d+",                                       # #3 (mark number)
]
_ANNOT_RE = [re.compile(p, re.IGNORECASE) for p in _ANNOT_PATTERNS]


def classify_text(text: str) -> str:
    """Classify a text token by content: dimension / annotation / text."""
    t = text.strip()
    for rx in _DIM_RE:
        if rx.search(t):
            return "dimension"
    for rx in _ANNOT_RE:
        if rx.search(t):
            return "annotation"
    return "text"


def group_into_lines(items: list, gap_factor: float = 1.5) -> list:
    """Merge word items that belong to the same text line.

    Words whose vertical centers are close and whose horizontal gap is within
    gap_factor times the character height are merged into one line. The bbox
    of each individual word is preserved in the "words" list.
    """
    if not items:
        return []
    items = sorted(items, key=lambda it: ((it["bbox"][1] + it["bbox"][3]) / 2, it["bbox"][0]))

    # Step 1: cluster words into rows by vertical center distance.
    rows: list[list] = []
    for it in items:
        y0, y1 = it["bbox"][1], it["bbox"][3]
        cy, h = (y0 + y1) / 2, max(y1 - y0, 1)
        for row in rows:
            if abs(cy - row[-1]["_cy"]) < 0.6 * max(h, row[-1]["_h"]):
                row.append({**it, "_cy": cy, "_h": h})
                break
        else:
            rows.append([{**it, "_cy": cy, "_h": h}])

    # Step 2: split each row into lines where the horizontal gap is too large.
    lines = []
    for row in rows:
        row.sort(key=lambda it: it["bbox"][0])
        chunk = [row[0]]
        for it in row[1:]:
            gap = it["bbox"][0] - chunk[-1]["bbox"][2]
            if gap <= gap_factor * max(it["_h"], chunk[-1]["_h"]):
                chunk.append(it)
            else:
                lines.append(chunk)
                chunk = [it]
        lines.append(chunk)

    # Step 3: merge each chunk into a single line item.
    merged = []
    for chunk in lines:
        xs0 = min(w["bbox"][0] for w in chunk); ys0 = min(w["bbox"][1] for w in chunk)
        xs1 = max(w["bbox"][2] for w in chunk); ys1 = max(w["bbox"][3] for w in chunk)
        merged.append({
            "bbox": [xs0, ys0, xs1, ys1],
            "text": " ".join(w["text"] for w in chunk),
            "confidence": round(min(w["confidence"] for w in chunk), 3),
            "words": [{"bbox": w["bbox"], "text": w["text"], "confidence": w["confidence"]}
                      for w in chunk],
        })
    return merged


def classify_line(line: dict) -> str:
    """Classify a merged line.

    - Short lines (3 tokens or fewer): apply dimension/annotation patterns.
    - Long lines (sentences): dimension only if most tokens look like
      dimensions, otherwise plain text.
    """
    tokens = line["text"].split()
    if len(tokens) <= 3:
        return classify_text(line["text"])
    dim_hits = sum(1 for t in tokens if classify_text(t) == "dimension")
    if dim_hits / len(tokens) > 0.5:
        return "dimension"
    return "text"


def get_text_items(page, cfg: dict) -> list:
    """Extract text from a page (native words preferred, OCR as fallback),
    merge words into lines, then classify each line."""
    if page.native_words and len(page.native_words) >= cfg["pdf"]["min_native_words"]:
        words = [dict(w) for w in page.native_words]
        source = "pdf_native"
    else:
        words = run_ocr(page.image, cfg)
        source = "ocr"
    items = group_into_lines(words)
    for it in items:
        it["type"] = classify_line(it)
        it["source"] = source
    return items
