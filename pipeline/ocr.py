"""Text detection with EasyOCR plus text / dimension / annotation classification.

- OCR: EasyOCR supplements native PDF text; overlapping duplicates are merged.
- Classification: regex/heuristic rules distinguish dimensions (numbers,
  diameter/radius marks, tolerances, rebar callouts), annotations (grid
  labels, section marks, title keywords) and plain text.
"""
import re
import unicodedata

import numpy as np

_reader = None
_reader_key = None


def get_reader(cfg: dict):
    """Create the EasyOCR reader once and reuse it (model load is expensive)."""
    global _reader, _reader_key
    key = (tuple(cfg["ocr"]["languages"]), cfg["ocr"]["gpu"])
    if _reader is None or _reader_key != key:
        import easyocr
        _reader = easyocr.Reader(cfg["ocr"]["languages"], gpu=cfg["ocr"]["gpu"], verbose=False)
        _reader_key = key
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
        rotation_info=o.get("rotation_info") or None,
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


def group_into_lines(items: list, gap_factor: float = 1.5, row_factor: float = 0.6) -> list:
    """Merge word items that belong to the same text line.

    Words whose vertical centers are close (within row_factor times the
    character height) and whose horizontal gap is within gap_factor times the
    character height are merged into one line. The bbox of each individual
    word is preserved in the "words" list.
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
            if abs(cy - row[-1]["_cy"]) < row_factor * max(h, row[-1]["_h"]):
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
                      | {key: w[key] for key in ("source", "native_matches") if key in w}
                      for w in chunk],
        })
    return merged


def classify_line(line: dict, max_tokens: int = 3, dim_ratio: float = 0.5) -> str:
    """Classify a merged line.

    - Short lines (max_tokens or fewer): apply dimension/annotation patterns.
    - Long lines (sentences): dimension only if the fraction of tokens that
      look like dimensions exceeds dim_ratio, otherwise plain text.
    """
    tokens = line["text"].split()
    if len(tokens) <= max_tokens:
        return classify_text(line["text"])
    dim_hits = sum(1 for t in tokens if classify_text(t) == "dimension")
    if dim_hits / len(tokens) > dim_ratio:
        return "dimension"
    return "text"


def _normalized_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _overlap_fraction(first: list, second: list) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap / max(1, (first[2] - first[0]) * (first[3] - first[1]))


def merge_text_words(native_words: list, ocr_words: list) -> list:
    """Keep native precision for duplicates and retain OCR-only text."""
    words = []
    for incoming in native_words + ocr_words:
        normalized = _normalized_text(incoming["text"])
        matches = [word for word in words if max(
            _overlap_fraction(word["bbox"], incoming["bbox"]),
            _overlap_fraction(incoming["bbox"], word["bbox"])) >= 0.7]
        if any(_normalized_text(word["text"]) == normalized for word in matches):
            continue
        ordered = sorted(matches, key=lambda word: (word["bbox"][1], word["bbox"][0]))
        combined = _normalized_text("".join(word["text"] for word in ordered))
        if combined and (normalized == combined or any(
                normalized in _normalized_text(word["text"]) and
                _overlap_fraction(incoming["bbox"], word["bbox"]) >= 0.8 for word in matches)):
            continue
        covered = [word for word in matches if _normalized_text(word["text"]) in normalized
                   and _overlap_fraction(word["bbox"], incoming["bbox"]) >= 0.8]
        if covered and incoming["source"] == "ocr":
            native_matches = [word for word in covered if "pdf_native" in word["source"]]
            incoming = {**incoming, "source": "ocr+pdf_native" if native_matches else "ocr",
                        "native_matches": native_matches}
            words = [word for word in words if all(word is not match for match in covered)]
        words.append(dict(incoming))
    return words


def _ocr_page(image: np.ndarray, cfg: dict) -> list:
    size = cfg["ocr"].get("tile_size", 0)
    height, width = image.shape[:2]
    if size <= 0 or max(height, width) <= size:
        return run_ocr(image, cfg)
    overlap = cfg["ocr"].get("tile_overlap", 128)
    if not 0 <= overlap < size:
        raise ValueError("ocr.tile_overlap must be between 0 and tile_size - 1")
    words = []
    for top in range(0, height, size - overlap):
        for left in range(0, width, size - overlap):
            for word in run_ocr(image[top:top + size, left:left + size], cfg):
                words.append({**word, "bbox": [value + (left if index % 2 == 0 else top)
                                               for index, value in enumerate(word["bbox"])], "source": "ocr"})
    return merge_text_words([], words)


def get_text_items(page, cfg: dict) -> list:
    """Merge usable native words with OCR, retaining per-word provenance."""
    height, width = page.image.shape[:2]
    native = [{**word, "source": "pdf_native"} for word in page.native_words
              if word["text"].strip() and "\ufffd" not in word["text"]
              and all(character.isprintable() for character in word["text"])
              and 0 <= word["bbox"][0] < word["bbox"][2] <= width
              and 0 <= word["bbox"][1] < word["bbox"][3] <= height]
    supplement = cfg["pdf"].get("supplement_native_with_ocr", True)
    detected = _ocr_page(page.image, cfg) if supplement or not native else []
    words = merge_text_words(native, [{**word, "source": "ocr"} for word in detected])
    o = cfg["ocr"]
    items = group_into_lines(words, gap_factor=o.get("line_gap_factor", 1.5),
                             row_factor=o.get("line_row_factor", 0.6))
    for it in items:
        it["type"] = classify_line(it, max_tokens=o.get("short_line_max_tokens", 3),
                                   dim_ratio=o.get("dim_token_ratio", 0.5))
        it["source"] = "+".join(sorted({source for word in it["words"]
                          for source in word["source"].split("+")}))
    return items
