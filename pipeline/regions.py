"""Graphic region (layout) detection and drawing/image classification.

1) Binarize the page, mask out text boxes, dilate the remaining ink and take
   connected components as graphic region proposals.
2) Classify each region as a line-based drawing or a continuous-tone image
   using heuristics (saturation, mid-tone ratio, ink ratio). An optional VLM
   can re-check ambiguous regions (see vlm.py).
"""
import cv2
import numpy as np


def binarize_ink(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Return a binary mask where ink pixels (lines, characters) are 255."""
    v = cfg["vectorize"]
    block = v["binarize_block_size"]
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, block, v["binarize_C"])


def mask_text(ink: np.ndarray, text_items: list, pad: int) -> np.ndarray:
    """Return the ink mask with all text boxes erased (set to 0)."""
    out = ink.copy()
    h, w = out.shape
    for it in text_items:
        x0, y0, x1, y1 = it["bbox"]
        x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
        x1 = min(w, int(x1) + pad); y1 = min(h, int(y1) + pad)
        out[y0:y1, x0:x1] = 0
    return out


def detect_graphic_regions(page_img: np.ndarray, text_items: list, cfg: dict,
                           exclude_bboxes: list = None):
    """Detect graphic regions by clustering non-text ink.

    exclude_bboxes: page areas already claimed (e.g. page-level tables);
    their ink is erased so they do not produce regions again.

    Returns (regions, labels):
      regions: [{"bbox": [x0,y0,x1,y1], "label": component id}]
      labels : page-sized connected-component label map. Even when bounding
               boxes overlap, each region vectorizes only the pixels of its
               own component by using this map as a mask.
    """
    lay = cfg["layout"]
    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    ink = binarize_ink(gray, cfg)
    ink = mask_text(ink, text_items, lay["text_mask_padding"])
    if exclude_bboxes:
        ink = mask_text(ink, [{"bbox": b} for b in exclude_bboxes],
                        lay["text_mask_padding"])

    # Scan/frame borders hugging the page edge would otherwise become tall
    # thin "drawing" regions; erase the outermost strip of ink.
    m = lay.get("page_border_margin_px", 0)
    if m > 0:
        ink[:m, :] = 0; ink[-m:, :] = 0
        ink[:, :m] = 0; ink[:, -m:] = 0

    k = lay["dilate_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blob = cv2.dilate(ink, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    page_area = page_img.shape[0] * page_img.shape[1]
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h < lay["min_region_area"]:
            continue
        if (w * h) / page_area > lay["max_page_cover_ratio"]:
            continue
        regions.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "label": i})
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return regions, labels


def reclassify_text_regions(regions: list, cfg: dict) -> None:
    """Override content-rule text types by geometric context.

    Dimension/annotation/text rules are unreliable inside graphics, so any
    text-based region lying mostly (text_region_overlap_ratio of its area)
    inside a table becomes layout.text_in_table_type (default "text") and
    inside a drawing becomes layout.text_in_drawing_type (default
    "annotation"). Tables win when a table sits inside a drawing bbox.
    """
    lay = cfg["layout"]
    min_ov = lay["text_region_overlap_ratio"]
    tables = [r["bbox"] for r in regions if r["type"] == "table"]
    drawings = [r["bbox"] for r in regions if r["type"] == "drawing"]

    def frac_inside(tb, gb):
        ix = max(0.0, min(tb[2], gb[2]) - max(tb[0], gb[0]))
        iy = max(0.0, min(tb[3], gb[3]) - max(tb[1], gb[1]))
        return ix * iy / max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1e-6)

    for r in regions:
        if r["type"] not in ("text", "dimension", "annotation"):
            continue
        if any(frac_inside(r["bbox"], b) >= min_ov for b in tables):
            new_type, tag = lay["text_in_table_type"], "in_table"
        elif any(frac_inside(r["bbox"], b) >= min_ov for b in drawings):
            new_type, tag = lay["text_in_drawing_type"], "in_drawing"
        else:
            continue
        if new_type != r["type"]:
            r["type"] = new_type
            r["source"] = f'{r["source"]}+{tag}'


def classify_graphic_heuristic(crop: np.ndarray, cfg: dict) -> tuple[str, float, dict]:
    """Decide whether a region is a line drawing or a raster image.

    All thresholds/weights come from config classify.heuristic.
    Returns (label, confidence 0..1, metrics dict).
    - Drawing: white background with thin dark lines, so the mid-tone ratio
      and the saturation are both low.
    - Image/photo: continuous tone, so the mid-tone ratio or saturation is high.
    """
    hh = cfg["classify"]["heuristic"]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    total = gray.size
    dark_max, light_min = hh["dark_gray_max"], hh["light_gray_min"]
    mid_ratio = float(((gray > dark_max) & (gray < light_min)).sum()) / total  # continuous-tone pixels
    dark_ratio = float((gray <= dark_max).sum()) / total                       # ink pixels

    # Photo-likeness score: 0 = drawing, 1 = photo.
    photo_score = 0.0
    photo_score += min(sat_mean / hh["sat_photo_norm"], 1.0) * hh["sat_weight"]  # colorful -> photo
    photo_score += min(mid_ratio / hh["mid_photo_norm"], 1.0) * hh["mid_weight"] # continuous tone -> photo
    photo_score += (hh["dark_ratio_bonus"] if dark_ratio > hh["dark_ratio_bonus_threshold"] else 0.0)

    thr = hh["photo_score_threshold"]
    label = "image" if photo_score >= thr else "drawing"
    confidence = round(min(abs(photo_score - thr) / max(thr, 1 - thr), 1.0), 3)
    metrics = {"sat_mean": round(sat_mean, 1), "mid_ratio": round(mid_ratio, 3),
               "dark_ratio": round(dark_ratio, 3), "photo_score": round(photo_score, 3)}
    return label, confidence, metrics
