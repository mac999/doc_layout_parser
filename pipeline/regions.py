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


def detect_graphic_regions(page_img: np.ndarray, text_items: list, cfg: dict):
    """Detect graphic regions by clustering non-text ink.

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


def classify_graphic_heuristic(crop: np.ndarray) -> tuple[str, float, dict]:
    """Decide whether a region is a line drawing or a raster image.

    Returns (label, confidence 0..1, metrics dict).
    - Drawing: white background with thin dark lines, so the mid-tone ratio
      and the saturation are both low.
    - Image/photo: continuous tone, so the mid-tone ratio or saturation is high.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    total = gray.size
    mid_ratio = float(((gray > 60) & (gray < 200)).sum()) / total   # continuous-tone pixels
    dark_ratio = float((gray <= 60).sum()) / total                  # ink pixels

    # Photo-likeness score: 0 = drawing, 1 = photo.
    photo_score = 0.0
    photo_score += min(sat_mean / 60.0, 1.0) * 0.45          # colorful regions look like photos
    photo_score += min(mid_ratio / 0.5, 1.0) * 0.45          # continuous tone looks like photos
    photo_score += (0.10 if dark_ratio > 0.5 else 0.0)       # mostly-dark regions may be photos

    label = "image" if photo_score >= 0.5 else "drawing"
    confidence = round(abs(photo_score - 0.5) * 2.0, 3)
    metrics = {"sat_mean": round(sat_mean, 1), "mid_ratio": round(mid_ratio, 3),
               "dark_ratio": round(dark_ratio, 3), "photo_score": round(photo_score, 3)}
    return label, confidence, metrics
