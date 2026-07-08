"""Save parsing results: layout.json, region crops, vector json/svg, overlay image."""
import json
from pathlib import Path

import cv2
import numpy as np

from .loader import imwrite_unicode
from .vectorize import polylines_to_svg

# Overlay colors per region type (BGR).
COLORS = {
    "text": (255, 128, 0),        # blue
    "dimension": (0, 0, 255),     # red
    "annotation": (0, 165, 255),  # orange
    "drawing": (0, 180, 0),       # green
    "image": (200, 0, 200),       # purple
}


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def draw_overlay(page_img: np.ndarray, regions: list, vectors_by_region: dict) -> np.ndarray:
    """Draw classified region boxes and vectorized polylines on a page copy."""
    ov = page_img.copy()
    for r in regions:
        x0, y0, x1, y1 = [int(round(c)) for c in r["bbox"]]
        color = COLORS.get(r["type"], (128, 128, 128))
        cv2.rectangle(ov, (x0, y0), (x1, y1), color, 2)
        cv2.putText(ov, f'{r["id"]}:{r["type"]}', (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for polylines in vectors_by_region.values():
        for pl in polylines:
            pts = np.array(pl["points"], dtype=np.int32)
            cv2.polylines(ov, [pts], pl.get("closed", False), (0, 180, 0), 1)
    return ov


def export_page(page_dir: Path, page, regions: list, vectors_by_region: dict,
                native_vectors: list, cfg: dict) -> dict:
    """Write all page results into page_dir and return the layout.json content."""
    exp = cfg["export"]
    page_dir.mkdir(parents=True, exist_ok=True)
    h, w = page.image.shape[:2]

    if exp["save_page_image"]:
        imwrite_unicode(page_dir / "page.png", page.image)

    if exp["save_region_crops"]:
        crop_dir = page_dir / "regions"
        crop_dir.mkdir(exist_ok=True)
        for r in regions:
            if r["type"] not in ("drawing", "image"):
                continue
            x0, y0, x1, y1 = [int(round(c)) for c in r["bbox"]]
            crop = page.image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            if crop.size == 0:
                continue
            fname = f'{r["id"]}_{r["type"]}.png'
            imwrite_unicode(crop_dir / fname, crop)
            r["image_file"] = f"regions/{fname}"

    if vectors_by_region:
        vec_dir = page_dir / "vectors"
        vec_dir.mkdir(exist_ok=True)
        for rid, polylines in vectors_by_region.items():
            region = next(r for r in regions if r["id"] == rid)
            save_json(vec_dir / f"{rid}.json", {
                "region_id": rid,
                "bbox": region["bbox"],
                "coordinate_system": "page_pixel",
                "num_polylines": len(polylines),
                "num_groups": len({pl["group"] for pl in polylines}),
                "polylines": polylines,
            })
            region["vector_file"] = f"vectors/{rid}.json"
            if exp["save_svg"]:
                svg = polylines_to_svg(polylines, w, h)
                (vec_dir / f"{rid}.svg").write_text(svg, encoding="utf-8")
                region["svg_file"] = f"vectors/{rid}.svg"

    if native_vectors:
        save_json(page_dir / "native_vectors.json", {
            "source": "pdf_native",
            "coordinate_system": "page_pixel",
            "num_polylines": len(native_vectors),
            "polylines": native_vectors,
        })

    layout = {
        "page": page.page_no,
        "size": {"width": w, "height": h},
        "original_size": {"width": page.original_size[0], "height": page.original_size[1]}
        if page.original_size else {"width": w, "height": h},
        "scale": round(page.scale, 3),
        "num_regions": len(regions),
        "regions": regions,
    }
    save_json(page_dir / "layout.json", layout)

    if exp["save_overlay"]:
        imwrite_unicode(page_dir / "overlay.png", draw_overlay(page.image, regions, vectors_by_region))

    return layout
