"""Save parsing results: layout.json, region crops, vector json/svg, overlay image."""
import json
import hashlib
import platform
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from .loader import imwrite_unicode
from .vectorize import polylines_to_svg

# Overlay colors per region type (BGR).
COLORS = {
    "text": (255, 128, 0),        # blue
    "dimension": (0, 0, 255),     # red
    "annotation": (0, 165, 255),  # orange
    "drawing": (0, 180, 0),       # green
    "image": (200, 0, 200),       # purple
    "table": (0, 200, 200),       # yellow
}


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


@lru_cache(maxsize=1)
def output_validator():
    schema = json.loads((Path(__file__).parents[1] / "doc" / "result.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def build_manifest(source: Path, cfg: dict, run_id: str, warnings: list) -> dict:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    versions = {"python": platform.python_version()}
    for package in ("PyMuPDF", "opencv-python", "opencv-python-headless", "numpy", "scikit-image",
                    "shapely", "easyocr", "torch", "ollama", "openai", "google-genai", "jsonschema"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return {"schema_version": "1.0", "run_id": run_id, "status": "running",
            "source_file": source.name, "input": {"path": str(source.resolve()), "sha256": digest.hexdigest()},
            "config": cfg, "versions": versions, "warnings": list(warnings), "pages": [], "num_pages": 0,
            "started_at": datetime.now(timezone.utc).isoformat()}


def validate_layout(layout: dict, page_dir: Path, vectors_by_region: dict) -> None:
    """Validate serialized shape, geometry and referenced artifacts."""
    output_validator().validate(layout)
    width, height = layout["size"]["width"], layout["size"]["height"]
    regions = layout["regions"]
    if layout["num_regions"] != len(regions) or len({region["id"] for region in regions}) != len(regions):
        raise ValueError("Invalid region count or duplicate IDs")

    def check_bbox(bounds, container):
        left, top, right, bottom = bounds
        if not np.isfinite(bounds).all() or not (container[0] <= left < right <= container[2]
                                                 and container[1] <= top < bottom <= container[3]):
            raise ValueError("Bounding box outside its container")

    for region in regions:
        check_bbox(region["bbox"], [0, 0, width, height])
        for word in region.get("words", []):
            check_bbox(word["bbox"], [0, 0, width, height])
        for key in ("image_file", "vector_file", "svg_file", "table_file"):
            if key in region:
                target = (page_dir / region[key]).resolve()
                if not target.is_relative_to(page_dir.resolve()) or not target.is_file():
                    raise ValueError(f"Missing or unsafe artifact: {key}")
        table = region.get("table")
        if table:
            rows, columns = table["rows"], table["cols"]
            if type(rows) is not int or type(columns) is not int or min(rows, columns) < 1:
                raise ValueError("Invalid table size")
            if table["num_cells"] != len(table["cells"]):
                raise ValueError("Invalid cell count")
            occupancy = np.zeros((rows, columns), dtype=np.uint8)
            for cell in table["cells"]:
                row, column, row_span, col_span = (cell[key] for key in ("row", "col", "row_span", "col_span"))
                if any(type(value) is not int for value in (row, column, row_span, col_span)) or not (
                        0 <= row < row + row_span <= rows and 0 <= column < column + col_span <= columns):
                    raise ValueError("Invalid cell span")
                occupancy[row:row + row_span, column:column + col_span] += 1
                check_bbox(cell["bbox"], region["bbox"])
            if not np.all(occupancy == 1):
                raise ValueError("Table cells overlap or leave gaps")
        for polyline in vectors_by_region.get(region["id"], []):
            points = np.asarray(polyline["points"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError("Invalid vector points")
            bounds = region["bbox"]
            if not (np.all(points >= bounds[:2]) and np.all(points <= bounds[2:])):
                raise ValueError("Vector outside its region")
            if polyline["num_points"] != len(points) or type(polyline["group"]) is not int:
                raise ValueError("Invalid vector metadata")
            if polyline.get("closed") and not np.allclose(points[0], points[-1]):
                raise ValueError("Open endpoints on a closed vector")


def draw_overlay(page_img: np.ndarray, regions: list, vectors_by_region: dict) -> np.ndarray:
    """Draw classified region boxes and vectorized polylines on a page copy."""
    ov = page_img.copy()
    for r in regions:
        x0, y0, x1, y1 = [int(round(c)) for c in r["bbox"]]
        color = COLORS.get(r["type"], (128, 128, 128))
        cv2.rectangle(ov, (x0, y0), (x1, y1), color, 2)
        cv2.putText(ov, f'{r["id"]}:{r["type"]}', (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for cell in r.get("table", {}).get("cells", []):
            cx0, cy0, cx1, cy1 = [int(round(c)) for c in cell["bbox"]]
            cv2.rectangle(ov, (cx0, cy0), (cx1, cy1), color, 1)
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
            if r["type"] not in ("drawing", "image", "table"):
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

    table_regions = [r for r in regions if r.get("table")]
    if table_regions:
        tab_dir = page_dir / "tables"
        tab_dir.mkdir(exist_ok=True)
        for r in table_regions:
            save_json(tab_dir / f'{r["id"]}.json', {
                "region_id": r["id"],
                "bbox": r["bbox"],
                "coordinate_system": "page_pixel",
                **r["table"],
            })
            r["table_file"] = f'tables/{r["id"]}.json'

    if native_vectors:
        save_json(page_dir / "native_vectors.json", {
            "source": "pdf_native",
            "coordinate_system": "page_pixel",
            "num_polylines": len(native_vectors),
            "polylines": native_vectors,
        })

    layout = {
        "schema_version": "1.0",
        "status": "complete",
        "page": page.page_no,
        "size": {"width": w, "height": h},
        "original_size": {"width": page.original_size[0], "height": page.original_size[1]}
        if page.original_size else {"width": w, "height": h},
        "scale": round(page.scale, 3),
        "coordinate_transform": page.coordinate_transform or {
            "source_space": "image_pixel", "source_to_pixel": [1, 0, 0, 1, 0, 0],
            "pixel_to_source": [1, 0, 0, 1, 0, 0]},
        "num_regions": len(regions),
        "regions": regions,
        "warnings": [f"{region['id']}: VLM {region['vlm'].get('reason', 'fallback')}"
                     for region in regions if region.get("vlm", {}).get("fallback")]
                    + [f"{region['id']}: uncertain cell merge" for region in regions
                       if any(cell.get("merge_uncertain") for cell in region.get("table", {}).get("cells", []))],
    }
    for region in regions:
        region.setdefault("confidence_kind", "native_text" if region["source"] == "pdf_native" else
                          "grid_coverage" if region["source"] == "table_grid" else "ocr_score")
    validate_layout(layout, page_dir, vectors_by_region)
    save_json(page_dir / "layout.json", layout)

    if exp["save_overlay"]:
        imwrite_unicode(page_dir / "overlay.png", draw_overlay(page.image, regions, vectors_by_region))

    return layout
