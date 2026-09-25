"""Load input files (jpg/png/pdf) and split them into pages.

PDF files are rendered page by page with PyMuPDF. Native text (words with
coordinates) and native vector paths (page.get_drawings) are extracted at the
same time. All coordinates are converted to the pixel coordinate system of the
rendered page image.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PageData:
    page_no: int                     # 1-based page number
    image: np.ndarray                # BGR page image
    native_words: list = field(default_factory=list)      # [{bbox:[x0,y0,x1,y1], text, confidence}]
    native_polylines: list = field(default_factory=list)  # [{points:[[x,y],...], closed:bool}]
    scale: float = 1.0               # upscale factor relative to the original
    original_size: tuple = None      # (width, height) of the original page in pixels
    coordinate_transform: dict = field(default_factory=dict)


def maybe_upscale(page: PageData, cfg: dict) -> PageData:
    """Upscale a low-resolution page to improve OCR and vectorization.

    After upscaling, every coordinate refers to the upscaled page. The scale
    factor and the original size are stored on the page and written to
    layout.json so results can be mapped back to original coordinates.
    """
    pre = cfg.get("preprocess", {})
    target = pre.get("upscale_target_min_side", 0)
    h, w = page.image.shape[:2]
    page.original_size = (w, h)
    if target <= 0 or min(h, w) >= target:
        return page
    scale = min(float(pre.get("max_scale", 3.0)), target / min(h, w))
    page.image = cv2.resize(page.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    page.scale = scale
    scale_x = page.image.shape[1] / w
    scale_y = page.image.shape[0] / h
    for wd in page.native_words:
        wd["bbox"] = [value * (scale_x if index % 2 == 0 else scale_y)
                      for index, value in enumerate(wd["bbox"])]
    for pl in page.native_polylines:
        pl["points"] = [[x * scale_x, y * scale_y] for x, y in pl["points"]]
    transform = fitz.Matrix(page.coordinate_transform.get("source_to_pixel", [1, 0, 0, 1, 0, 0]))
    transform = transform * fitz.Matrix(scale_x, scale_y)
    page.coordinate_transform.update(source_to_pixel=list(transform), pixel_to_source=list(~transform),
                                     upscale=[scale_x, scale_y])
    return page


def imread_unicode(path: Path) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters (Windows-safe)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters (Windows-safe)."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Image encoding failed: {path}")
    buf.tofile(str(path))


def _bezier_points(p0, p1, p2, p3, n: int, tolerance: float = 0.25):
    """Sample a cubic with a second-derivative bound on chord error."""
    control = np.asarray([p0, p1, p2, p3], dtype=float)
    curvature = 6 * max(np.linalg.norm(control[0] - 2 * control[1] + control[2]),
                        np.linalg.norm(control[1] - 2 * control[2] + control[3]))
    n = max(n, 2, int(np.ceil(np.sqrt(curvature / (8 * max(tolerance, 1e-6))))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append([float(x), float(y)])
    return pts


def _extract_native_vectors(page: fitz.Page, zoom: float, bezier_samples: int,
                            transform=None, tolerance: float = 0.25) -> list:
    """Convert PDF vector paths to polylines in page pixel coordinates.

    Consecutive line/curve items whose endpoints chain together are merged
    into a single polyline.
    """
    polylines = []
    for path in page.get_drawings():
        if path.get("type") == "f" or path.get("stroke_opacity", 1) == 0:
            continue
        current: list = []

        def flush(closed=False):
            nonlocal current
            if len(current) >= 2:
                polylines.append({
                    "points": [[round(x * zoom, 2), round(y * zoom, 2)] for x, y in current],
                    "closed": closed,
                })
            current = []

        for item in path["items"]:
            op = item[0]
            if op == "l":  # straight line: p1 -> p2
                p1, p2 = item[1], item[2]
                if current and current[-1] == [p1.x, p1.y]:
                    current.append([p2.x, p2.y])
                else:
                    flush()
                    current = [[p1.x, p1.y], [p2.x, p2.y]]
            elif op == "c":  # cubic bezier: control points p1..p4
                p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                pts = _bezier_points((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y),
                                     bezier_samples, tolerance / zoom)
                if current and current[-1] == [p1.x, p1.y]:
                    current.extend(pts[1:])
                else:
                    flush()
                    current = pts
            elif op == "re":  # rectangle
                flush()
                r = item[1]
                polylines.append({
                    "points": [[round(v * zoom, 2) for v in pt] for pt in
                               [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1], [r.x0, r.y0]]],
                    "closed": True,
                })
            elif op == "qu":  # quadrilateral
                flush()
                q = item[1]
                pts = [q.ul, q.ur, q.lr, q.ll]
                polylines.append({
                    "points": [[round(p.x * zoom, 2), round(p.y * zoom, 2)] for p in pts + [pts[0]]],
                    "closed": True,
                })
        flush(closed=bool(path.get("closePath")))
    transform = transform if transform is not None else page.rotation_matrix * fitz.Matrix(zoom, zoom)
    for polyline in polylines:
        polyline["points"] = [list(fitz.Point(x / zoom, y / zoom) * transform)
                              for x, y in polyline["points"]]
        polyline["source"] = "pdf_native"
    return polylines


def load_pages(file_path: Path, cfg: dict):
    """Yield PageData objects, one per page of the input file."""
    ext = file_path.suffix.lower()
    pdf_cfg = cfg["pdf"]

    if ext == ".pdf":
        with fitz.open(str(file_path)) as document:
            for page in document:
                yield _render_pdf_page(page, cfg)
    elif ext in IMAGE_EXTS:
        yield PageData(1, imread_unicode(file_path), coordinate_transform={
            "source_space": "image_pixel", "source_to_pixel": [1, 0, 0, 1, 0, 0],
            "pixel_to_source": [1, 0, 0, 1, 0, 0], "upscale": [1.0, 1.0],
        })
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def _render_pdf_page(page, cfg: dict) -> PageData:
    pdf_cfg = cfg["pdf"]
    zoom = pdf_cfg["render_dpi"] / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
    transform = page.rotation_matrix * matrix * fitz.Matrix(1, 0, 0, 1, -pixmap.x, -pixmap.y)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    words = []
    if pdf_cfg["use_native_text"]:
        for word in page.get_text("words"):
            if word[4].strip():
                words.append({"bbox": list(fitz.Rect(word[:4]) * transform),
                              "text": word[4], "confidence": 1.0})
    polylines = []
    if pdf_cfg["use_native_vectors"]:
        tolerance = cfg["vectorize"].get("native_curve_tolerance_px", 0.25)
        tolerance /= max(1.0, cfg.get("preprocess", {}).get("max_scale", 3.0))
        polylines = _extract_native_vectors(page, zoom, cfg["vectorize"]["bezier_samples"], transform, tolerance)
    return PageData(page.number + 1, image, words, polylines, coordinate_transform={
        "source_space": "pymupdf_unrotated_cropbox_points",
        "source_to_pixel": list(transform), "pixel_to_source": list(~transform),
        "rotation_degrees": page.rotation, "cropbox": list(page.cropbox),
        "mediabox": list(page.mediabox), "render_origin": [pixmap.x, pixmap.y],
        "render_dpi": pdf_cfg["render_dpi"], "upscale": [1.0, 1.0]})
