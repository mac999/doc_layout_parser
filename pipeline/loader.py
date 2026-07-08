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
    for wd in page.native_words:
        wd["bbox"] = [round(v * scale, 1) for v in wd["bbox"]]
    for pl in page.native_polylines:
        pl["points"] = [[round(x * scale, 2), round(y * scale, 2)] for x, y in pl["points"]]
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


def _bezier_points(p0, p1, p2, p3, n: int):
    """Sample a cubic Bezier curve into n points."""
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        pts.append([float(x), float(y)])
    return pts


def _extract_native_vectors(page: fitz.Page, zoom: float, bezier_samples: int) -> list:
    """Convert PDF vector paths to polylines in page pixel coordinates.

    Consecutive line/curve items whose endpoints chain together are merged
    into a single polyline.
    """
    polylines = []
    for path in page.get_drawings():
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
                pts = _bezier_points((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y), bezier_samples)
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
    return polylines


def load_pages(file_path: Path, cfg: dict):
    """Yield PageData objects, one per page of the input file."""
    ext = file_path.suffix.lower()
    pdf_cfg = cfg["pdf"]

    if ext == ".pdf":
        doc = fitz.open(str(file_path))
        zoom = pdf_cfg["render_dpi"] / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            native_words = []
            if pdf_cfg["use_native_text"]:
                for w in page.get_text("words"):
                    x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                    if not text.strip():
                        continue
                    native_words.append({
                        "bbox": [round(x0 * zoom, 1), round(y0 * zoom, 1),
                                 round(x1 * zoom, 1), round(y1 * zoom, 1)],
                        "text": text,
                        "confidence": 1.0,
                    })

            native_polylines = []
            if pdf_cfg["use_native_vectors"]:
                native_polylines = _extract_native_vectors(page, zoom, cfg["vectorize"]["bezier_samples"])

            yield PageData(i + 1, img, native_words, native_polylines)
        doc.close()
    elif ext in IMAGE_EXTS:
        yield PageData(1, imread_unicode(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
