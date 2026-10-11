"""Resolve how many page pixels make one millimetre, and at what drawing scale.

Two separate questions that are easy to confuse:

  paper mm    - exact, derived from the render DPI or the PDF point transform.
                A 1:100 wall 5 m long measures 50 mm on the sheet.
  drawing mm  - paper mm multiplied by the drawing scale denominator, which is
                printed in the title block and has to be read, not computed.

This stage always resolves paper mm. It resolves the drawing scale only when it
can find it, and says so in `confidence` either way, so a downstream exporter
can decide whether to trust it.
"""
import re

from pipeline.stages import Artifact, Stage

POINTS_PER_MM = 25.4 / 72.0
# "SCALE 1:100", "S=1/50", "축척 1:200"
SCALE_RE = re.compile(r"(?:scale|s|축척)\s*[=:]?\s*1\s*[:/]\s*(\d{1,5})", re.IGNORECASE)


def _paper_px_per_mm(layout: dict) -> tuple:
    """(px_per_mm, method). Falls back to the render DPI when no matrix is usable."""
    transform = layout.get("coordinate_transform") or {}
    matrix = transform.get("pixel_to_source")
    space = transform.get("source_space", "")
    if matrix and "point" in space:
        # One pixel spans |a| source points along x; points convert to mm exactly.
        mm_per_px = abs(matrix[0]) * POINTS_PER_MM
        if mm_per_px > 0:
            return 1.0 / mm_per_px, "pdf_point_transform"
    dpi = transform.get("render_dpi")
    if dpi:
        return dpi / 25.4, "render_dpi"
    return None, "unknown"


def _drawing_scale(regions: list) -> tuple:
    """(denominator, region_id) read from any text the parser already extracted."""
    for region in regions:
        text = (region.get("text") or "").strip()
        if not text:
            continue
        found = SCALE_RE.search(text)
        if found:
            value = int(found.group(1))
            if 1 <= value <= 20000:
                return value, region["id"]
    return None, None


class ScaleCalibrate(Stage):
    name = "scale_calibrate"
    version = "1.0"
    provides = ("scale",)

    def setup(self, cfg: dict) -> None:
        self.assume = cfg.get("assume_drawing_scale")
        self.read_title_block = cfg.get("read_title_block", True)

    def on_page(self, ctx):
        px_per_mm, method = _paper_px_per_mm(ctx.layout)
        if px_per_mm is None:
            raise ValueError("No render DPI or point transform; cannot resolve page scale")

        denominator, source_region = (None, None)
        if self.read_title_block:
            denominator, source_region = _drawing_scale(
                ctx.regions("text", "annotation", "dimension"))
        if denominator is None and self.assume:
            denominator, source_region, method = int(self.assume), None, method + "+assumed"

        payload = {
            "px_per_mm_paper": round(px_per_mm, 6),
            "method": method,
            "drawing_scale": denominator,
            "scale_source_region": source_region,
            # mm of real building per page pixel, only meaningful once the scale is known
            "mm_per_px_drawing": round(denominator / px_per_mm, 6) if denominator else None,
            "confidence": 1.0 if denominator and source_region else (0.5 if denominator else 0.0),
            "page": ctx.page_no,
        }
        path = ctx.write_json("scale.json", payload)
        label = f"1:{denominator}" if denominator else "paper only"
        print(f"  [stage] scale_calibrate p{ctx.page_no}: "
              f"{px_per_mm:.3f} px/mm ({method}), {label}")
        return [Artifact(kind="scale", path=path, meta=payload)]


STAGE = ScaleCalibrate()
