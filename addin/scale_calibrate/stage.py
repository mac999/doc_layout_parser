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
MM_PER_INCH = 25.4
# "SCALE 1:100", "SCALE = 1/50", "S=1:50", "축척 1:200", "S C A L E  1:30".
# The separator accepts . ; and - as well as : and / because OCR reliably
# confuses them on drawing text (measured: "SCALE 1:100" read as "SCALE 1.100").
# A bare "1:2" is not accepted: ratios appear all over a drawing (slopes, bar
# splice ratios) and would be read as a sheet scale. Requiring the keyword is
# what keeps the loose separator safe.
SCALE_RE = re.compile(
    r"(?:\bscale|\bs\s*c\s*a\s*l\s*e|\bs(?=\s*[=:])|축척|척도)"
    r"\s*[=:]?\s*1\s*[:/.;-]\s*(\d{1,5})\b", re.IGNORECASE)


def _paper_px_per_mm(layout: dict, assume_dpi=None) -> tuple:
    """(px_per_mm, method), or (None, "unknown") when the page has no physical size.

    A PDF carries one; a scanned image does not, so the caller has to supply the
    scanner DPI or accept pixel units.
    """
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
        return dpi / MM_PER_INCH, "render_dpi"
    if assume_dpi:
        return float(assume_dpi) / MM_PER_INCH, "assumed_dpi"
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


DEFAULTS = {
    "read_title_block": True,
    # A raster scan carries no physical page size. Give the scanner DPI here to
    # get millimetres instead of pixels.
    "assume_scan_dpi": None,
    # Used only when no sheet scale is printed; recorded as "assumed" so a
    # consumer can tell a read value from a supplied one.
    "assume_drawing_scale": None,
}


class ScaleCalibrate(Stage):
    name = "scale_calibrate"
    version = "1.1"
    provides = ("scale",)

    def setup(self, cfg: dict) -> None:
        unknown = set(cfg) - set(DEFAULTS) - {"budget_sec"}
        if unknown:
            raise ValueError(f"Unknown scale_calibrate setting(s): {sorted(unknown)}")
        merged = {**DEFAULTS, **cfg}
        for key in ("assume_scan_dpi", "assume_drawing_scale"):
            value = merged[key]
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"scale_calibrate.{key} must be a positive number or null")
        self.assume = merged["assume_drawing_scale"]
        self.assume_dpi = merged["assume_scan_dpi"]
        self.read_title_block = merged["read_title_block"]

    def on_page(self, ctx):
        px_per_mm, method = _paper_px_per_mm(ctx.layout, self.assume_dpi)
        if px_per_mm is None:
            # A raster scan has no physical size. Report pixel units with zero
            # confidence rather than refusing: a downstream exporter can still
            # produce correct geometry, just not in millimetres.
            px_per_mm, method = 1.0, "pixel"

        denominator, source_region = (None, None)
        if self.read_title_block:
            denominator, source_region = _drawing_scale(
                ctx.regions("text", "annotation", "dimension"))
        if denominator is None and self.assume:
            denominator, source_region, method = int(self.assume), None, method + "+assumed"

        unit = "px" if method == "pixel" else "mm"
        # Real-world size needs both the page's physical size and the drawing
        # scale. The denominator alone is still reported: it was read from the
        # sheet, and a consumer may supply the missing half.
        drawing = (round(denominator / px_per_mm, 6)
                   if denominator and unit == "mm" else None)
        payload = {
            "px_per_mm_paper": round(px_per_mm, 6),
            "unit": unit,
            "method": method,
            "drawing_scale": denominator,
            "scale_source_region": source_region,
            "mm_per_px_drawing": drawing,
            "confidence": 0.0 if unit == "px" else
                          (1.0 if denominator and source_region else
                           (0.5 if denominator else 0.0)),
            "page": ctx.page_no,
        }
        path = ctx.write_json("scale.json", payload)
        label = (f"1:{denominator}" if denominator else "no sheet scale found")
        if unit == "px":
            label += ", pixel units (no physical page size)"
        print(f"  [stage] scale_calibrate p{ctx.page_no}: "
              f"{px_per_mm:.3f} px/mm ({method}), {label}")
        return [Artifact(kind="scale", path=path, meta=payload)]


STAGE = ScaleCalibrate()
