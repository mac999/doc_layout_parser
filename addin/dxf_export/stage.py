"""Write one DXF per page from the parsed regions, polylines and tables.

Consumes the `scale` artifact the previous stage produced, so coordinates come
out in millimetres instead of pixels. Page pixels have y growing downward and
DXF has it growing upward, so every y is flipped once, here, and nowhere else.
"""
from pipeline.stages import Artifact, Stage


def _signature(polyline: dict) -> tuple:
    """Identify a polyline by geometry. The same polyline read from two files is
    two different objects, so identity comparison silently duplicates it."""
    points = polyline["points"]
    return (round(points[0][0], 2), round(points[0][1], 2),
            round(points[-1][0], 2), round(points[-1][1], 2), len(points))

# Defaults only. Like the core pipeline, every tuned value is overridable from
# config.json, here through this stage's own settings block.
DEFAULTS = {
    "units": "mm",
    "use_drawing_scale": True,
    "write_text": True,
    "write_tables": True,
    "min_text_height_mm": 1.0,
    # Region bounding boxes enclose the ink; DXF text height is the cap height.
    "text_height_ratio": 0.8,
    # A table cell bounding box is the cell, not the glyphs inside it, so cell
    # text is sized from the row height instead of filling the box.
    "cell_text_height_ratio": 0.5,
    # Rough glyph width per unit of height for the default DXF font, used only
    # to decide whether a string would overflow its cell.
    "char_width_ratio": 0.75,
    # Below this multiple of the text height a region is too narrow to fit text
    # into, so the string is left-aligned instead of condensed to nothing.
    "min_fit_width_ratio": 0.1,
    "layers": {
        "text":       ["TEXT", 7],
        "dimension":  ["DIMENSION", 1],
        "annotation": ["ANNOTATION", 3],
        "drawing":    ["DRAWING", 5],
        "table":      ["TABLE", 4],
        "image":      ["IMAGE", 8],
    },
}
FALLBACK_LAYER = ["DRAWING", 5]


class DxfExport(Stage):
    name = "dxf_export"
    version = "1.1"
    requires = ("scale",)
    provides = ("dxf",)

    def setup(self, cfg: dict) -> None:
        # Settings are checked before the optional dependency is imported: a
        # typo in config.json is worth reporting whether or not ezdxf is here,
        # and it is the more actionable of the two failures.
        unknown = set(cfg) - set(DEFAULTS) - {"budget_sec"}
        if unknown:
            raise ValueError(f"Unknown dxf_export setting(s): {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **cfg}
        if self.cfg["units"] not in ("mm", "m"):
            raise ValueError("dxf_export.units must be 'mm' or 'm'")
        for key in ("min_text_height_mm", "text_height_ratio", "cell_text_height_ratio",
                    "min_fit_width_ratio", "char_width_ratio"):
            if not isinstance(self.cfg[key], (int, float)) or self.cfg[key] <= 0:
                raise ValueError(f"dxf_export.{key} must be a positive number")

        import ezdxf                                  # add-in's own dependency
        from ezdxf.enums import TextEntityAlignment
        self._ezdxf = ezdxf
        self._fit = TextEntityAlignment.FIT
        self._left = TextEntityAlignment.LEFT

        self.units = self.cfg["units"]
        self.use_drawing_scale = self.cfg["use_drawing_scale"]
        self.min_text_height = float(self.cfg["min_text_height_mm"])
        self.write_text = self.cfg["write_text"]
        self.write_tables = self.cfg["write_tables"]
        self.layers = {k: tuple(v) for k, v in self.cfg["layers"].items()}

    def _mm_per_px(self, scale: dict) -> tuple:
        px_per_mm = scale["px_per_mm_paper"]
        if self.use_drawing_scale and scale.get("mm_per_px_drawing"):
            return scale["mm_per_px_drawing"], f'1:{scale["drawing_scale"]}'
        return 1.0 / px_per_mm, "paper" if scale.get("unit", "mm") == "mm" else "pixel"

    def _place(self, msp, text, bbox, layer, mm_per_px, pt, extent="ink"):
        """Place one string, sized by what its bounding box means.

        extent="ink"  the box is the glyph extent the parser measured, so the
                      string should fill it: FIT keeps the height and adjusts
                      the width. Sizing from the box height alone overflowed it
                      for 52 of 54 strings on one sheet, because the source
                      glyphs are narrower than the default DXF font.
        extent="box"  the box is a container (a table cell), so the string is
                      left-aligned at its natural size and only condensed when
                      it would otherwise run past the right edge.
        """
        x0, y0, x1, y1 = bbox
        ratio = self.cfg["text_height_ratio" if extent == "ink" else "cell_text_height_ratio"]
        height = max(self.min_text_height, abs(y1 - y0) * mm_per_px * ratio)
        entity = msp.add_text(text, height=height, dxfattribs={"layer": layer})
        left, right = pt(x0, y1), pt(x1, y1)
        span = right[0] - left[0]
        if span <= height * self.cfg["min_fit_width_ratio"]:
            entity.set_placement(left, align=self._left)
        elif extent == "ink" or len(text) * height * self.cfg["char_width_ratio"] > span:
            entity.set_placement(left, right, align=self._fit)
        else:
            entity.set_placement(left, align=self._left)
        return entity

    def on_page(self, ctx):
        scale = ctx.artifact("scale")
        if scale is None:
            raise ValueError("No scale artifact on this page")
        mm_per_px, basis = self._mm_per_px(scale.meta)
        height_px = ctx.size["height"]

        def pt(x, y):
            """Page pixel to DXF millimetre, flipping the y axis once."""
            return (x * mm_per_px, (height_px - y) * mm_per_px)

        doc = self._ezdxf.new("R2010", setup=True)
        doc.header["$INSUNITS"] = 4 if self.units == "mm" else 6      # 4=mm, 6=m
        for layer, color in self.layers.values():
            doc.layers.add(layer, color=color)
        msp = doc.modelspace()
        counts = {"lwpolyline": 0, "text": 0, "line": 0}

        for region in ctx.regions():
            layer = self.layers.get(region["type"], tuple(FALLBACK_LAYER))[0]
            x0, y0, x1, y1 = region["bbox"]

            for polyline in ctx.vectors(region["id"]):
                points = [pt(x, y) for x, y in polyline["points"]]
                if len(points) < 2:
                    continue
                msp.add_lwpolyline(points, close=bool(polyline.get("closed")),
                                   dxfattribs={"layer": layer})
                counts["lwpolyline"] += 1

            text = (region.get("text") or "").strip()
            if self.write_text and text and region["type"] in ("text", "dimension", "annotation"):
                self._place(msp, text, region["bbox"], layer, mm_per_px, pt)
                counts["text"] += 1

            table = ctx.table(region["id"]) if self.write_tables else None
            if table:
                for cell in table.get("cells", []):
                    cx0, cy0, cx1, cy1 = cell["bbox"]
                    corners = [pt(cx0, cy0), pt(cx1, cy0), pt(cx1, cy1), pt(cx0, cy1)]
                    table_layer = self.layers.get("table", tuple(FALLBACK_LAYER))[0]
                    msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": table_layer})
                    counts["lwpolyline"] += 1
                    cell_text = (cell.get("text") or "").strip()
                    if cell_text and self.write_text:
                        self._place(msp, cell_text, cell["bbox"], table_layer, mm_per_px,
                                    pt, extent="box")
                        counts["text"] += 1

        # Native PDF vectors the parser kept but no drawing region claimed.
        claimed = {_signature(p) for region in ctx.regions() for p in ctx.vectors(region["id"])}
        for polyline in ctx.native_vectors():
            if _signature(polyline) in claimed:
                continue
            points = [pt(x, y) for x, y in polyline["points"]]
            if len(points) >= 2:
                msp.add_lwpolyline(points, close=bool(polyline.get("closed")),
                                   dxfattribs={"layer": FALLBACK_LAYER[0]})
                counts["lwpolyline"] += 1

        name = f"page_{ctx.page_no:03d}.dxf"
        path = ctx.workdir / name
        doc.saveas(str(path))
        meta = {"page": ctx.page_no, "units": "px" if basis == "pixel" else self.units,
                "scale_basis": basis,
                "mm_per_px": round(mm_per_px, 6), "entities": counts,
                "extents_mm": [round(ctx.size["width"] * mm_per_px, 2),
                               round(height_px * mm_per_px, 2)]}
        print(f"  [stage] dxf_export p{ctx.page_no}: {name} "
              f"({counts['lwpolyline']} polylines, {counts['text']} texts, {basis})")
        return [Artifact(kind="dxf", path=path, meta=meta)]

    def on_document(self, ctx):
        """Index every page's DXF so a consumer does not have to walk the tree."""
        produced = ctx.page_artifacts("dxf")
        if not produced:
            return []
        index = {
            "source_file": ctx.source_file,
            "output_key": ctx.output_key,
            "units": self.units,
            "pages": [{"page": a.meta["page"],
                       "file": str(a.path.relative_to(ctx.out_dir)),
                       "scale_basis": a.meta["scale_basis"],
                       "entities": a.meta["entities"]} for a in produced],
        }
        path = ctx.write_json("index.json", index)
        print(f"  [stage] dxf_export: indexed {len(produced)} DXF file(s)")
        return [Artifact(kind="dxf_index", path=path, meta={"count": len(produced)})]


STAGE = DxfExport()
