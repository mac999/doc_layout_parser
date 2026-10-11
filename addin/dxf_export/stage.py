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

LAYERS = {
    "text":       ("TEXT", 7),
    "dimension":  ("DIMENSION", 1),
    "annotation": ("ANNOTATION", 3),
    "drawing":    ("DRAWING", 5),
    "table":      ("TABLE", 4),
    "image":      ("IMAGE", 8),
}


class DxfExport(Stage):
    name = "dxf_export"
    version = "1.0"
    requires = ("scale",)
    provides = ("dxf",)

    def setup(self, cfg: dict) -> None:
        import ezdxf                                  # add-in's own dependency
        self._ezdxf = ezdxf
        self.units = cfg.get("units", "mm")
        self.use_drawing_scale = cfg.get("use_drawing_scale", True)
        self.min_text_height = float(cfg.get("min_text_height_mm", 1.0))
        self.write_text = cfg.get("write_text", True)
        self.write_tables = cfg.get("write_tables", True)

    def _mm_per_px(self, scale: dict) -> tuple:
        px_per_mm = scale["px_per_mm_paper"]
        if self.use_drawing_scale and scale.get("mm_per_px_drawing"):
            return scale["mm_per_px_drawing"], f'1:{scale["drawing_scale"]}'
        return 1.0 / px_per_mm, "paper"

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
        for layer, color in LAYERS.values():
            doc.layers.add(layer, color=color)
        msp = doc.modelspace()
        counts = {"lwpolyline": 0, "text": 0, "line": 0}

        for region in ctx.regions():
            layer = LAYERS.get(region["type"], ("DRAWING", 5))[0]
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
                height = max(self.min_text_height, abs(y1 - y0) * mm_per_px * 0.8)
                msp.add_text(text, height=height,
                             dxfattribs={"layer": layer}).set_placement(pt(x0, y1))
                counts["text"] += 1

            table = ctx.table(region["id"]) if self.write_tables else None
            if table:
                for cell in table.get("cells", []):
                    cx0, cy0, cx1, cy1 = cell["bbox"]
                    corners = [pt(cx0, cy0), pt(cx1, cy0), pt(cx1, cy1), pt(cx0, cy1)]
                    msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": "TABLE"})
                    counts["lwpolyline"] += 1
                    cell_text = (cell.get("text") or "").strip()
                    if cell_text and self.write_text:
                        height = max(self.min_text_height, abs(cy1 - cy0) * mm_per_px * 0.5)
                        msp.add_text(cell_text, height=height,
                                     dxfattribs={"layer": "TABLE"}).set_placement(pt(cx0, cy1))
                        counts["text"] += 1

        # Native PDF vectors the parser kept but no drawing region claimed.
        claimed = {_signature(p) for region in ctx.regions() for p in ctx.vectors(region["id"])}
        for polyline in ctx.native_vectors():
            if _signature(polyline) in claimed:
                continue
            points = [pt(x, y) for x, y in polyline["points"]]
            if len(points) >= 2:
                msp.add_lwpolyline(points, close=bool(polyline.get("closed")),
                                   dxfattribs={"layer": "DRAWING"})
                counts["lwpolyline"] += 1

        name = f"page_{ctx.page_no:03d}.dxf"
        path = ctx.workdir / name
        doc.saveas(str(path))
        meta = {"page": ctx.page_no, "units": self.units, "scale_basis": basis,
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
