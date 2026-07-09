"""Table detection and table structure parsing.

A graphic region is treated as a table when its crop contains long
horizontal and vertical ruling lines whose crossings form a regular
lattice. The ruling-line positions become row/column boundaries; cells are
built between neighbouring boundaries and merged into row/col spans where
the separating ruling line is missing. Text items (OCR or PDF native)
whose word centers fall inside a cell become the cell content.

Detection is deliberately conservative (minimum rows/cols, minimum line
length, crossing-coverage checks) so hatched drawings or frames are not
misclassified as tables; anything rejected here falls through to the
existing drawing/image classification untouched.
"""
import cv2
import numpy as np

DEFAULTS = {
    "enable": True,
    "min_rows": 2,                   # minimum data rows
    "min_cols": 2,                   # minimum columns
    "line_kernel_divisor": 20,       # morphology kernel = side / divisor
    "min_line_length_ratio": 0.35,   # keep segments longer than ratio * table span
    "boundary_merge_tol_px": 10,     # cluster ruling lines closer than this
    "min_intersection_ratio": 0.55,  # found crossings / expected crossings
    "separator_coverage": 0.45,      # ruling coverage needed to keep a cell border
    "min_cell_size_px": 12,          # reject lattices finer than this (hatching)
    "min_grid_cover_ratio": 0.5,     # grid extent area / crop area
    "max_stray_ink_ratio": 0.008,    # non-ruling, non-text ink allowed inside the grid
    "min_line_kernel_px": 10,        # lower bound of the ruling-line morphology kernel
    "stray_dilate_px": 5,            # line-mask dilation when measuring stray ink
    "merge_max_gap_px": 200,         # split-table merge: max gap between aligned regions
    "merge_axis_overlap": 0.8,       # split-table merge: required bbox alignment
    "merge_bridge_coverage": 0.9,    # ruling must cross this fraction of the gap
    "page_level_detection": True,    # detect tables from the page's ruling-line network
    "network_gap_px": 3,             # bridge breaks this small when connecting ruling lines
    "min_boundary_span_ratio": 0.6,  # every ruling line must cover this much of the grid
    "min_cell_text_ratio": 0.45,     # cells containing text / all cells
}


def _table_cfg(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    out.update(cfg.get("table", {}))
    return out


def _line_masks(crop: np.ndarray, cfg: dict):
    """Binarize a crop and extract horizontal/vertical ruling-line masks."""
    t = _table_cfg(cfg)
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    v = cfg["vectorize"]
    block = v["binarize_block_size"] | 1
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, block, v["binarize_C"])
    hk = max(w // t["line_kernel_divisor"], t["min_line_kernel_px"])
    vk = max(h // t["line_kernel_divisor"], t["min_line_kernel_px"])
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    return binary, horiz, vert


def _line_positions(mask: np.ndarray, axis: int, min_len: float, tol: float) -> list:
    """Extract ruling-line positions from a horizontal/vertical line mask.

    axis=0: horizontal lines -> [(y, x_lo, x_hi)]
    axis=1: vertical lines   -> [(x, y_lo, y_hi)]
    Segments lying at the same position (within tol) are merged so a line
    interrupted by cell text still counts as one boundary.
    """
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        length = w if axis == 0 else h
        if length < min_len:
            continue
        pos = cents[i][1] if axis == 0 else cents[i][0]
        lo, hi = (x, x + w) if axis == 0 else (y, y + h)
        segs.append((float(pos), float(lo), float(hi)))
    segs.sort()

    clusters = []
    for pos, lo, hi in segs:
        if clusters and pos - clusters[-1][-1][0] <= tol:
            clusters[-1].append((pos, lo, hi))
        else:
            clusters.append([(pos, lo, hi)])
    merged = []
    for cl in clusters:
        span = sum(s[2] - s[1] for s in cl)
        pos = sum(s[0] * (s[2] - s[1]) for s in cl) / max(span, 1e-6)
        merged.append((pos, min(s[1] for s in cl), max(s[2] for s in cl)))
    return merged


def _crossing_ratio(horiz: np.ndarray, vert: np.ndarray,
                    row_pos: list, col_pos: list, tol: int) -> float:
    """Fraction of expected boundary crossings that actually have ink."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol + 1, 2 * tol + 1))
    cross = cv2.dilate(horiz, kernel) & cv2.dilate(vert, kernel)
    h, w = cross.shape
    found = 0
    for y, _, _ in row_pos:
        for x, _, _ in col_pos:
            yi = min(max(int(round(y)), 0), h - 1)
            xi = min(max(int(round(x)), 0), w - 1)
            if cross[yi, xi]:
                found += 1
    return found / max(len(row_pos) * len(col_pos), 1)


def _separator_present(mask: np.ndarray, pos: float, lo: float, hi: float,
                       axis: int, tol: int, coverage: float) -> bool:
    """Check whether a ruling line actually exists on a cell border segment.

    axis=1: vertical separator at x=pos over y range lo..hi.
    axis=0: horizontal separator at y=pos over x range lo..hi.
    """
    h, w = mask.shape
    p = int(round(pos))
    a, b = int(round(lo)), int(round(hi))
    if axis == 1:
        band = mask[max(0, a):min(h, b), max(0, p - tol):min(w, p + tol + 1)]
    else:
        band = mask[max(0, p - tol):min(h, p + tol + 1), max(0, a):min(w, b)]
    if band.size == 0:
        return False
    hit = band.any(axis=1) if axis == 1 else band.any(axis=0)
    return float(hit.mean()) >= coverage


def _merge_spans(nrows: int, ncols: int, merge_right: set, merge_down: set) -> list:
    """Union-find cells whose shared border is missing; return span groups.

    Returns [{"row", "col", "row_span", "col_span"}] with one entry per
    merged cell group (rectangular hull of the group).
    """
    parent = list(range(nrows * ncols))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j) in merge_right:
        union(i * ncols + j, i * ncols + j + 1)
    for (i, j) in merge_down:
        union(i * ncols + j, (i + 1) * ncols + j)

    groups = {}
    for i in range(nrows):
        for j in range(ncols):
            r = find(i * ncols + j)
            g = groups.setdefault(r, [i, j, i, j])
            g[0] = min(g[0], i); g[1] = min(g[1], j)
            g[2] = max(g[2], i); g[3] = max(g[3], j)
    return [{"row": g[0], "col": g[1],
             "row_span": g[2] - g[0] + 1, "col_span": g[3] - g[1] + 1}
            for g in sorted(groups.values(), key=lambda g: (g[0], g[1]))]


def _stray_ink_ratio(binary: np.ndarray, horiz: np.ndarray, vert: np.ndarray,
                     row_pos: list, col_pos: list, bbox: list,
                     text_items: list, dilate_px: int) -> float:
    """Ink inside the grid extent that is neither ruling line nor known text.

    Real tables are nearly empty between their ruling lines and cell text,
    while drawings (frame elevations, rebar details) keep hatching,
    diagonals and symbols there. Ratio is stray pixels / grid extent area.
    """
    h, w = binary.shape
    line_mask = cv2.dilate(horiz | vert,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px)))
    text_mask = np.zeros((h, w), np.uint8)
    ox, oy = float(bbox[0]), float(bbox[1])
    for it in text_items:
        for wd in (it.get("words") or [it]):
            wx0, wy0, wx1, wy1 = wd["bbox"]
            a0, b0 = int(wx0 - ox) - 2, int(wy0 - oy) - 2
            a1, b1 = int(wx1 - ox) + 2, int(wy1 - oy) + 2
            if a1 > 0 and b1 > 0 and a0 < w and b0 < h:
                text_mask[max(0, b0):min(h, b1), max(0, a0):min(w, a1)] = 255
    stray = binary & ~line_mask & ~text_mask
    gy0, gy1 = int(round(row_pos[0][0])), int(round(row_pos[-1][0]))
    gx0, gx1 = int(round(col_pos[0][0])), int(round(col_pos[-1][0]))
    sub = stray[max(0, gy0):min(h, gy1), max(0, gx0):min(w, gx1)]
    if sub.size == 0:
        return 1.0
    return float((sub > 0).mean())


def _assign_cell_text(cells: list, text_items: list) -> None:
    """Fill each cell's text from words whose center lies inside the cell bbox
    (page pixel coordinates). Word order: top-to-bottom, left-to-right."""
    words = []
    for it in text_items:
        for w in (it.get("words") or [it]):
            x0, y0, x1, y1 = w["bbox"]
            words.append(((x0 + x1) / 2, (y0 + y1) / 2, w["text"]))
    words.sort(key=lambda w: (w[1], w[0]))
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        toks = [t for cx, cy, t in words if x0 <= cx < x1 and y0 <= cy < y1]
        cell["text"] = " ".join(toks)


def try_parse_table(crop: np.ndarray, bbox: list, text_items: list, cfg: dict):
    """Detect a ruling-line table in a region crop and parse its structure.

    bbox is the region's page-pixel bbox (crop origin offset). Returns None
    when the region is not a table, otherwise a dict (page pixel coords):
      {"confidence", "rows", "cols", "num_cells",
       "row_boundaries", "col_boundaries",
       "cells": [{"row","col","row_span","col_span","bbox","text"}]}
    """
    t = _table_cfg(cfg)
    if not t["enable"] or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if min(h, w) < 3 * t["min_cell_size_px"]:
        return None

    binary, horiz, vert = _line_masks(crop, cfg)

    tol = t["boundary_merge_tol_px"]
    row_pos = _line_positions(horiz, 0, t["min_line_length_ratio"] * w, tol)
    col_pos = _line_positions(vert, 1, t["min_line_length_ratio"] * h, tol)
    if len(row_pos) < t["min_rows"] + 1 or len(col_pos) < t["min_cols"] + 1:
        return None

    # Reject fine lattices (hatching) and grids that only cover a corner.
    row_gaps = np.diff([p for p, _, _ in row_pos])
    col_gaps = np.diff([p for p, _, _ in col_pos])
    if np.median(row_gaps) < t["min_cell_size_px"] or np.median(col_gaps) < t["min_cell_size_px"]:
        return None
    gw = col_pos[-1][0] - col_pos[0][0]
    gh = row_pos[-1][0] - row_pos[0][0]
    if gw * gh / (h * w) < t["min_grid_cover_ratio"]:
        return None

    # Table rules run across the whole grid; crossing line networks in
    # drawings (beam/slab edges, rebar ticks) leave short boundaries.
    span = t["min_boundary_span_ratio"]
    if (min((hi - lo) / gw for _, lo, hi in row_pos) < span
            or min((hi - lo) / gh for _, lo, hi in col_pos) < span):
        return None

    ratio = _crossing_ratio(horiz, vert, row_pos, col_pos, max(tol, 3))
    if ratio < t["min_intersection_ratio"]:
        return None

    # Drawings (hatched frames, rebar details) can form a lattice too, but
    # unlike tables they keep extra ink between the ruling lines.
    if _stray_ink_ratio(binary, horiz, vert, row_pos, col_pos, bbox,
                        text_items, t["stray_dilate_px"]) > t["max_stray_ink_ratio"]:
        return None

    # Build cells between neighbouring boundaries; drop missing separators
    # (merged cells) by checking ruling coverage on every internal border.
    nrows, ncols = len(row_pos) - 1, len(col_pos) - 1
    band = max(tol // 2, 3)
    merge_right, merge_down = set(), set()
    for i in range(nrows):
        y0, y1 = row_pos[i][0], row_pos[i + 1][0]
        for j in range(ncols - 1):
            if not _separator_present(vert, col_pos[j + 1][0], y0, y1, 1,
                                      band, t["separator_coverage"]):
                merge_right.add((i, j))
    for j in range(ncols):
        x0, x1 = col_pos[j][0], col_pos[j + 1][0]
        for i in range(nrows - 1):
            if not _separator_present(horiz, row_pos[i + 1][0], x0, x1, 0,
                                      band, t["separator_coverage"]):
                merge_down.add((i, j))

    ox, oy = float(bbox[0]), float(bbox[1])
    cells = _merge_spans(nrows, ncols, merge_right, merge_down)
    for cell in cells:
        i, j = cell["row"], cell["col"]
        cell["bbox"] = [round(ox + col_pos[j][0], 1),
                        round(oy + row_pos[i][0], 1),
                        round(ox + col_pos[j + cell["col_span"]][0], 1),
                        round(oy + row_pos[i + cell["row_span"]][0], 1)]
    _assign_cell_text(cells, text_items)

    # Tables exist to hold content: reject lattices whose cells are mostly
    # empty (line networks in drawings form grids but carry no cell text).
    filled = sum(1 for c in cells if c["text"].strip())
    if not cells or filled / len(cells) < t["min_cell_text_ratio"]:
        return None

    return {
        "confidence": round(ratio, 3),
        "rows": nrows,
        "cols": ncols,
        "num_cells": len(cells),
        "row_boundaries": [round(oy + p, 1) for p, _, _ in row_pos],
        "col_boundaries": [round(ox + p, 1) for p, _, _ in col_pos],
        "cells": cells,
    }


def detect_page_tables(page_img: np.ndarray, text_items: list, cfg: dict) -> list:
    """Detect tables from the whole page's ruling-line network.

    Graphic region proposals are built from text-masked ink, which can cut
    one table into several fragments; the printed ruling lines themselves
    stay connected. Long horizontal/vertical lines are extracted from the
    raw page, each connected line network becomes a table candidate, and its
    bbox goes through the normal try_parse_table gates (drawings' line
    networks fail the stray-ink check there). Returns [{"bbox", "table"}].
    """
    t = _table_cfg(cfg)
    if not t["enable"] or not t["page_level_detection"]:
        return []
    _, horiz, vert = _line_masks(page_img, cfg)
    lines = horiz | vert
    g = t["network_gap_px"]
    if g > 0:
        lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (g, g)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    min_w = (t["min_cols"] + 1) * t["min_cell_size_px"]
    min_h = (t["min_rows"] + 1) * t["min_cell_size_px"]
    out = []
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if w < min_w or h < min_h:
            continue
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        crop = page_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        table = try_parse_table(crop, bbox, text_items, cfg)
        if table is not None:
            out.append({"bbox": bbox, "table": table})
    return out


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap of two 1-D intervals relative to the shorter interval."""
    inter = min(a1, b1) - max(a0, b0)
    return inter / max(min(a1 - a0, b1 - b0), 1e-6)


def _gap_bridged(page_img: np.ndarray, union_bbox: list, gap: tuple, axis: int,
                 cfg: dict, coverage: float) -> bool:
    """Check that ruling lines physically cross the gap between two regions.

    axis=0: an x-gap must be crossed by a horizontal line; axis=1: a y-gap by
    a vertical line. gap is (lo, hi) in page coordinates along that axis.
    Text masking can split one table into several ink components while the
    printed ruling lines still run continuously through the gap — that is the
    case this test accepts; separate tables with plain text between them have
    no ruling across the gap and are rejected.
    """
    x0, y0, x1, y1 = [int(round(v)) for v in union_bbox]
    crop = page_img[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    _, horiz, vert = _line_masks(crop, cfg)
    if axis == 0:
        lo, hi = int(gap[0] - x0), int(gap[1] - x0)
        band = horiz[:, max(0, lo):max(0, hi)]
        hit = band.mean(axis=1) if band.size else np.array([0.0])
    else:
        lo, hi = int(gap[0] - y0), int(gap[1] - y0)
        band = vert[max(0, lo):max(0, hi), :]
        hit = band.mean(axis=0) if band.size else np.array([0.0])
    return bool((hit >= coverage * 255).any())


def merge_split_tables(page_img: np.ndarray, graphic_regions: list,
                       text_items: list, cfg: dict) -> list:
    """Reunite tables that text masking split into several graphic regions.

    Two regions aligned along one axis (merge_axis_overlap) with a small gap
    (merge_max_gap_px) are unioned when ruling lines cross the gap
    (merge_bridge_coverage) and the union parses as a table. The merged
    region carries the parsed result in "table" so the caller can skip
    reparsing. Non-mergeable regions pass through untouched.
    """
    t = _table_cfg(cfg)
    if not t["enable"] or len(graphic_regions) < 2:
        return graphic_regions
    regions = [dict(g) for g in graphic_regions]
    changed = True
    while changed:
        changed = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i]["bbox"], regions[j]["bbox"]
                y_ov = _axis_overlap(a[1], a[3], b[1], b[3])
                x_ov = _axis_overlap(a[0], a[2], b[0], b[2])
                gx = (min(a[2], b[2]), max(a[0], b[0]))   # x-gap when side by side
                gy = (min(a[3], b[3]), max(a[1], b[1]))   # y-gap when stacked
                if y_ov >= t["merge_axis_overlap"] and gx[1] - gx[0] <= t["merge_max_gap_px"]:
                    axis, gap = 0, gx
                elif x_ov >= t["merge_axis_overlap"] and gy[1] - gy[0] <= t["merge_max_gap_px"]:
                    axis, gap = 1, gy
                else:
                    continue
                union = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                if gap[1] > gap[0] and not _gap_bridged(page_img, union, gap, axis, cfg,
                                                        t["merge_bridge_coverage"]):
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in union]
                table = try_parse_table(page_img[y0:y1, x0:x1], union, text_items, cfg)
                if table is None:
                    continue
                regions[i] = {"bbox": union, "label": None, "table": table}
                del regions[j]
                changed = True
                break
            if changed:
                break
    return regions
