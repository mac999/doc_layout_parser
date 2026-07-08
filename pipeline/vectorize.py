"""Vectorize drawing regions: binarize -> skeletonize -> graph tracing -> polylines.

- skimage.skeletonize thins every line down to a 1-pixel-wide skeleton.
- The skeleton is treated as a pixel graph. Endpoints (degree 1) and junctions
  (degree 3 or more) become nodes, and the pixel path between two nodes becomes
  one polyline, so connected segments are naturally merged.
- Each path is simplified with Douglas-Peucker (cv2.approxPolyDP). Polylines
  that share an endpoint are assigned the same connectivity group id using
  union-find.
- All output coordinates are page pixel coordinates.
"""
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .regions import binarize_ink, mask_text

# 8-connected neighbor offsets (dy, dx).
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace_paths(skel: np.ndarray) -> list:
    """Extract pixel paths between nodes from a skeleton. Each path is [(y, x), ...]."""
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    if not pts:
        return []

    # Neighbor lookup restricted to skeleton pixels.
    nbrs = {}
    for p in pts:
        y, x = p
        nbrs[p] = [(y + dy, x + dx) for dy, dx in _NB8 if (y + dy, x + dx) in pts]

    # Nodes are endpoints (degree 1) and junctions (degree >= 3).
    nodes = {p for p, ns in nbrs.items() if len(ns) != 2}
    paths = []
    visited_steps = set()  # directed steps (p, q) that were already walked

    def walk(start, first):
        """Walk from node start in the direction of first until the next node.

        If the walk returns to start without meeting a node, the path is a
        closed loop (for example a circle) and ends at start.
        """
        path = [start, first]
        visited_steps.add((start, first))
        visited_steps.add((first, start))
        prev, cur = start, first
        while cur not in nodes and cur != start:
            nxt = None
            for n in nbrs[cur]:
                if n != prev:
                    nxt = n
                    break
            if nxt is None:  # isolated dead end
                break
            visited_steps.add((cur, nxt))
            visited_steps.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for node in nodes:
        for n in nbrs[node]:
            if (node, n) in visited_steps:
                continue
            paths.append(walk(node, n))

    # Handle pure cycles that contain no node at all (for example circles).
    in_path = set()
    for p in paths:
        in_path.update(p)
    for p in pts:
        if p in in_path or len(nbrs[p]) != 2 or (p, nbrs[p][0]) in visited_steps:
            continue
        cycle = walk(p, nbrs[p][0])  # walk returns to the start, giving a closed path
        if len(cycle) >= 4 and cycle[0] == cycle[-1]:
            paths.append(cycle)
        in_path.update(cycle)

    return paths


def _simplify(path: list, epsilon: float) -> np.ndarray:
    """Convert a pixel path to a Douglas-Peucker simplified (x, y) point array."""
    arr = np.array([[x, y] for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
    closed = path[0] == path[-1] and len(path) > 3
    approx = cv2.approxPolyDP(arr, epsilon, closed)
    pts = approx.reshape(-1, 2)
    if closed and len(pts) >= 2 and not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1)).sum())


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def vectorize_region(page_img: np.ndarray, bbox: list, text_items: list, cfg: dict,
                     component_mask: np.ndarray = None) -> list:
    """Vectorize one region and return polylines in page pixel coordinates.

    component_mask: optional bool mask with the same size as the bbox crop.
    When bounding boxes overlap, it keeps only the pixels of this region's
    connected component so ink from other regions is not vectorized twice.

    Returns [{id, points:[[x,y],...], closed, length_px, num_points, group}].
    """
    v = cfg["vectorize"]
    x0, y0, x1, y1 = [int(round(c)) for c in bbox]
    crop = page_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = binarize_ink(gray, cfg)
    if component_mask is not None:
        ink[~component_mask] = 0

    # Exclude text inside the region from vectorization
    # (bboxes shifted to region-local coordinates).
    local_texts = []
    for it in text_items:
        bx0, by0, bx1, by1 = it["bbox"]
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        local_texts.append({"bbox": [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]})
    ink = mask_text(ink, local_texts, cfg["layout"]["text_mask_padding"])

    skel = skeletonize(ink > 0)
    paths = _trace_paths(skel)

    polylines = []
    uf = _UnionFind()
    endpoint_of = {}  # endpoint pixel -> list of polyline indices (for grouping)

    for path in paths:
        if len(path) < 2:
            continue
        pts = _simplify(path, v["simplify_epsilon"])
        if len(pts) < 2:
            continue
        length = _polyline_length(pts)
        closed = bool(np.array_equal(pts[0], pts[-1]) and len(pts) > 3)
        if length < v["min_polyline_length_px"] and not closed:
            continue
        idx = len(polylines)
        polylines.append({
            "points": pts, "closed": closed, "length_px": round(length, 1),
        })
        uf.find(idx)
        for end in (path[0], path[-1]):
            endpoint_of.setdefault(end, []).append(idx)

    # Polylines that share a node pixel (or touch an adjacent node pixel)
    # belong to the same connectivity group.
    for end, idxs in endpoint_of.items():
        for other in idxs[1:]:
            uf.union(idxs[0], other)
        ey, ex = end
        for dy, dx in _NB8:  # junction clusters span adjacent pixels
            nb = (ey + dy, ex + dx)
            if nb in endpoint_of:
                uf.union(idxs[0], endpoint_of[nb][0])

    group_map = {}
    results = []
    for i, pl in enumerate(polylines):
        root = uf.find(i)
        group_map.setdefault(root, len(group_map))
        pts_page = [[int(px + x0), int(py + y0)] for px, py in pl["points"]]
        results.append({
            "id": i,
            "points": pts_page,
            "closed": pl["closed"],
            "length_px": pl["length_px"],
            "num_points": len(pts_page),
            "group": group_map[root],
        })
    return results


def polylines_to_svg(polylines: list, width: int, height: int) -> str:
    """Render a polyline list as an SVG string."""
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>']
    for pl in polylines:
        pts = " ".join(f"{p[0]},{p[1]}" for p in pl["points"])
        tag = "polygon" if pl.get("closed") else "polyline"
        if pl.get("closed") and pl["points"][0] == pl["points"][-1]:
            pts = " ".join(f"{p[0]},{p[1]}" for p in pl["points"][:-1])
        lines.append(f'<{tag} points="{pts}" fill="none" stroke="black" stroke-width="1"/>')
    lines.append("</svg>")
    return "\n".join(lines)
