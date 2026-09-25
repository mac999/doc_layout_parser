"""Compare reference and predicted layouts in the same pixel coordinates."""
import argparse
import json
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString, box
from shapely.ops import unary_union


def _scores(matches, expected, actual):
    precision = matches / actual if actual else float(expected == 0)
    recall = matches / expected if expected else float(actual == 0)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _distance(expected, actual):
    previous = list(range(len(actual) + 1))
    for row, expected_character in enumerate(expected, 1):
        current = [row]
        for column, actual_character in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected_character != actual_character)))
        previous = current
    return previous[-1]


def _cells(region):
    return {(cell["row"], cell["col"], cell["row_span"], cell["col_span"], cell.get("text", ""))
            for cell in region.get("table", {}).get("cells", [])}


def evaluate_layout(expected: dict, actual: dict, iou_threshold=0.5) -> dict:
    if expected["size"] != actual["size"]:
        raise ValueError("Layouts must use the same page size")
    references, predictions = expected["regions"], actual["regions"]
    graph = nx.Graph()
    for ref_index, reference in enumerate(references):
        for pred_index, prediction in enumerate(predictions):
            first, second = box(*reference["bbox"]), box(*prediction["bbox"])
            union_area = first.union(second).area
            overlap = first.intersection(second).area / union_area if union_area else 0.0
            if reference["type"] == prediction["type"] and overlap >= iou_threshold:
                graph.add_edge(("ref", ref_index), ("pred", pred_index), weight=overlap)
    pairs = []
    for first, second in nx.max_weight_matching(graph, maxcardinality=True):
        reference, prediction = (first, second) if first[0] == "ref" else (second, first)
        pairs.append((reference[1], prediction[1]))
    pairs.sort()
    result = {"regions": _scores(len(pairs), len(references), len(predictions)), "by_type": {}}
    for kind in sorted({region["type"] for region in references + predictions}):
        matched = [(ref_index, pred_index) for ref_index, pred_index in pairs if references[ref_index]["type"] == kind]
        result["by_type"][kind] = _scores(len(matched), sum(region["type"] == kind for region in references),
                                          sum(region["type"] == kind for region in predictions))
        result["by_type"][kind]["mean_iou"] = sum(graph[("ref", ref_index)][("pred", pred_index)]["weight"]
            for ref_index, pred_index in matched) / len(matched) if matched else None
    errors = sum(_distance(references[ref_index].get("text", ""), predictions[pred_index].get("text", ""))
                 for ref_index, pred_index in pairs)
    ref_matched, pred_matched = {pair[0] for pair in pairs}, {pair[1] for pair in pairs}
    errors += sum(len(region.get("text", "")) for index, region in enumerate(references) if index not in ref_matched)
    errors += sum(len(region.get("text", "")) for index, region in enumerate(predictions) if index not in pred_matched)
    characters = sum(len(region.get("text", "")) for region in references)
    result["text"] = {"edit_errors": errors, "reference_characters": characters,
                      "cer": errors / characters if characters else (0.0 if not errors else None)}
    cell_matches = sum(len(_cells(references[ref_index]) & _cells(predictions[pred_index]))
                       for ref_index, pred_index in pairs)
    result["table_cells"] = _scores(cell_matches, sum(len(_cells(region)) for region in references),
                                    sum(len(_cells(region)) for region in predictions))
    result["vectors"] = []
    for ref_index, pred_index in pairs:
        reference, prediction = references[ref_index].get("polylines", []), predictions[pred_index].get("polylines", [])
        if not reference and not prediction:
            continue
        ref_shape = unary_union([LineString(polyline["points"]) for polyline in reference])
        pred_shape = unary_union([LineString(polyline["points"]) for polyline in prediction])
        result["vectors"].append({"reference_id": references[ref_index].get("id", ref_index),
            "hausdorff_px": float(ref_shape.hausdorff_distance(pred_shape)) if reference and prediction else None,
            "reference_groups": len({polyline["group"] for polyline in reference}),
            "predicted_groups": len({polyline["group"] for polyline in prediction}),
            "reference_closed": sum(bool(polyline.get("closed")) for polyline in reference),
            "predicted_closed": sum(bool(polyline.get("closed")) for polyline in prediction)})
    result["unmatched_vector_regions"] = sum(bool(region.get("polylines")) for index, region in enumerate(references)
        if index not in ref_matched) + sum(bool(region.get("polylines")) for index, region in enumerate(predictions)
        if index not in pred_matched)
    return result


def load_layout(path: Path) -> dict:
    layout = json.loads(path.read_text(encoding="utf-8"))
    for region in layout["regions"]:
        for field, target in (("vector_file", "polylines"), ("table_file", "table")):
            if region.get(field):
                artifact = (path.parent / region[field]).resolve()
                if not artifact.is_relative_to(path.parent.resolve()):
                    raise ValueError("Artifact path escapes the page directory")
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                region[target] = payload["polylines"] if target == "polylines" else payload
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-region-f1", type=float)
    parser.add_argument("--max-cer", type=float)
    parser.add_argument("--min-cell-f1", type=float)
    parser.add_argument("--max-vector-error", type=float)
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    result = evaluate_layout(load_layout(args.expected), load_layout(args.actual), args.iou)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    failed = args.min_region_f1 is not None and result["regions"]["f1"] < args.min_region_f1
    failed |= args.min_cell_f1 is not None and result["table_cells"]["f1"] < args.min_cell_f1
    if args.max_cer is not None:
        failed |= result["text"]["cer"] is None or result["text"]["cer"] > args.max_cer
    if args.max_vector_error is not None:
        failed |= bool(result["unmatched_vector_regions"]) or any(
            item["hausdorff_px"] is None or item["hausdorff_px"] > args.max_vector_error for item in result["vectors"])
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())