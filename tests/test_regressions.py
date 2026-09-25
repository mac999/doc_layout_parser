import tempfile
import copy
import json
import base64
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import fitz
import numpy as np
import cv2
from pipeline.vlm import VlmSession, _parse_label
from pipeline.config import DEFAULTS, load_config, validate_config, preflight
from pipeline.loader import load_pages, maybe_upscale, _bezier_points, PageData
from shapely.geometry import LineString
from pipeline.regions import mask_text, reclassify_text_regions
from pipeline.vectorize import vectorize_region, native_vectors_for_region, native_coverage
from pipeline.ocr import classify_text, get_text_items, merge_text_words, _ocr_page
from pipeline.table import _merge_spans, _assign_cell_text
from pipeline.evaluate import evaluate_layout
from pipeline.export import save_json, validate_layout, output_validator, export_page
from viewer import create_app


class ContractTests(unittest.TestCase):
    def test_missing_fields_bounds_and_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = PageData(1, np.full((100, 100, 3), 255, np.uint8))
            region = {"id": "r001", "type": "drawing", "bbox": [10, 10, 90, 90],
                      "source": "heuristic", "confidence": 0.5, "confidence_kind": "heuristic_score"}
            layout = export_page(root, page, [region], {}, [], load_config())
            missing = copy.deepcopy(layout)
            del missing["schema_version"]
            self.assertFalse(output_validator().is_valid(missing))
            invalid = copy.deepcopy(layout)
            invalid["regions"][0]["bbox"][2] = 101
            with self.assertRaisesRegex(ValueError, "Bounding box"):
                validate_layout(invalid, root, {})
            invalid = copy.deepcopy(layout)
            invalid["regions"][0]["vector_file"] = "vectors/missing.json"
            with self.assertRaisesRegex(ValueError, "artifact"):
                validate_layout(invalid, root, {})
            invalid = copy.deepcopy(layout)
            invalid["regions"][0]["table"] = {"rows": 1, "cols": 1, "num_cells": 2,
                "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 1, "bbox": [10, 10, 90, 90]}] * 2}
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_layout(invalid, root, {})
            with self.assertRaisesRegex(ValueError, "Vector outside"):
                validate_layout(layout, root, {"r001": [{"points": [[10, 10], [100, 100]],
                    "num_points": 2, "group": 0, "closed": False}]})


class ViewerTests(unittest.TestCase):
    def test_failure_records_and_incomplete_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("document", ".staging", "_failures"):
                (root / name).mkdir()
            result = {"pages": [{"page": 1, "dir": "page_001"}], "num_pages": 1, "source_file": "scan.png"}
            save_json(root / "document" / "result.json", result)
            save_json(root / ".staging" / "result.json", result)
            save_json(root / "_failures" / "attempt.json", {"source_file": "scan.png", "status": "failed"})
            client = create_app(root).test_client()
            response = client.get("/api/files").get_json()
            self.assertEqual(len(response["files"]), 1)
            self.assertEqual(response["files"][0]["status"], "incomplete")
            self.assertEqual(len(response["failures"]), 1)
            self.assertEqual(client.get("/").status_code, 200)


class EvaluationTests(unittest.TestCase):
    def test_known_scores_and_duplicate_penalty(self):
        reference = {"size": {"width": 100, "height": 100}, "regions": [
            {"type": "text", "bbox": [0, 0, 40, 20], "text": "ABC"},
            {"type": "drawing", "bbox": [0, 30, 90, 90], "polylines": [
                {"points": [[10, 40], [80, 80]], "group": 0, "closed": False}]}]}
        perfect = evaluate_layout(reference, reference)
        self.assertEqual(perfect["regions"]["f1"], 1)
        self.assertEqual(perfect["text"]["cer"], 0)
        self.assertEqual(perfect["vectors"][0]["hausdorff_px"], 0)
        prediction = copy.deepcopy(reference)
        prediction["regions"][0]["text"] = "ADC"
        self.assertAlmostEqual(evaluate_layout(reference, prediction)["text"]["cer"], 1 / 3)
        prediction["regions"] *= 2
        self.assertEqual(evaluate_layout(reference, prediction)["regions"]["precision"], 0.5)
        missing = evaluate_layout(reference, {**reference, "regions": []})
        self.assertEqual(missing["regions"]["recall"], 0)
        self.assertEqual(missing["text"]["cer"], 1)
        self.assertEqual(missing["unmatched_vector_regions"], 1)


class VlmTests(unittest.TestCase):
    def test_provider_image_limits_and_budget(self):
        for provider in ("ollama", "openai", "gemini"):
            cfg = load_config()
            cfg["classify"].update(provider=provider, vlm_max_image_side=32, max_calls_per_document=1)
            session = VlmSession()
            with patch("pipeline.vlm._request_label", return_value="drawing") as request:
                result = session.classify(np.zeros((100, 1000, 3), np.uint8), cfg)
                self.assertEqual(result["label"], "drawing")
                encoded, settings, timeout = request.call_args.args
                image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
                self.assertLessEqual(max(image.shape[:2]), 32)
                self.assertLessEqual(timeout, cfg["classify"]["timeout_sec"])
                self.assertEqual(session.classify(image, cfg)["reason"], "budget_exhausted")
                request.assert_called_once()

    def test_failures_and_invalid_responses_open_circuit(self):
        cfg = load_config()
        image = np.zeros((10, 10, 3), np.uint8)
        session = VlmSession()
        with patch("pipeline.vlm._request_label", side_effect=[TimeoutError(), "drawing or image"]) as request:
            self.assertEqual(session.classify(image, cfg)["reason"], "TimeoutError")
            self.assertEqual(session.classify(image, cfg)["reason"], "invalid_response")
            self.assertEqual(session.classify(image, cfg)["reason"], "circuit_open")
            self.assertEqual(request.call_count, 2)
        self.assertIsNone(_parse_label("not an image"))
        session = VlmSession(elapsed_sec=119)
        with patch("pipeline.vlm._request_label", return_value="image") as request:
            session.classify(image, cfg)
            self.assertEqual(request.call_args.args[2], 1)


class ConfigTests(unittest.TestCase):
    def test_cpu_fallback_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "scan.png"
            source.touch()
            torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict("sys.modules", {"torch": torch_stub}), \
                    patch("pipeline.config.importlib.util.find_spec", return_value=True):
                cfg, warnings = preflight(load_config(), [source])
                self.assertFalse(cfg["ocr"]["gpu"])
                self.assertTrue(any("CPU" in warning for warning in warnings))
                cfg["ocr"].update(gpu=True, cpu_fallback=False)
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    preflight(cfg, [source])
                with self.assertRaises(FileNotFoundError):
                    preflight(load_config(), [source.with_name("missing.png")])

    def test_defaults_and_repository_config(self):
        validate_config(load_config())
        validate_config(load_config(Path(__file__).parents[1] / "config.json"))
        first, second = load_config(), load_config()
        first["ocr"]["languages"].append("test")
        self.assertNotIn("test", second["ocr"]["languages"])

    def test_rejects_invalid_settings(self):
        for section, key, value in [("pdf", "render_dpi", 0), ("ocr", "paragraph", True),
                                    ("layout", "dilate_kernel", -1), ("ocr", "min_confidence", 2),
                                    ("classify", "provider", "invalid"), ("pdf", "render_dpi", True),
                                    ("ocr", "typo", 10), ("ocr", "rotation_info", [45])]:
            with self.subTest(section=section, key=key):
                cfg = load_config()
                cfg[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(temporary) / "missing.json")


class NativeVectorTests(unittest.TestCase):
    def test_partial_native_content_uses_raster(self):
        image = np.full((200, 200, 3), 255, np.uint8)
        image[70:73, 30:170] = 0
        image[71:150, 100:103] = 0
        page = PageData(1, image, native_polylines=[{"points": [[30, 71], [169, 71]], "closed": False}])
        cfg = load_config()
        cfg["table"]["enable"] = False
        cfg["layout"]["min_region_area"] = 100
        with tempfile.TemporaryDirectory() as temporary, patch.object(main, "get_text_items", return_value=[]):
            layout = main.process_page(page, cfg, Path(temporary))
        drawing = next(region for region in layout["regions"] if region["type"] == "drawing")
        self.assertEqual(drawing["vector_source"], "raster")
        self.assertLess(drawing["native_coverage"], 0.9)

    def test_curve_approximation_error(self):
        control = [(0, 0), (0, 1000), (1000, 1000), (1000, 0)]
        sampled = LineString(_bezier_points(*control, 8, tolerance=0.25))
        reference = LineString(_bezier_points(*control, 10001, tolerance=0.001))
        self.assertLessEqual(sampled.hausdorff_distance(reference), 0.25)

    def test_page_export_native_and_raster_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for native in (False, True):
                with self.subTest(native=native):
                    image = np.full((200, 200, 3), 255, dtype=np.uint8)
                    image[70:73, 30:170] = 0
                    page = PageData(1, image, native_polylines=[
                        {"points": [[30, 71], [169, 71]], "closed": False}] if native else [])
                    cfg = copy.deepcopy(DEFAULTS)
                    cfg["table"]["enable"] = False
                    cfg["layout"]["min_region_area"] = 100
                    target = Path(temporary) / str(native)
                    with patch.object(main, "get_text_items", return_value=[]):
                        layout = main.process_page(page, cfg, target)
                    drawings = [region for region in layout["regions"] if region["type"] == "drawing"]
                    self.assertEqual(len(drawings), 1)
                    region = drawings[0]
                    self.assertEqual(region["vector_source"], "pdf_native" if native else "raster")
                    vectors = json.loads((target / region["vector_file"]).read_text())
                    self.assertEqual(vectors["num_polylines"], 1)
                    self.assertTrue((target / region["svg_file"]).exists())

    def test_clip_dedupe_exclude_and_group(self):
        native = [{"points": [[-10, 50], [110, 50]], "closed": False},
                  {"points": [[110, 50], [-10, 50]], "closed": False},
                  {"points": [[10, 10], [30, 10], [30, 30], [10, 30]], "closed": True}]
        result = native_vectors_for_region(native, [0, 0, 100, 100], [], DEFAULTS,
                                            exclude_bboxes=[[40, 40, 60, 60]])
        self.assertAlmostEqual(sum(item["length_px"] for item in result), 160)
        self.assertEqual(sum(item["closed"] for item in result), 1)
        self.assertEqual(len({item["group"] for item in result}), 3)
        self.assertTrue(all(item["source"] == "pdf_native" for item in result))
        self.assertEqual(native_vectors_for_region(native, [0, 0, 100, 100], [], DEFAULTS,
                                                   exclude_bboxes=[[40, 40, 60, 60]], claimed=result), [])

    def test_component_and_word_exclusion(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:61, 10:91] = True
        native = [{"points": [[0, 50], [100, 50]], "closed": False},
                  {"points": [[0, 20], [100, 20]], "closed": False}]
        result = native_vectors_for_region(native, [0, 0, 100, 100],
                                            [{"bbox": [40, 40, 60, 60]}], DEFAULTS, mask)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(sum(item["length_px"] for item in result), 52)

    def test_intersections_share_connectivity_group(self):
        native = [{"points": [[10, 50], [90, 50]]}, {"points": [[50, 10], [50, 90]]}]
        result = native_vectors_for_region(native, [0, 0, 100, 100], [], DEFAULTS)
        self.assertEqual(len(result), 4)
        self.assertEqual(len({item["group"] for item in result}), 1)


class TableTests(unittest.TestCase):
    def test_all_small_grid_merges_have_single_ownership(self):
        for rows, columns in ((2, 2), (2, 3)):
            right_edges = [(row, column) for row in range(rows) for column in range(columns - 1)]
            down_edges = [(row, column) for row in range(rows - 1) for column in range(columns)]
            for mask in range(1 << (len(right_edges) + len(down_edges))):
                right = {edge for index, edge in enumerate(right_edges) if mask & (1 << index)}
                down = {edge for index, edge in enumerate(down_edges)
                        if mask & (1 << (index + len(right_edges)))}
                cells = _merge_spans(rows, columns, right, down)
                occupancy = np.zeros((rows, columns), dtype=int)
                for cell in cells:
                    top, left = cell["row"], cell["col"]
                    bottom, end = top + cell["row_span"], left + cell["col_span"]
                    occupancy[top:bottom, left:end] += 1
                    cell["bbox"] = [left * 10, top * 10, end * 10, bottom * 10]
                np.testing.assert_array_equal(occupancy, np.ones((rows, columns)))
                _assign_cell_text(cells, [{"bbox": [11, 11, 19, 19], "text": "unique"}])
                self.assertEqual(sum(cell["text"] == "unique" for cell in cells), 1)

    def test_l_shape_is_flagged_and_rectangle_still_merges(self):
        cells = _merge_spans(2, 2, {(0, 0)}, {(0, 0)})
        self.assertEqual(len(cells), 4)
        self.assertEqual(sum(cell.get("merge_uncertain", False) for cell in cells), 3)
        cells = _merge_spans(2, 2, {(0, 0), (1, 0)}, {(0, 0)})
        self.assertEqual(cells, [{"row": 0, "col": 0, "row_span": 2, "col_span": 2}])


class HybridTextTests(unittest.TestCase):
    def test_native_title_does_not_skip_scanned_text(self):
        title = {"text": "Title", "bbox": [10, 10, 40, 20], "confidence": 1.0}
        scanned = {"text": "scanned", "bbox": [10, 100, 60, 110], "confidence": 0.9}
        page = SimpleNamespace(image=np.full((200, 200, 3), 255, np.uint8),
                               native_words=[dict(title) for _ in range(5)])
        with patch("pipeline.ocr.run_ocr", return_value=[title, scanned]) as ocr:
            items = get_text_items(page, DEFAULTS)
        ocr.assert_called_once()
        self.assertEqual([item["text"] for item in items], ["Title", "scanned"])
        self.assertEqual(items[0]["words"][0]["source"], "pdf_native")
        self.assertEqual(items[1]["source"], "ocr")

    def test_native_and_ocr_segmentation_duplicates(self):
        native = [{"text": text, "bbox": bbox, "source": "pdf_native", "confidence": 1.0}
                  for text, bbox in [("Hello", [0, 0, 30, 10]), ("world", [35, 0, 65, 10])]]
        incoming = {"text": "Hello world", "bbox": [0, 0, 65, 10], "source": "ocr", "confidence": 0.9}
        self.assertEqual(len(merge_text_words(native, [incoming])), 2)
        incoming.update(text="Hello world extra", bbox=[0, 0, 100, 10])
        result = merge_text_words(native, [incoming])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello world extra")
        self.assertEqual(len(result[0]["native_matches"]), 2)

    def test_tiled_ocr_maps_offsets(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["ocr"].update(tile_size=100, tile_overlap=0)
        detected = [{"text": "word", "bbox": [10, 10, 20, 20], "confidence": 0.9}]
        with patch("pipeline.ocr.run_ocr", return_value=detected):
            words = _ocr_page(np.zeros((100, 200, 3), np.uint8), cfg)
        self.assertEqual([word["bbox"] for word in words], [[10, 10, 20, 20], [110, 10, 120, 20]])


class TextGeometryTests(unittest.TestCase):
    def test_word_masks_preserve_intervening_line(self):
        ink = np.zeros((60, 100), dtype=np.uint8)
        ink[30, 35:66] = 255
        ink[20:35, 10:20] = 255
        ink[20:35, 80:90] = 255
        items = [{"bbox": [10, 20, 90, 35], "words": [
            {"bbox": [10, 20, 20, 35]}, {"bbox": [80, 20, 90, 35]}]}]
        masked = mask_text(ink, items, 2)
        self.assertEqual(np.count_nonzero(masked), 31)
        image = np.repeat((255 - ink)[:, :, None], 3, axis=2)
        vectors = vectorize_region(image, [5, 5, 95, 55], items, DEFAULTS)
        self.assertEqual(len(vectors), 1)
        self.assertEqual({tuple(point) for point in vectors[0]["points"]}, {(35, 30), (65, 30)})

    def test_dimension_semantics_survive_context(self):
        for text in ("1200", "R25", "\u230050", "25\u00b10.1"):
            with self.subTest(text=text):
                regions = [{"type": classify_text(text), "bbox": [20, 20, 60, 40], "source": "ocr"},
                           {"type": "drawing", "bbox": [0, 0, 100, 100]}]
                reclassify_text_regions(regions, DEFAULTS)
                self.assertEqual(regions[0]["type"], "dimension")
                self.assertEqual(regions[0]["spatial_context"], "in_drawing")
                regions.append({"type": "table", "bbox": [0, 0, 100, 100]})
                reclassify_text_regions(regions, DEFAULTS)
                self.assertEqual(regions[0]["type"], "text")
                self.assertEqual(regions[0]["semantic_type"], "dimension")
        self.assertEqual(classify_text("#3"), "annotation")
        self.assertEqual(classify_text("NOTE"), "annotation")


class CoordinateTests(unittest.TestCase):
    def test_pdf_rotation_cropbox_and_upscale(self):
        with tempfile.TemporaryDirectory() as temporary:
            for rotation in (0, 90, 180, 270):
                for cropped in (False, True):
                    for upscale in (0, 401):
                        with self.subTest(rotation=rotation, cropped=cropped, upscale=upscale):
                            source = Path(temporary) / "coordinates.pdf"
                            with fitz.open() as document:
                                pdf_page = document.new_page(width=300, height=200)
                                pdf_page.insert_text((120, 100), "TEST")
                                pdf_page.draw_line((100, 130), (220, 130), width=2)
                                if cropped:
                                    pdf_page.set_cropbox(fitz.Rect(30, 20, 280, 190))
                                pdf_page.set_rotation(rotation)
                                document.save(source)
                            cfg = copy.deepcopy(DEFAULTS)
                            cfg["pdf"]["render_dpi"] = 72
                            cfg["preprocess"]["upscale_target_min_side"] = upscale
                            page = maybe_upscale(list(load_pages(source, cfg))[0], cfg)
                            height, width = page.image.shape[:2]
                            for word in page.native_words:
                                left, top, right, bottom = word["bbox"]
                                self.assertTrue(0 <= left < right <= width)
                                self.assertTrue(0 <= top < bottom <= height)
                                self.assertLess(page.image[int(top):int(bottom), int(left):int(right)].min(), 100)
                            self.assertEqual(len(page.native_words), 1)
                            matrix = fitz.Matrix(page.coordinate_transform["source_to_pixel"])
                            inverse = fitz.Matrix(page.coordinate_transform["pixel_to_source"])
                            original = fitz.Point(100 - (30 if cropped else 0), 130 - (20 if cropped else 0))
                            actual = fitz.Point(page.native_polylines[0]["points"][0])
                            self.assertLess(abs(actual - original * matrix), 1)
                            self.assertLess(abs(actual * inverse - original), 0.01)
                            for x_coord, y_coord in page.native_polylines[0]["points"]:
                                x_pixel, y_pixel = round(x_coord), round(y_coord)
                                self.assertLess(page.image[max(0, y_pixel-1):y_pixel+2,
                                                           max(0, x_pixel-1):x_pixel+2].min(), 100)


class OutputTests(unittest.TestCase):
    def test_publication_failure_restores_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sheet.pdf"
            source.write_bytes(b"fixture")
            target = root / main.output_key(source)
            target.mkdir()
            (target / "previous").touch()
            original_rename = Path.rename

            def rename(path, destination):
                if destination == target and path != target and ".backup-" not in path.name:
                    raise OSError("Publish failed")
                return original_rename(path, destination)

            with patch.object(main, "_process_file", return_value={}), patch.object(Path, "rename", rename):
                with self.assertRaisesRegex(OSError, "Publish failed"):
                    main.process_file(source, load_config(), root)
            self.assertTrue((target / "previous").exists())

    def test_invalid_config_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            cfg = load_config()
            cfg["pdf"]["render_dpi"] = -1
            with self.assertRaises(ValueError):
                main.process_file(Path(temporary) / "missing.pdf", cfg, root)
            self.assertFalse(root.exists())

    def test_distinct_input_keys(self):
        self.assertNotEqual(main.output_key(Path("sheet.jpg")), main.output_key(Path("sheet.pdf")))
        self.assertNotEqual(main.output_key(Path("one/sheet.pdf")), main.output_key(Path("two/sheet.pdf")))

    def test_failed_page_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sheet.pdf"
            source.write_bytes(b"fixture")
            target = root / main.output_key(source)
            target.mkdir()
            (target / "result.json").write_text("previous", encoding="utf-8")
            page = SimpleNamespace(page_no=1)
            with patch.object(main, "load_pages", return_value=(item for item in [page, page])), \
                    patch.object(main, "maybe_upscale", side_effect=lambda page, cfg: page), \
                    patch.object(main, "process_page", side_effect=[
                        {"regions": [], "num_regions": 0, "size": [10, 10]},
                        RuntimeError("page failure")]):
                with self.assertRaisesRegex(RuntimeError, "page failure"):
                    main.process_file(source, load_config(), root)
            self.assertEqual((target / "result.json").read_text(), "previous")
            self.assertEqual(len(list((root / "_failures").glob("*.json"))), 1)
            failure = json.loads(next((root / "_failures").glob("*.json")).read_text())
            self.assertEqual([page["status"] for page in failure["pages"]], ["complete", "failed"])
            self.assertEqual(failure["status"], "failed")

    def test_success_replaces_stale_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sheet.pdf"
            source.write_bytes(b"fixture")
            target = root / main.output_key(source)
            target.mkdir()
            (target / "stale").touch()
            with patch.object(main, "load_pages", return_value=(page for page in [SimpleNamespace(page_no=1)])), \
                    patch.object(main, "maybe_upscale", side_effect=lambda page, cfg: page), \
                    patch.object(main, "process_page", return_value={
                        "regions": [], "num_regions": 0, "size": [10, 10]}):
                result = main.process_file(source, load_config(), root)
            self.assertFalse((target / "stale").exists())
            self.assertTrue((target / "result.json").exists())
            self.assertEqual(len(result["input"]["sha256"]), 64)
            self.assertEqual(result["config"], load_config())
            self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()