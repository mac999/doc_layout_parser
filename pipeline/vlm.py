"""VLM (vision language model) based region classification: drawing vs image.

The provider is selected in config.json: ollama (local models such as llava,
fits in 8 GB VRAM), openai, or gemini. Calling the VLM only for regions where
the heuristic is uncertain is recommended (classify.ambiguous_only).
API keys are read from environment variables (OPENAI_API_KEY, GOOGLE_API_KEY).
"""
import base64
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

_PROMPT = (
    "Look at this image region extracted from an engineering document. "
    "Answer with exactly one word:\n"
    "- 'drawing' if it is a line-based technical/CAD drawing, diagram, plan, section or schematic\n"
    "- 'image' if it is a photograph, rendering, logo or continuous-tone raster picture\n"
    "Answer:"
)


def _crop_to_png_b64(crop: np.ndarray, max_side: int = 768) -> str:
    """Downscale a crop if needed and encode it as base64 PNG."""
    h, w = crop.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode()


def _parse_label(answer: str) -> str | None:
    label = answer.strip().lower().strip(".\"'` ")
    return label if label in {"drawing", "image"} else None


def _request_label(image: str, settings: dict, timeout: float) -> str:
    provider = settings["provider"]
    model = settings[f"{provider}_model"]
    if provider == "ollama":
        from ollama import Client
        client = Client(timeout=timeout)
        response = client.chat(model=model, messages=[{
            "role": "user", "content": _PROMPT, "images": [image]}],
            options={"temperature": 0, "num_predict": 10})
        return response["message"]["content"]
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Missing API key")
        from openai import OpenAI
        with OpenAI(timeout=timeout, max_retries=0) as client:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                ]}], max_tokens=10, temperature=0)
            return response.choices[0].message.content or ""
    if provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key")
        from google import genai
        from google.genai import types
        with genai.Client(api_key=api_key, http_options=types.HttpOptions(
                timeout=max(1, int(timeout * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1))) as client:
            response = client.models.generate_content(
                model=model, contents=[types.Part.from_bytes(
                    data=base64.b64decode(image), mime_type="image/png"), _PROMPT],
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=16))
            return response.text or ""
    raise ValueError("Invalid VLM provider")


@dataclass
class VlmSession:
    calls: int = 0
    failures: int = 0
    elapsed_sec: float = 0.0

    def classify(self, crop: np.ndarray, cfg: dict) -> dict:
        settings = cfg["classify"]
        provider = settings["provider"]
        result = {"provider": provider, "model": settings[f"{provider}_model"],
                  "label": None, "fallback": True, "status": "skipped"}
        remaining = settings.get("budget_sec", 120) - self.elapsed_sec
        if self.calls >= settings.get("max_calls_per_document", 20) or remaining <= 0:
            return {**result, "reason": "budget_exhausted"}
        if self.failures >= settings.get("max_failures", 2):
            return {**result, "reason": "circuit_open"}
        self.calls += 1
        started = time.monotonic()
        try:
            image = _crop_to_png_b64(crop, settings.get("vlm_max_image_side", 768))
            answer = _request_label(image, settings, min(settings["timeout_sec"], remaining))
            label = _parse_label(answer)
            if label is None:
                self.failures += 1
                result.update(status="failed", reason="invalid_response")
            else:
                self.failures = 0
                result.update(status="ok", label=label, fallback=False)
        except Exception as error:
            self.failures += 1
            result.update(status="failed", reason=type(error).__name__)
            print(f"  [VLM] {provider}: {type(error).__name__}; using heuristic")
        finally:
            duration = time.monotonic() - started
            self.elapsed_sec += duration
            result["elapsed_sec"] = round(duration, 3)
        return result


def classify_with_vlm(crop: np.ndarray, cfg: dict) -> str | None:
    return VlmSession().classify(crop, cfg)["label"]
