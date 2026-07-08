"""VLM (vision language model) based region classification: drawing vs image.

The provider is selected in config.json: ollama (local models such as llava,
fits in 8 GB VRAM), openai, or gemini. Calling the VLM only for regions where
the heuristic is uncertain is recommended (classify.ambiguous_only).
API keys are read from environment variables (OPENAI_API_KEY, GOOGLE_API_KEY).
"""
import base64
import os

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
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".png", crop)
    return base64.b64encode(buf.tobytes()).decode()


def _parse_label(answer: str) -> str | None:
    a = answer.strip().lower()
    if "drawing" in a and "image" not in a:
        return "drawing"
    if "image" in a or "photo" in a:
        return "image"
    return None


def classify_with_vlm(crop: np.ndarray, cfg: dict) -> str | None:
    """Classify a region crop with a VLM. Returns None on failure so the
    caller keeps the heuristic result."""
    c = cfg["classify"]
    provider = c["provider"]
    try:
        if provider == "ollama":
            import ollama
            resp = ollama.chat(
                model=c["ollama_model"],
                messages=[{"role": "user", "content": _PROMPT,
                           "images": [_crop_to_png_b64(crop)]}],
                options={"temperature": 0},
            )
            return _parse_label(resp["message"]["content"])

        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                return None
            from openai import OpenAI
            client = OpenAI(timeout=c["timeout_sec"])
            resp = client.chat.completions.create(
                model=c["openai_model"],
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{_crop_to_png_b64(crop)}"}},
                ]}],
                max_tokens=10, temperature=0,
            )
            return _parse_label(resp.choices[0].message.content or "")

        if provider == "gemini":
            if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
                return None
            from google import genai
            from google.genai import types
            client = genai.Client()
            ok, buf = cv2.imencode(".png", crop)
            resp = client.models.generate_content(
                model=c["gemini_model"],
                contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png"), _PROMPT],
            )
            return _parse_label(resp.text or "")
    except Exception as e:
        print(f"  [VLM] {provider} call failed, keeping heuristic result: {e}")
    return None
