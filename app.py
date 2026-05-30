from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from inference_pipeline import DenseNet121InferencePipeline, discover_class_names


BASE_DIR = Path(__file__).resolve().parent


def _detect_default_model_path() -> str:
    for filename in ("best_dencenet122.pth", "best_densenet122.pth"):
        candidate = BASE_DIR / filename
        if candidate.exists():
            return str(candidate)
    return str(BASE_DIR / "best_densenet122.pth")


MODEL_PATH = os.environ.get("MODEL_PATH", _detect_default_model_path())
TRAIN_DIR = os.environ.get("TRAIN_DIR", str(BASE_DIR / "data/train"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

pipeline = DenseNet121InferencePipeline(
    model_path=MODEL_PATH,
    class_names=discover_class_names(TRAIN_DIR),
)


def _data_url(image_bytes: bytes, mimetype: str | None) -> str:
    image_type = mimetype or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{image_type};base64,{encoded}"


def _read_uploaded_image() -> tuple[Image.Image, bytes, str | None]:
    if "image" not in request.files:
        raise ValueError("No image part found in request.")

    uploaded_file = request.files["image"]
    if not uploaded_file.filename:
        raise ValueError("No image selected.")

    image_bytes = uploaded_file.read()
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("File is not a valid image.") from exc

    return image, image_bytes, uploaded_file.mimetype


def _template_payload(**kwargs: Any) -> dict[str, Any]:
    return {
        "model_path": MODEL_PATH,
        "class_names": pipeline.class_names,
        "error": None,
        "result": None,
        "preview_data_url": None,
        **kwargs,
    }


@app.route("/", methods=["GET"])
def home() -> str:
    return render_template("index.html", **_template_payload())


@app.route("/predict", methods=["POST"])
def predict_web() -> str:
    try:
        image, image_bytes, mimetype = _read_uploaded_image()
        prediction = pipeline.predict(image)
    except ValueError as exc:
        return render_template("index.html", **_template_payload(error=str(exc))), 400

    return render_template(
        "index.html",
        **_template_payload(
            result=prediction,
            preview_data_url=_data_url(image_bytes, mimetype),
        ),
    )


@app.route("/api/predict", methods=["POST"])
def predict_api() -> tuple[Any, int] | Any:
    try:
        image, _, _ = _read_uploaded_image()
        prediction = pipeline.predict(image)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, **prediction})


@app.errorhandler(413)
def too_large(_error: Any) -> tuple[str, int]:
    return render_template(
        "index.html",
        **_template_payload(error=f"Image too large. Max allowed size is {MAX_UPLOAD_MB}MB."),
    ), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
