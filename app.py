from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import torch
import os
import uuid

from config import Config
from model import build_model, load_trained_model
from utils import (
    preprocess_image_for_inference,
    save_prediction_outputs,
    ensure_dir,
)

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY

ensure_dir(Config.OUTPUT_DIR)

device = torch.device(Config.DEVICE)
model = build_model()
model = load_trained_model(model, Config.MODEL_PATH, device)
model.eval()


@app.get("/")
def home():
    return jsonify(
        message="House segmentation service is running.",
        endpoints={
            "health": "GET /health",
            "predict": "POST /predict with JSON {'image_path': '...'}"
        }
    )


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_path=Config.MODEL_PATH,
        device=Config.DEVICE,
        image_size=Config.IMAGE_SIZE
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path")

    if not image_path or not isinstance(image_path, str):
        return jsonify(
            error='Invalid input. Provide JSON: {"image_path": "path/to/image.png"}'
        ), 400

    if not os.path.exists(image_path):
        return jsonify(error=f"Image file not found: {image_path}"), 400

    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Failed to open image: {str(e)}"), 400

    image_tensor, original_np = preprocess_image_for_inference(
        original_image,
        image_size=Config.IMAGE_SIZE,
        device=device
    )

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > Config.THRESHOLD).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255

    run_id = uuid.uuid4().hex[:8]
    mask_path, overlay_path = save_prediction_outputs(
        original_np=original_np,
        pred_mask_np=pred_mask_np,
        output_dir=Config.OUTPUT_DIR,
        base_name=f"prediction_{run_id}"
    )

    return jsonify(
        status="ok",
        model_path=Config.MODEL_PATH,
        image_path=image_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
        threshold=Config.THRESHOLD
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=(Config.FLASK_ENV == "development"))