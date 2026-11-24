"""
Model Export Pipeline for Web & Android.
Converts PyTorch checkpoints to:
1. ONNX (for Web with ONNX Runtime Web / WebAssembly)
2. TorchScript & Android assets (for Android native / Flutter / React Native)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("ModelExport")


def export_models(
    weights_path: str = "backend/models_store/crop_disease_vision.pt",
    output_dir: str = "backend/models_store",
    class_indices_path: str = "backend/models_store/vision_class_indices.json",
):
    try:
        import torch
    except ImportError:
        logger.error("PyTorch required for model export.")
        return

    from training.model import HierarchicalAgriNet

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class indices or default
    num_crops, num_conditions = 10, 38
    if Path(class_indices_path).exists():
        with open(class_indices_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
            num_crops = len(idx_data.get("crop_to_idx", {}))
            num_conditions = len(idx_data.get("label_to_idx", {}))

    model = HierarchicalAgriNet(num_crops=num_crops, num_conditions=num_conditions, pretrained=False)
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning("No checkpoint found at %s. Exporting architecture with initialized weights.", weights_path)

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # 1. Export to ONNX (Web deployment)
    onnx_path = out_dir / "crop_disease_model.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["crop_logits", "condition_logits"],
            dynamic_axes={"input_image": {0: "batch_size"}, "crop_logits": {0: "batch_size"}, "condition_logits": {0: "batch_size"}},
        )
        logger.info("✓ Exported ONNX model for Web: %s (%.2f MB)", onnx_path, onnx_path.stat().st_size / (1024 * 1024))
    except Exception as e:
        logger.error("ONNX export failed: %s", e)

    # 2. Export TorchScript (Android PyTorch Mobile / Flutter)
    pt_mobile_path = out_dir / "crop_disease_model.ptl"
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(pt_mobile_path))
        logger.info("✓ Exported TorchScript model for Android: %s (%.2f MB)", pt_mobile_path, pt_mobile_path.stat().st_size / (1024 * 1024))
    except Exception as e:
        logger.error("TorchScript export failed: %s", e)

    # 3. Create Android Model Configuration Manifest
    android_meta = {
        "model_name": "Karnataka Agriculture AI Vision",
        "version": "1.0.0",
        "input": {"name": "input_image", "shape": [1, 3, 224, 224], "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
        "outputs": [
            {"name": "crop_logits", "classes": num_crops, "type": "crop_species"},
            {"name": "condition_logits", "classes": num_conditions, "type": "disease_pest_status"},
        ],
        "ood_rejection_threshold": 0.55,
        "supported_crops": ["rice", "tomato", "maize", "cotton", "sugarcane", "ragi", "arecanut", "coconut", "wheat", "all_other_crops"],
        "languages": ["kn", "en", "hi"],
    }
    with open(out_dir / "android_model_config.json", "w", encoding="utf-8") as f:
        json.dump(android_meta, f, indent=2)
    logger.info("✓ Saved Android configuration metadata to %s", out_dir / "android_model_config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="backend/models_store/crop_disease_vision.pt")
    parser.add_argument("--output-dir", default="backend/models_store")
    args = parser.parse_args()
    export_models(weights_path=args.weights, output_dir=args.output_dir)
