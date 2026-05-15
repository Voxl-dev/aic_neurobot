#!/usr/bin/env python3
"""Inferencia rapida para el Keypoint Estimator de AIC."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_training_module(repo_root: Path):
    module_path = repo_root / "scripts" / "train_keypoint_estimator.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No existe el modulo de entrenamiento: {module_path}")

    spec = importlib.util.spec_from_file_location("aic_train_keypoint_estimator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo importar {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preprocess_image(image_bgr: np.ndarray, out_height: int, out_width: int, train_mod) -> np.ndarray:
    resized = cv2.resize(image_bgr, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb = (image_rgb - train_mod.IMAGENET_MEAN) / train_mod.IMAGENET_STD
    chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    return chw


def draw_keypoints(image_bgr: np.ndarray, keypoints_xy: np.ndarray) -> np.ndarray:
    debug = image_bgr.copy()
    for idx, (x, y) in enumerate(keypoints_xy):
        pt = (int(round(float(x))), int(round(float(y))))
        cv2.circle(debug, pt, 5, (0, 255, 0), -1)
        cv2.putText(
            debug,
            str(idx),
            (pt[0] + 5, pt[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return debug


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predice 9 keypoints 2D con un checkpoint entrenado.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--save_debug", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    train_mod = load_training_module(repo_root)
    torch_mod, _, _, _, _ = train_mod.require_torch()

    device = args.device or ("cuda" if torch_mod.cuda.is_available() else "cpu")
    checkpoint = torch_mod.load(args.checkpoint, map_location=device, weights_only=False)
    image_height = int(checkpoint.get("image_height", 256))
    image_width = int(checkpoint.get("image_width", 288))
    num_keypoints = int(checkpoint.get("num_keypoints", 9))

    model = train_mod.build_model(num_keypoints=num_keypoints, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {args.image}")

    orig_height, orig_width = image_bgr.shape[:2]
    tensor = preprocess_image(image_bgr, image_height, image_width, train_mod)
    tensor = torch_mod.from_numpy(tensor).unsqueeze(0).to(device)

    with torch_mod.no_grad():
        pred_norm = model(tensor).detach().cpu().numpy().reshape(num_keypoints, 2)

    pred_resized = pred_norm.copy()
    pred_resized[:, 0] *= float(image_width)
    pred_resized[:, 1] *= float(image_height)

    pred_original = pred_resized.copy()
    pred_original[:, 0] *= float(orig_width) / float(image_width)
    pred_original[:, 1] *= float(orig_height) / float(image_height)

    output = {
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "connector_type": checkpoint.get("connector_type"),
        "device": device,
        "model_input_size": {"height": image_height, "width": image_width},
        "original_image_size": {"height": orig_height, "width": orig_width},
        "keypoints_xy": pred_original.astype(float).tolist(),
        "keypoints_xy_model_input": pred_resized.astype(float).tolist(),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as fh:
        json.dump(output, fh, indent=2)

    if args.save_debug is not None:
        args.save_debug.parent.mkdir(parents=True, exist_ok=True)
        debug = draw_keypoints(image_bgr, pred_original)
        ok = cv2.imwrite(str(args.save_debug), debug)
        if not ok:
            raise RuntimeError(f"No se pudo guardar debug image en {args.save_debug}")

    print(f"Keypoints guardados en: {args.output_json}")
    if args.save_debug is not None:
        print(f"Debug image guardada en: {args.save_debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
