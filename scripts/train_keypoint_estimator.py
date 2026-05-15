#!/usr/bin/env python3
"""
Entrenamiento minimo del Keypoint Estimator para Dataset A.

El dataset se interpreta como una coleccion de muestras por camara:
  1 JSON + 3 imagenes  ->  3 samples de entrenamiento

Cada sample contiene:
  - una imagen RGB
  - 9 keypoints 2D del puerto objetivo

El modelo propuesto es MobileNetV3-Small con una cabeza de regresion
que predice 18 valores = 9 x (u, v).

Ejemplos:
  pixi run python scripts/train_keypoint_estimator.py \
    --dataset_dir ~/aic_datasets/dataset_A \
    --connector_type sfp \
    --output_dir outputs/keypoint_sfp

  pixi run python scripts/train_keypoint_estimator.py \
    --dataset_dir ~/aic_datasets/dataset_A \
    --connector_type sfp \
    --output_dir outputs/keypoint_sfp \
    --preview_augmentations 12
"""
"""
Entrenamiento minimo del Keypoint Estimator para Dataset A.

El dataset se interpreta como una coleccion de muestras por camara:
  1 JSON + 3 imagenes  ->  3 samples de entrenamiento

Cada sample contiene:
  - una imagen RGB
  - 9 keypoints 2D del puerto objetivo

El modelo propuesto es MobileNetV3-Small con una cabeza de regresion
que predice 18 valores = 9 x (u, v).

Ejemplos:
  pixi run python scripts/train_keypoint_estimator.py \
    --dataset_dir ~/aic_datasets/dataset_A \
    --connector_type sfp \
    --output_dir outputs/keypoint_sfp

  pixi run python scripts/train_keypoint_estimator.py \
    --dataset_dir ~/aic_datasets/dataset_A \
    --connector_type sfp \
    --output_dir outputs/keypoint_sfp \
    --preview_augmentations 12
"""

"""
Entrenamiento minimo del Keypoint Estimator para Dataset A.

El dataset se interpreta como una coleccion de muestras por camara:
  1 JSON + 3 imagenes  ->  3 samples de entrenamiento

Cada sample contiene:
  - una imagen RGB
  - 9 keypoints 2D del puerto objetivo

El modelo propuesto es MobileNetV3-Small con una cabeza de regresion
que predice 18 valores = 9 x (u, v).

Ejemplos:
  pixi run python scripts/train_keypoint_estimator.py \
    --dataset_dir ~/aic_datasets/dataset_A \
    --connector_type sfp \
    --output_dir outputs/keypoint_sfp

  pixi run python scripts/train_keypoint_estimator.py \
    --dataset_dir ~/aic_datasets/dataset_A \
    --connector_type sfp \
    --output_dir outputs/keypoint_sfp \
    --preview_augmentations 12
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

torch = None
nn = None
DataLoader = None
models = None
MobileNet_V3_Small_Weights = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAMERAS = ("left", "center", "right")


def require_torch():
    global torch, nn, DataLoader, models, MobileNet_V3_Small_Weights
    if torch is None:
        import torch as _torch
        from torch import nn as _nn
        from torch.utils.data import DataLoader as _DataLoader
        from torchvision import models as _models
        from torchvision.models import MobileNet_V3_Small_Weights as _weights

        torch = _torch
        nn = _nn
        DataLoader = _DataLoader
        models = _models
        MobileNet_V3_Small_Weights = _weights
    return torch, nn, DataLoader, models, MobileNet_V3_Small_Weights


@dataclass(frozen=True)
class SampleRecord:
    json_path: Path
    image_path: Path
    connector_type: str
    camera: str
    scene_idx: int
    sample_id: int
    keypoints_xy: np.ndarray  # (N, 2), float32, original image coords


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch_mod, _, _, _, _ = require_torch()
    except ModuleNotFoundError:
        return
    torch_mod.manual_seed(seed)
    torch_mod.cuda.manual_seed_all(seed)


def build_motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot_mat = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def affine_transform_image_and_keypoints(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    angle_deg: float,
    scale: float,
    tx_px: float,
    ty_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, scale)
    mat[0, 2] += tx_px
    mat[1, 2] += ty_px

    warped = cv2.warpAffine(
        image,
        mat,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    keypoints_h = np.concatenate(
        [keypoints_xy.astype(np.float32), np.ones((len(keypoints_xy), 1), dtype=np.float32)],
        axis=1,
    )
    warped_keypoints = (mat @ keypoints_h.T).T.astype(np.float32)
    return warped, warped_keypoints


def resize_image_and_keypoints(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    out_height: int,
    out_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    in_height, in_width = image.shape[:2]
    resized = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

    scale_x = out_width / float(in_width)
    scale_y = out_height / float(in_height)
    resized_keypoints = keypoints_xy.astype(np.float32).copy()
    resized_keypoints[:, 0] *= scale_x
    resized_keypoints[:, 1] *= scale_y
    return resized, resized_keypoints


class KeypointAugmenter:
    """Augmentations leves y plausibles para las camaras del robot."""

    def __init__(self, out_height: int, out_width: int) -> None:
        self.out_height = out_height
        self.out_width = out_width

    def __call__(self, image: np.ndarray, keypoints_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        aug_img = image.copy()
        aug_kps = keypoints_xy.astype(np.float32).copy()

        if random.random() < 0.9:
            height, width = aug_img.shape[:2]
            aug_img, aug_kps = affine_transform_image_and_keypoints(
                aug_img,
                aug_kps,
                angle_deg=random.uniform(-8.0, 8.0),
                scale=random.uniform(0.92, 1.08),
                tx_px=random.uniform(-0.06, 0.06) * width,
                ty_px=random.uniform(-0.06, 0.06) * height,
            )

        aug_img, aug_kps = resize_image_and_keypoints(
            aug_img, aug_kps, self.out_height, self.out_width
        )

        aug_img = self._apply_photometric_augmentations(aug_img)

        if random.random() < 0.25:
            aug_img = self._apply_cutout_near_keypoints(aug_img, aug_kps)

        return aug_img, aug_kps

    def _apply_photometric_augmentations(self, image: np.ndarray) -> np.ndarray:
        aug_img = image.astype(np.float32)

        if random.random() < 0.85:
            alpha = random.uniform(0.85, 1.20)  # contraste
            beta = random.uniform(-18.0, 18.0)  # brillo
            aug_img = aug_img * alpha + beta

        if random.random() < 0.50:
            gamma = random.uniform(0.85, 1.15)
            norm = np.clip(aug_img / 255.0, 0.0, 1.0)
            aug_img = (norm ** gamma) * 255.0

        aug_img = np.clip(aug_img, 0.0, 255.0).astype(np.uint8)

        if random.random() < 0.60:
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.85, 1.15)  # saturacion
            hsv[..., 2] *= random.uniform(0.90, 1.10)  # value
            hsv[..., 0] += random.uniform(-4.0, 4.0)   # hue
            hsv[..., 0] = np.mod(hsv[..., 0], 180.0)
            hsv[..., 1:] = np.clip(hsv[..., 1:], 0.0, 255.0)
            aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if random.random() < 0.35:
            sigma = random.uniform(1.0, 5.0)
            noise = np.random.normal(0.0, sigma, size=aug_img.shape).astype(np.float32)
            aug_img = np.clip(aug_img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        blur_draw = random.random()
        if blur_draw < 0.20:
            ksize = random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), sigmaX=0.0)
        elif blur_draw < 0.35:
            kernel = build_motion_blur_kernel(random.choice([3, 5, 7]), random.uniform(-30.0, 30.0))
            aug_img = cv2.filter2D(aug_img, -1, kernel)

        if random.random() < 0.25:
            quality = random.randint(45, 90)
            ok, enc = cv2.imencode(".jpg", aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    aug_img = dec

        return aug_img

    def _apply_cutout_near_keypoints(self, image: np.ndarray, keypoints_xy: np.ndarray) -> np.ndarray:
        aug_img = image.copy()
        height, width = aug_img.shape[:2]
        center = keypoints_xy.mean(axis=0)

        occ_width = int(random.uniform(0.08, 0.18) * width)
        occ_height = int(random.uniform(0.08, 0.18) * height)
        occ_cx = int(center[0] + random.uniform(-0.10, 0.10) * width)
        occ_cy = int(center[1] + random.uniform(-0.10, 0.10) * height)

        x1 = max(0, occ_cx - occ_width // 2)
        y1 = max(0, occ_cy - occ_height // 2)
        x2 = min(width, x1 + occ_width)
        y2 = min(height, y1 + occ_height)

        fill = np.array(
            [
                random.randint(0, 25),
                random.randint(0, 25),
                random.randint(0, 25),
            ],
            dtype=np.uint8,
        )
        aug_img[y1:y2, x1:x2] = fill
        return aug_img


def all_keypoints_visible(keypoints_xy: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.all(keypoints_xy[:, 0] >= 0.0)
        and np.all(keypoints_xy[:, 0] < float(width))
        and np.all(keypoints_xy[:, 1] >= 0.0)
        and np.all(keypoints_xy[:, 1] < float(height))
    )


def augment_until_valid(
    augmenter: KeypointAugmenter,
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    max_tries: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    last_img = image
    last_kps = keypoints_xy

    for _ in range(max_tries):
        aug_img, aug_kps = augmenter(image, keypoints_xy)
        last_img, last_kps = aug_img, aug_kps
        if all_keypoints_visible(aug_kps, width=width, height=height):
            return aug_img, aug_kps

    clipped = last_kps.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height - 1))
    return last_img, clipped


class KeypointDataset:
    def __init__(
        self,
        dataset_dir: Path,
        connector_type: str,
        split: str,
        out_height: int,
        out_width: int,
        val_ratio: float,
        seed: int,
        apply_augmentation: bool,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.connector_type = connector_type
        self.split = split
        self.out_height = out_height
        self.out_width = out_width
        self.apply_augmentation = apply_augmentation and split == "train"
        self.augmenter = KeypointAugmenter(out_height, out_width)
        self.records = self._load_records(val_ratio=val_ratio, seed=seed)

        if not self.records:
            raise RuntimeError(
                f"No se encontraron muestras validas para connector_type={connector_type!r} "
                f"en split={split!r} dentro de {dataset_dir}"
            )

    def _load_records(self, val_ratio: float, seed: int) -> List[SampleRecord]:
        json_paths = sorted(self.dataset_dir.glob("*.json"))
        all_records: List[SampleRecord] = []

        for json_path in json_paths:
            with open(json_path, "r") as fh:
                data = json.load(fh)

            if data.get("connector_type") != self.connector_type:
                continue

            scene_idx = int(data["scene_idx"])
            sample_id = int(data["sample_id"])
            for camera in CAMERAS:
                image_rel = data["images"][camera]
                image_path = self.dataset_dir / image_rel
                keypoints_xy = np.asarray(data[f"keypoints_{camera}"], dtype=np.float32)

                if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] != 2:
                    continue

                if np.any(keypoints_xy < 0):
                    continue

                all_records.append(
                    SampleRecord(
                        json_path=json_path,
                        image_path=image_path,
                        connector_type=self.connector_type,
                        camera=camera,
                        scene_idx=scene_idx,
                        sample_id=sample_id,
                        keypoints_xy=keypoints_xy,
                    )
                )

        unique_scenes = sorted({r.scene_idx for r in all_records})
        scene_rng = random.Random(seed)
        scene_rng.shuffle(unique_scenes)

        n_val = max(1, int(len(unique_scenes) * val_ratio)) if len(unique_scenes) > 1 else 0
        val_scenes = set(unique_scenes[:n_val])

        if self.split == "train":
            selected_scenes = set(unique_scenes[n_val:])
            if not selected_scenes:
                selected_scenes = set(unique_scenes)
        elif self.split == "val":
            selected_scenes = val_scenes if val_scenes else set(unique_scenes)
        else:
            raise ValueError(f"split desconocido: {self.split}")

        return [r for r in all_records if r.scene_idx in selected_scenes]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {record.image_path}")

        keypoints_xy = record.keypoints_xy.copy()

        if self.apply_augmentation:
            image, keypoints_xy = self.augmenter(image, keypoints_xy)
        else:
            image, keypoints_xy = resize_image_and_keypoints(
                image, keypoints_xy, self.out_height, self.out_width
            )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
        image_array = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)

        keypoints_norm = keypoints_xy.astype(np.float32).copy()
        keypoints_norm[:, 0] /= float(self.out_width)
        keypoints_norm[:, 1] /= float(self.out_height)
        target_array = keypoints_norm.reshape(-1).astype(np.float32)

        return {
            "image": image_array,
            "target": target_array,
            "keypoints_xy": keypoints_xy.astype(np.float32),
            "sample_id": record.sample_id,
            "scene_idx": record.scene_idx,
            "camera": record.camera,
            "image_path": str(record.image_path),
        }


def build_model(num_keypoints: int = 9, pretrained: bool = True):
    _, nn_mod, _, torchvision_models, weights_enum = require_torch()

    weights = None
    if pretrained:
        try:
            weights = weights_enum.DEFAULT
        except Exception:
            weights = None

    try:
        backbone = torchvision_models.mobilenet_v3_small(weights=weights)
    except Exception:
        backbone = torchvision_models.mobilenet_v3_small(weights=None)

    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn_mod.Sequential(
        nn_mod.Linear(in_features, 256),
        nn_mod.Hardswish(),
        nn_mod.Dropout(p=0.2),
        nn_mod.Linear(256, num_keypoints * 2),
    )
    return backbone


def freeze_backbone(model) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model) -> None:
    for param in model.features.parameters():
        param.requires_grad = True


def freeze_features_partial(model, n_trainable_blocks: int) -> None:
    """Congela todos los bloques de features excepto los ultimos n_trainable_blocks."""
    blocks = list(model.features)
    n_freeze = max(0, len(blocks) - n_trainable_blocks)
    for block in blocks[:n_freeze]:
        for param in block.parameters():
            param.requires_grad = False
    for block in blocks[n_freeze:]:
        for param in block.parameters():
            param.requires_grad = True


def print_trainable_summary(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(total, 1)
    print(f"Parametros entrenables: {trainable:,} / {total:,} ({pct:.1f}%)")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrenamiento del Keypoint Estimator.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Ruta a dataset_A.")
    parser.add_argument("--connector_type", choices=["sfp", "sc", "all"], required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=288)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--preview_augmentations",
        type=int,
        default=0,
        help="Si >0, guarda un mosaico de N augmentations y termina.",
    )
    parser.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="Desactiva pesos de ImageNet en MobileNetV3-Small.",
    )
    parser.add_argument(
        "--disable_augmentations",
        action="store_true",
        help="Entrena sin augmentations.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Congela el backbone (features) y entrena solo la cabeza de regresion.",
    )
    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=0,
        help="Epoca en la que se descongela el backbone (0 = nunca). Requiere --freeze_backbone.",
    )
    parser.add_argument(
        "--backbone_lr_factor",
        type=float,
        default=0.1,
        help="Factor del LR para los bloques de backbone entrenables (default: 0.1).",
    )
    parser.add_argument(
        "--finetune_last_n_blocks",
        type=int,
        default=0,
        help=(
            "Fine-tuning parcial: congela todos los bloques de features excepto los ultimos N. "
            "Usa 2 grupos de LR: backbone*backbone_lr_factor y cabeza*lr. "
            "Recomendado para ~1200 muestras: --finetune_last_n_blocks 3"
        ),
    )
    parser.add_argument(
        "--export_augmented_dataset",
        action="store_true",
        help="Exporta un dataset aumentado offline en output_dir y termina.",
    )
    parser.add_argument(
        "--num_augmented_copies",
        type=int,
        default=2,
        help="Copias aumentadas por cada sample original al exportar offline.",
    )
    parser.add_argument(
        "--skip_originals",
        action="store_true",
        help="Si se exporta offline, no copia las muestras originales al dataset final.",
    )
    return parser


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    if hasattr(image_tensor, "detach"):
        image = image_tensor.detach().cpu().numpy()
    else:
        image = np.asarray(image_tensor)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_preview(
    dataset: KeypointDataset,
    count: int,
    save_path: Path,
) -> None:
    tiles = []
    count = min(count, len(dataset))

    for idx in range(count):
        sample = dataset[idx]
        image = denormalize_image(sample["image"])
        keypoints_xy = np.asarray(sample["keypoints_xy"])

        for kp_idx, (x, y) in enumerate(keypoints_xy):
            pt = (int(round(x)), int(round(y)))
            cv2.circle(image, pt, 4, (0, 255, 0), -1)
            cv2.putText(
                image,
                str(kp_idx),
                (pt[0] + 4, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        label = f"id={sample['sample_id']} scene={sample['scene_idx']} {sample['camera']}"
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)
        tiles.append(image)

    if not tiles:
        raise RuntimeError("No hubo muestras para el preview de augmentations.")

    rows = []
    row_size = 3
    blank = np.zeros_like(tiles[0])
    for row_start in range(0, len(tiles), row_size):
        row_tiles = tiles[row_start:row_start + row_size]
        while len(row_tiles) < row_size:
            row_tiles.append(blank.copy())
        rows.append(np.concatenate(row_tiles, axis=1))

    mosaic = np.concatenate(rows, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), mosaic)
    if not ok:
        raise RuntimeError(f"No se pudo guardar el preview en {save_path}")


def export_sample_record(
    source_data: dict,
    images_bgr: dict[str, np.ndarray],
    keypoints_by_camera: dict[str, np.ndarray],
    output_dir: Path,
    sample_id: int,
    is_augmented: bool,
    augmentation_copy_idx: int | None,
    source_json_name: str,
) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sid = f"{sample_id:07d}"
    record = copy.deepcopy(source_data)
    record["sample_id"] = sample_id
    record["images"] = {}
    record["is_augmented"] = is_augmented
    record["augmentation_copy_idx"] = augmentation_copy_idx
    record["source_json"] = source_json_name
    record["source_sample_id"] = int(source_data.get("sample_id", sample_id))

    for cam in CAMERAS:
        image_name = f"{sid}_{cam}.png"
        image_path = images_dir / image_name
        ok = cv2.imwrite(str(image_path), images_bgr[cam])
        if not ok:
            raise RuntimeError(f"No se pudo guardar la imagen exportada en {image_path}")
        record["images"][cam] = str(Path("images") / image_name)
        record[f"keypoints_{cam}"] = keypoints_by_camera[cam].astype(float).tolist()

    json_path = output_dir / f"{sid}.json"
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2)


def export_augmented_dataset(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    connector_types = {"sfp", "sc"} if args.connector_type == "all" else {args.connector_type}
    json_paths = sorted(dataset_dir.glob("*.json"))
    source_jsons = []
    for json_path in json_paths:
        with open(json_path, "r") as fh:
            data = json.load(fh)
        if data.get("connector_type") in connector_types:
            source_jsons.append((json_path, data))

    if not source_jsons:
        raise RuntimeError(
            f"No se encontraron samples para connector_type={args.connector_type!r} en {dataset_dir}"
        )

    exported_originals = 0
    exported_augmented = 0
    next_sample_id = 0

    for json_path, data in source_jsons:
        base_images: dict[str, np.ndarray] = {}
        base_keypoints: dict[str, np.ndarray] = {}

        for cam in CAMERAS:
            image_path = dataset_dir / data["images"][cam]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
            base_images[cam] = image
            base_keypoints[cam] = np.asarray(data[f"keypoints_{cam}"], dtype=np.float32)

        if not args.skip_originals:
            export_sample_record(
                source_data=data,
                images_bgr=base_images,
                keypoints_by_camera=base_keypoints,
                output_dir=output_dir,
                sample_id=next_sample_id,
                is_augmented=False,
                augmentation_copy_idx=None,
                source_json_name=json_path.name,
            )
            next_sample_id += 1
            exported_originals += 1

        for aug_idx in range(args.num_augmented_copies):
            aug_images: dict[str, np.ndarray] = {}
            aug_keypoints: dict[str, np.ndarray] = {}

            for cam in CAMERAS:
                image = base_images[cam]
                keypoints = base_keypoints[cam]
                height, width = image.shape[:2]
                augmenter = KeypointAugmenter(out_height=height, out_width=width)
                aug_img, aug_kps = augment_until_valid(augmenter, image, keypoints)
                aug_images[cam] = aug_img
                aug_keypoints[cam] = aug_kps

            export_sample_record(
                source_data=data,
                images_bgr=aug_images,
                keypoints_by_camera=aug_keypoints,
                output_dir=output_dir,
                sample_id=next_sample_id,
                is_augmented=True,
                augmentation_copy_idx=aug_idx,
                source_json_name=json_path.name,
            )
            next_sample_id += 1
            exported_augmented += 1

    summary = {
        "source_dataset_dir": str(dataset_dir),
        "output_dataset_dir": str(output_dir),
        "connector_type": args.connector_type,
        "num_source_samples": len(source_jsons),
        "num_original_samples_exported": exported_originals,
        "num_augmented_samples_exported": exported_augmented,
        "num_total_samples_exported": exported_originals + exported_augmented,
        "num_total_images_exported": (exported_originals + exported_augmented) * len(CAMERAS),
        "num_augmented_copies_per_sample": args.num_augmented_copies,
        "includes_originals": not args.skip_originals,
    }
    with open(output_dir / "export_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        "Dataset aumentado exportado en "
        f"{output_dir} | samples={summary['num_total_samples_exported']} "
        f"| images={summary['num_total_images_exported']}"
    )
    return 0


def compute_pixel_mae(
    predictions,
    targets,
    image_width: int,
    image_height: int,
) -> float:
    torch_mod, _, _, _, _ = require_torch()
    scale = torch_mod.tensor(
        [image_width, image_height] * (predictions.shape[1] // 2),
        device=predictions.device,
        dtype=predictions.dtype,
    )
    error = (predictions - targets).abs() * scale
    return float(error.mean().item())


def run_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: str,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    torch_mod, _, _, _, _ = require_torch()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_mae_px = 0.0
    total_items = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch_mod.set_grad_enabled(is_train):
            predictions = model(images)
            loss = loss_fn(predictions, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_mae_px += compute_pixel_mae(
            predictions, targets, image_width=image_width, image_height=image_height
        ) * batch_size
        total_items += batch_size

    mean_loss = total_loss / max(total_items, 1)
    mean_mae_px = total_mae_px / max(total_items, 1)
    return mean_loss, mean_mae_px


def main() -> int:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.dataset_dir.expanduser()

    if args.export_augmented_dataset:
        return export_augmented_dataset(args)

    if args.connector_type == "all":
        raise ValueError("connector_type=all solo se admite con --export_augmented_dataset")

    train_dataset = KeypointDataset(
        dataset_dir=dataset_dir,
        connector_type=args.connector_type,
        split="train",
        out_height=args.image_height,
        out_width=args.image_width,
        val_ratio=args.val_ratio,
        seed=args.seed,
        apply_augmentation=not args.disable_augmentations,
    )

    if args.preview_augmentations > 0:
        preview_path = args.output_dir / f"augmentations_{args.connector_type}.png"
        render_preview(train_dataset, count=args.preview_augmentations, save_path=preview_path)
        print(f"Preview de augmentations guardado en: {preview_path}")
        return 0

    val_dataset = KeypointDataset(
        dataset_dir=dataset_dir,
        connector_type=args.connector_type,
        split="val",
        out_height=args.image_height,
        out_width=args.image_width,
        val_ratio=args.val_ratio,
        seed=args.seed,
        apply_augmentation=False,
    )

    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"Connector: {args.connector_type}"
    )

    torch_mod, nn_mod, DataLoader_cls, _, _ = require_torch()
    if args.device is None:
        args.device = "cuda" if torch_mod.cuda.is_available() else "cpu"

    train_loader = DataLoader_cls(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader_cls(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model = build_model(num_keypoints=9, pretrained=not args.disable_pretrained).to(args.device)

    backbone_lr = args.lr * args.backbone_lr_factor

    if args.finetune_last_n_blocks > 0:
        # Modo fine-tuning parcial: congela primeros bloques, entrena ultimos N + cabeza
        freeze_features_partial(model, args.finetune_last_n_blocks)
        n_total_blocks = len(list(model.features))
        n_frozen = max(0, n_total_blocks - args.finetune_last_n_blocks)
        print(
            f"Fine-tuning parcial | bloques congelados: {n_frozen}/{n_total_blocks} | "
            f"LR backbone={backbone_lr:.2e} | LR cabeza={args.lr:.2e}"
        )
        print_trainable_summary(model)
        optimizer = torch_mod.optim.AdamW(
            [
                {"params": model.features[n_frozen:].parameters(), "lr": backbone_lr},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    elif args.freeze_backbone:
        # Modo cabeza-solo: backbone completamente congelado
        freeze_backbone(model)
        print("Backbone congelado | entrenando solo la cabeza de regresion")
        print_trainable_summary(model)
        optimizer = torch_mod.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        # Modo completo: toda la red
        optimizer = torch_mod.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    loss_fn = nn_mod.SmoothL1Loss(beta=0.02)
    scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone and args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            unfreeze_backbone(model)
            optimizer = torch_mod.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": backbone_lr},
                    {"params": model.classifier.parameters(), "lr": args.lr},
                ],
                weight_decay=args.weight_decay,
            )
            scheduler = torch_mod.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1
            )
            print(f"[{epoch:03d}] Backbone descongelado | LR backbone={backbone_lr:.2e} | LR head={args.lr:.2e}")

        t0_train = time.time()
        train_loss, train_mae_px = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=args.device,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        t_train = time.time() - t0_train

        t0_val = time.time()
        val_loss, val_mae_px = run_epoch(
            model,
            val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=args.device,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        t_val = time.time() - t0_val
        scheduler.step()

        t_epoch = t_train + t_val
        elapsed = time.time() - start_time
        epochs_left = args.epochs - epoch
        eta_sec = t_epoch * epochs_left
        eta_str = f"{int(eta_sec // 3600):02d}h{int((eta_sec % 3600) // 60):02d}m{int(eta_sec % 60):02d}s"
        elapsed_str = f"{int(elapsed // 3600):02d}h{int((elapsed % 3600) // 60):02d}m{int(elapsed % 60):02d}s"

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae_px": train_mae_px,
            "val_loss": val_loss,
            "val_mae_px": val_mae_px,
            "lr": optimizer.param_groups[0]["lr"],
            "t_train_s": round(t_train, 2),
            "t_val_s": round(t_val, 2),
            "t_epoch_s": round(t_epoch, 2),
        }
        history.append(row)

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"loss={train_loss:.5f} mae={train_mae_px:.2f}px | "
            f"val_loss={val_loss:.5f} val_mae={val_mae_px:.2f}px | "
            f"train={t_train:.1f}s val={t_val:.1f}s epoca={t_epoch:.1f}s | "
            f"elapsed={elapsed_str} ETA={eta_str}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = args.output_dir / f"best_{args.connector_type}.pt"
            torch_mod.save(
                {
                    "model_state_dict": model.state_dict(),
                    "connector_type": args.connector_type,
                    "num_keypoints": 9,
                    "image_height": args.image_height,
                    "image_width": args.image_width,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "args": vars(args),
                },
                ckpt_path,
            )

    summary = {
        "connector_type": args.connector_type,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "image_height": args.image_height,
        "image_width": args.image_width,
        "best_val_loss": best_val_loss,
        "total_time_sec": time.time() - start_time,
        "history": history,
    }
    summary_path = args.output_dir / f"summary_{args.connector_type}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Resumen guardado en: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
