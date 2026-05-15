#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-dataset_A/dataset_A_complete_merged/dataset_A_complete_merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/keypoint_estimator}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
VAL_RATIO="${VAL_RATIO:-0.15}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
IMAGE_WIDTH="${IMAGE_WIDTH:-288}"
SEED="${SEED:-42}"
DISABLE_PRETRAINED="${DISABLE_PRETRAINED:-0}"
DISABLE_AUGMENTATIONS="${DISABLE_AUGMENTATIONS:-0}"

EXTRA_ARGS=()
if [[ "${DISABLE_PRETRAINED}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_pretrained)
fi
if [[ "${DISABLE_AUGMENTATIONS}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_augmentations)
fi

mkdir -p "${OUTPUT_ROOT}/sfp" "${OUTPUT_ROOT}/sc"

echo "[keypoint] Dataset: ${DATASET_DIR}"
echo "[keypoint] Output:  ${OUTPUT_ROOT}"
echo "[keypoint] Training SFP model"
pixi run python scripts/train_keypoint_estimator.py \
  --dataset_dir "${DATASET_DIR}" \
  --connector_type sfp \
  --output_dir "${OUTPUT_ROOT}/sfp" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --val_ratio "${VAL_RATIO}" \
  --num_workers "${NUM_WORKERS}" \
  --image_height "${IMAGE_HEIGHT}" \
  --image_width "${IMAGE_WIDTH}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[keypoint] Training SC model"
pixi run python scripts/train_keypoint_estimator.py \
  --dataset_dir "${DATASET_DIR}" \
  --connector_type sc \
  --output_dir "${OUTPUT_ROOT}/sc" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --val_ratio "${VAL_RATIO}" \
  --num_workers "${NUM_WORKERS}" \
  --image_height "${IMAGE_HEIGHT}" \
  --image_width "${IMAGE_WIDTH}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[keypoint] Done. Check ${OUTPUT_ROOT}/sfp and ${OUTPUT_ROOT}/sc"
