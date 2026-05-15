#!/usr/bin/env bash
# Fine-tune ACT para AIC con LeRobot 0.5.x.
#
# Variables útiles:
#   ACT_STEPS=1000          pasos de entrenamiento
#   BATCH_SIZE=4            batch si la GPU va justa
#   NUM_WORKERS=2           workers de dataloader
#   OUTPUT_DIR=...          carpeta de salida
#   DATASET_ROOT=...        dataset LeRobot local

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
cd "$PROJECT_ROOT"

INIT_CKPT="$REPO_ROOT/outputs/init_smart_ckpt"
MIGRATED_CKPT="$REPO_ROOT/outputs/init_smart_ckpt_migrated"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/data/dataset_lerobot_step4}"
DATASET_REPO_ID="${DATASET_REPO_ID:-aic_team/sfp_sc_insertion_step4}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/act_aic_run1}"
ACT_STEPS="${ACT_STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
DEVICE="${DEVICE:-cuda}"

echo "===================================================================="
echo "  PASO 1/3 — Preparar checkpoint ACT con smart partial init"
echo "===================================================================="
pixi run python "$REPO_ROOT/scripts/prepare_checkpoint.py"

echo
echo "===================================================================="
echo "  PASO 2/3 — Migrar checkpoint al formato processor de LeRobot 0.5.x"
echo "===================================================================="
if [[ -f "$MIGRATED_CKPT/policy_preprocessor.json" && -f "$MIGRATED_CKPT/policy_postprocessor.json" ]]; then
  echo "  Checkpoint migrado ya existe: $MIGRATED_CKPT"
else
  MIGRATE_SCRIPT="$(pixi run python -c 'import pathlib, lerobot.processor.migrate_policy_normalization as m; print(pathlib.Path(m.__file__))')"
  pixi run python "$MIGRATE_SCRIPT" --pretrained-path "$INIT_CKPT"
fi

echo
echo "===================================================================="
echo "  PASO 3/3 — Entrenar ACT"
echo "===================================================================="
echo "  Dataset : $DATASET_ROOT"
echo "  Output  : $OUTPUT_DIR"
echo "  Steps   : $ACT_STEPS"
echo "  Batch   : $BATCH_SIZE"
echo "  Device  : $DEVICE"
echo

pixi run lerobot-train \
  --policy.path="$MIGRATED_CKPT" \
  --policy.device="$DEVICE" \
  --policy.push_to_hub=false \
  --policy.use_amp=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_aic_v1 \
  --steps="$ACT_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --eval_freq=0 \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --wandb.enable=false

echo
echo "===================================================================="
echo "  FINE-TUNE COMPLETADO"
echo "===================================================================="
echo "  Checkpoints en: $OUTPUT_DIR/checkpoints/"
