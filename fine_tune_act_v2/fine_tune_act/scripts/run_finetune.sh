#!/usr/bin/env bash
# Lanza el fine-tune ACT completo en 2 pasos:
#   1. Prepara el checkpoint con smart partial init (~3 min)
#   2. Lanza lerobot-train oficial (5-6h en RTX 2000 Ada)
#
# Prerequisito: el dataset HF LeRobotDataset YA debe estar en data/dataset_lerobot/
# (correr primero: pixi run python scripts/bag_to_lerobot.py ...)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════════"
echo "  PASO 1/2 — Preparar checkpoint con smart partial init"
echo "════════════════════════════════════════════════════════════════════"
pixi run python scripts/prepare_checkpoint.py

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  PASO 2/2 — Fine-tune con lerobot-train (5-6h)"
echo "════════════════════════════════════════════════════════════════════"
echo "  Output: outputs/act_aic_run1/"
echo "  Checkpoints cada 5000 steps (~25 min)"
echo

pixi run lerobot-train \
  --config-path="$REPO_ROOT/configs" \
  --config-name=act_aic \
  policy.pretrained_path="$REPO_ROOT/outputs/init_smart_ckpt" \
  hydra.run.dir=outputs/act_aic_run1

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  FINE-TUNE COMPLETADO"
echo "════════════════════════════════════════════════════════════════════"
echo "  Checkpoint final: outputs/act_aic_run1/checkpoints/last/"
echo
echo "  Para copiarlo al modelo final:"
echo "    mkdir -p models/act_policy"
echo "    cp outputs/act_aic_run1/checkpoints/last/* models/act_policy/"
echo
