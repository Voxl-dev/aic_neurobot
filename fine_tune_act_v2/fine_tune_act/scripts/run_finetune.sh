#!/usr/bin/env bash
# Lanza el fine-tune ACT completo.
#
# Modos:
#   bash scripts/run_finetune.sh          # full (5-6h, backbone entrenable)
#   bash scripts/run_finetune.sh --fast   # fast (1.5-2h, backbone congelado)
#
# Prerequisito: el dataset HF LeRobotDataset YA debe estar en data/dataset_lerobot/
# (correr primero: pixi run python scripts/bag_to_lerobot.py ...)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAST_MODE=false
for arg in "$@"; do
  [[ "$arg" == "--fast" ]] && FAST_MODE=true
done

# ── PASO 1: Preparar checkpoint con smart partial init ──────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "  PASO 1/2 — Preparar checkpoint con smart partial init (~3 min)"
echo "════════════════════════════════════════════════════════════════════"
pixi run python scripts/prepare_checkpoint.py

echo ""

# ── PASO 2: Fine-tune ───────────────────────────────────────────────────────
if [[ "$FAST_MODE" == "true" ]]; then
  echo "════════════════════════════════════════════════════════════════════"
  echo "  PASO 2/2 — FAST MODE: backbone congelado + imágenes 240×320"
  echo "  10k steps · batch=16 · lr=5e-5 · img=240×320 · ~10-15 min"
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Output: outputs/act_fast_run1/"
  echo "  NOTA: política resultante espera imágenes 240×320 en inferencia."
  echo ""
  pixi run python scripts/train_act_fast.py \
    --steps      10000 \
    --batch_size 16    \
    --lr         5e-5  \
    --warmup     300   \
    --img_height 240   \
    --img_width  320   \
    --save_freq  2000  \
    --output_dir outputs/act_fast_run1 \
    --ckpt       "$REPO_ROOT/outputs/init_smart_ckpt"
else
  echo "════════════════════════════════════════════════════════════════════"
  echo "  PASO 2/2 — FULL MODE: lerobot-train oficial (5-6h)"
  echo "  60k steps · batch=8 · lr=1e-5 · backbone entrenable"
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Output: outputs/act_aic_run1/"
  echo "  Checkpoints cada 5000 steps (~25 min)"
  echo ""
  pixi run lerobot-train \
    --config-path="$REPO_ROOT/configs" \
    --config-name=act_aic \
    policy.pretrained_path="$REPO_ROOT/outputs/init_smart_ckpt" \
    hydra.run.dir=outputs/act_aic_run1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  FINE-TUNE COMPLETADO"
echo "════════════════════════════════════════════════════════════════════"
if [[ "$FAST_MODE" == "true" ]]; then
  echo "  Checkpoint final: outputs/act_fast_run1/checkpoints/step_010000/"
else
  echo "  Checkpoint final: outputs/act_aic_run1/checkpoints/last/"
fi
echo ""
echo "  Para copiarlo al modelo final:"
echo "    mkdir -p models/act_policy"
if [[ "$FAST_MODE" == "true" ]]; then
  echo "    cp outputs/act_fast_run1/checkpoints/step_010000/* models/act_policy/"
else
  echo "    cp outputs/act_aic_run1/checkpoints/last/* models/act_policy/"
fi
echo ""
