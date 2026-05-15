"""
Training loop ACT ultra-rápido: backbone congelado + imágenes reducidas.

Estrategia de velocidad en 2 ejes:
  1. Backbone congelado (requires_grad=False):
       - Backward NO pasa por los 3 ResNet18 → ~50% menos backward
       - PyTorch no almacena activaciones del backbone → menos VRAM
       - Solo se entrena: transformer + state_proj + action_head

  2. Imágenes reducidas a 240×320 en GPU (bilinear, sin cambio de pesos):
       - ResNet18 ve 4× menos píxeles → ~4× forward más rápido
       - Los positional embeddings en ACT son sinusoidales (no aprendidos)
         → adaptan dinámicamente a cualquier resolución → no rompe el modelo
       - NOTA: el policy entrenado así espera 240×320 en inferencia también.
         En RunACT agregar: img = F.interpolate(img, (240, 320), mode="bilinear")

Estimación RTX 2000 Ada 8GB / 31GB RAM:
  ┌─────────────────────────────────────────────┬────────┐
  │ Modo                                        │ Tiempo │
  ├─────────────────────────────────────────────┼────────┤
  │ lerobot-train full (60k, 480×640)           │ 5-6h   │
  │ train_act_fast.py (25k, 480×640, frozen)    │ 1.5-2h │
  │ train_act_fast.py (10k, 240×320, frozen) ★  │ 10-15m │
  └─────────────────────────────────────────────┴────────┘
  ★ = modo por defecto

Cuándo usar lerobot-train en vez de este script:
  - Cuando se tiene tiempo y se quiere máxima calidad (backbone fine-tuned)
  - Para reproducibilidad con el pipeline estándar de lerobot

Uso:
    pixi run python scripts/train_act_fast.py
    pixi run python scripts/train_act_fast.py --steps 15000
    pixi run python scripts/train_act_fast.py --img_height 480 --img_width 640  # resolución original
    pixi run python scripts/train_act_fast.py --ckpt outputs/init_smart_ckpt   # desde ckpt existente
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

from aic_model.load_act import freeze_vision_backbone, load_pretrained_act_for_aic

IMAGE_KEYS = [
    "observation.images.center",
    "observation.images.left",
    "observation.images.right",
]


def _build_act_config(cfg) -> ACTConfig:
    p = cfg.policy
    return ACTConfig(
        chunk_size=p.chunk_size,
        n_action_steps=p.n_action_steps,
        input_shapes={k: list(v) for k, v in p.input_shapes.items()},
        output_shapes={k: list(v) for k, v in p.output_shapes.items()},
        input_normalization_modes=dict(p.input_normalization_modes),
        output_normalization_modes=dict(p.output_normalization_modes),
        vision_backbone=p.vision_backbone,
        pretrained_backbone_weights=p.pretrained_backbone_weights,
        replace_final_stride_with_dilation=p.replace_final_stride_with_dilation,
        pre_norm=p.pre_norm,
        dim_model=p.dim_model,
        n_heads=p.n_heads,
        dim_feedforward=p.dim_feedforward,
        feedforward_activation=p.feedforward_activation,
        n_encoder_layers=p.n_encoder_layers,
        n_decoder_layers=p.n_decoder_layers,
        use_vae=p.use_vae,
        latent_dim=p.latent_dim,
        n_vae_encoder_layers=p.n_vae_encoder_layers,
        kl_weight=p.kl_weight,
    )


def _cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _resize_images(batch: dict, height: int, width: int) -> dict:
    """Resize todas las imágenes del batch en GPU (bilinear, sin cambio de pesos)."""
    for key in IMAGE_KEYS:
        if key in batch and batch[key].shape[-2:] != (height, width):
            batch[key] = F.interpolate(
                batch[key],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
    return batch


def _save_checkpoint(policy: ACTPolicy, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        policy.save_pretrained(str(path))
    except Exception:
        from safetensors.torch import save_file
        save_file(policy.state_dict(), str(path / "model.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT ultra-fast training")
    parser.add_argument("--steps",      type=int,   default=10000,
                        help="Pasos de entrenamiento (default 10k ≈ 10-15 min)")
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--warmup",     type=int,   default=300,
                        help="Steps de warmup lineal antes del cosine decay")
    parser.add_argument("--img_height", type=int,   default=240,
                        help="Altura de imagen en entrenamiento (default 240 → 4× más rápido)")
    parser.add_argument("--img_width",  type=int,   default=320,
                        help="Anchura de imagen (default 320)")
    parser.add_argument("--save_freq",  type=int,   default=2000)
    parser.add_argument("--log_freq",   type=int,   default=100)
    parser.add_argument("--output_dir", type=str,   default="outputs/act_fast_run1")
    parser.add_argument("--ckpt",       type=str,   default=None,
                        help="Path a init_smart_ckpt. Si None, descarga desde HF Hub.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = REPO_ROOT / args.output_dir
    cfg_path   = REPO_ROOT / "configs" / "act_aic.yaml"
    resize_hw  = (args.img_height, args.img_width)
    using_resize = resize_hw != (480, 640)

    # cuDNN autotuner: detecta la implementación más rápida para cada shape
    # (amortiza en ~20 pasos, luego ~10-15% speedup sostenido)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = OmegaConf.load(cfg_path)
    act_config = _build_act_config(cfg)

    print("=" * 65)
    print("TRAIN ACT FAST — backbone congelado + imágenes reducidas")
    print("=" * 65)
    print(f"  device      : {device}")
    print(f"  steps       : {args.steps}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  lr          : {args.lr:.1e}  (cosine decay + {args.warmup} warmup steps)")
    print(f"  img_size    : {args.img_height}×{args.img_width}"
          + (" [REDUCIDO — 4× más rápido]" if using_resize else " [resolución original]"))
    print(f"  output_dir  : {output_dir}")

    if using_resize:
        print(f"\n  AVISO: política entrenada con {args.img_height}×{args.img_width}.")
        print(f"  En inferencia (RunACT), redimensionar a este tamaño antes de forward.")

    # ── Cargar o inicializar policy ─────────────────────────────────────────
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if ckpt_path and ckpt_path.exists():
        print(f"\nCargando checkpoint: {ckpt_path}")
        policy = ACTPolicy.from_pretrained(str(ckpt_path))
    else:
        print("\nDescargando Aloha + smart partial init desde HF Hub...")
        policy = load_pretrained_act_for_aic(act_config, smart_init=True, verbose=True)

    n_frozen = freeze_vision_backbone(policy)
    policy = policy.to(device)
    policy.train()

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in policy.parameters())
    print(f"\n  Backbone congelado : {n_frozen} tensores (no backward, no almacena activaciones)")
    print(f"  Params entrenables : {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=str(REPO_ROOT / cfg.dataset.root),
    )
    if hasattr(policy, "normalize_inputs") and hasattr(dataset, "stats"):
        policy.normalize_inputs.stats = dataset.stats
    if hasattr(policy, "unnormalize_outputs") and hasattr(dataset, "stats"):
        policy.unnormalize_outputs.stats = dataset.stats

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.training.num_workers > 0),
        drop_last=True,
    )
    steps_per_epoch = len(dataloader)
    print(f"  Dataset            : {len(dataset)} frames, {dataset.num_episodes} episodios")
    print(f"  Steps/epoch        : {steps_per_epoch}  ({args.steps / steps_per_epoch:.1f} épocas totales)")

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_with_warmup(s, args.warmup, args.steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nIniciando entrenamiento ({args.steps} steps)...\n")
    data_iter    = iter(dataloader)
    t0           = time.time()
    running_loss = 0.0

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Resize en GPU (bilinear, despreciable en tiempo)
        if using_resize:
            batch = _resize_images(batch, args.img_height, args.img_width)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss, _ = policy.forward(batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

        if step % args.log_freq == 0:
            elapsed = time.time() - t0
            eta     = elapsed / step * (args.steps - step)
            avg     = running_loss / args.log_freq
            running_loss = 0.0
            print(
                f"  step {step:6d}/{args.steps}"
                f"  loss={avg:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  {elapsed/60:.1f}min / ETA {eta/60:.1f}min"
            )

        if step % args.save_freq == 0 or step == args.steps:
            ckpt_out = output_dir / "checkpoints" / f"step_{step:06d}"
            _save_checkpoint(policy, ckpt_out)
            print(f"  [ckpt] {ckpt_out.relative_to(REPO_ROOT)}")

    elapsed_total = time.time() - t0
    print(f"\nEntrenamiento completado en {elapsed_total / 60:.1f} min")
    print(f"Checkpoint final: {(output_dir / 'checkpoints' / f'step_{args.steps:06d}').relative_to(REPO_ROOT)}/")

    if using_resize:
        print(f"\nRECORDAR en RunACT / inferencia:")
        print(f"  img = F.interpolate(img.unsqueeze(0), ({args.img_height}, {args.img_width}), mode='bilinear').squeeze(0)")


if __name__ == "__main__":
    main()
