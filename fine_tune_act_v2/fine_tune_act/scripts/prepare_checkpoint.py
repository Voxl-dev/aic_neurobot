"""
ONE-SHOT script: descarga el checkpoint de Aloha desde HuggingFace Hub,
aplica SMART PARTIAL INIT para las dimensiones de AIC, y lo guarda
localmente para que `lerobot-train` lo use vía `policy.pretrained_path`.

Por qué este script existe:
  El CLI oficial `lerobot-train` no soporta partial-init de capas con
  dimensiones incompatibles. Si le pasamos directamente el checkpoint
  de Aloha, falla con shape mismatch. La solución es preprocesar el
  checkpoint UNA vez (este script) y entrenar contra el resultado.

Robustez:
  Usa solo APIs estables de PyTorch + safetensors + ACTPolicy.
  Si save_pretrained de LeRobot falla, cae a guardado manual.

Uso:
    pixi run python scripts/prepare_checkpoint.py
Output:
    outputs/init_smart_ckpt/{model.safetensors, config.json}
"""
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aic_model.load_act import load_pretrained_act_for_aic
from lerobot.common.policies.act.configuration_act import ACTConfig


OUTPUT_DIR = REPO_ROOT / "outputs" / "init_smart_ckpt"
CONFIG_PATH = REPO_ROOT / "configs" / "act_aic.yaml"


def build_act_config(cfg) -> ACTConfig:
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


def save_checkpoint_robust(policy, output_dir: Path):
    """save_pretrained con fallback a safetensors manual."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intento 1: API oficial save_pretrained
    try:
        policy.save_pretrained(str(output_dir))
        print(f"[prepare_checkpoint] Guardado con policy.save_pretrained()")
        return
    except Exception as e:
        print(f"[prepare_checkpoint] save_pretrained falló: {e}")
        print(f"[prepare_checkpoint] Cayendo a guardado manual...")

    # Intento 2: guardado manual
    from safetensors.torch import save_file
    save_file(policy.state_dict(), str(output_dir / "model.safetensors"))
    cfg_dict = {k: v for k, v in policy.config.__dict__.items()
                if not k.startswith("_")}

    def make_json_safe(o):
        if isinstance(o, (list, tuple)):
            return [make_json_safe(x) for x in o]
        if isinstance(o, dict):
            return {k: make_json_safe(v) for k, v in o.items()}
        return o

    with open(output_dir / "config.json", "w") as f:
        json.dump(make_json_safe(cfg_dict), f, indent=2)
    print(f"[prepare_checkpoint] Guardado manual completado")


def main():
    print("=" * 70)
    print("PREPARE CHECKPOINT — smart partial init para AIC")
    print("=" * 70)

    cfg = OmegaConf.load(CONFIG_PATH)
    act_config = build_act_config(cfg)
    print(f"[prepare_checkpoint] Config leída de {CONFIG_PATH}")
    print(f"[prepare_checkpoint] State dim AIC: {act_config.input_shapes['observation.state']}")
    print(f"[prepare_checkpoint] Action dim AIC: {act_config.output_shapes['action']}")

    policy = load_pretrained_act_for_aic(act_config, smart_init=True, verbose=True)

    save_checkpoint_robust(policy, OUTPUT_DIR)

    print()
    print("=" * 70)
    print(f"OK — checkpoint listo en: {OUTPUT_DIR}")
    print("=" * 70)
    print()
    print("Siguiente paso: ejecutar lerobot-train apuntando a este path:")
    print(f"  policy.pretrained_path={OUTPUT_DIR.absolute()}")
    print("(o simplemente: bash scripts/run_finetune.sh)")


if __name__ == "__main__":
    main()
