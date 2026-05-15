"""
Carga el checkpoint HuggingFace `lerobot/act_aloha_sim_insertion_scripted`
y lo adapta a las dimensiones de AIC con SMART PARTIAL INIT.

Para las 2 capas con dimensiones incompatibles:

  * encoder_robot_state_input_proj (Aloha [512,14] → AIC [512,32]):
      Copia las primeras 6 columnas de Aloha (codifican joint positions del
      primer brazo, semánticamente similares a los 6 joints del UR5e).
      Random init para las 26 columnas restantes (gripper + 25 keypoints).
      El bias [512] SÍ se copia íntegro (misma dim).

  * action_head (Aloha [14,512] → AIC [6,512]):
      Inicialización con magnitud pequeña (escala 0.01) en vez de Xavier
      estándar — reduce explosión inicial del loss durante warmup.
      La semántica es distinta (delta-joint vs Twist Cartesiano), por lo
      que copiar pesos directamente no aporta.

Resultado: NINGUNA capa se entrena 100% desde cero. El backbone visual,
todos los transformer layers, Y la mayor parte del state projection
arrancan desde pesos preentrenados.

Uso:
    from aic_model.load_act import load_pretrained_act_for_aic
    policy = load_pretrained_act_for_aic(config)

Smoke test (verifica descarga + smart init):
    python aic_model/load_act.py
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig


# Capas que requieren tratamiento especial (no se pueden copiar directo).
# Se filtra por substring para ser robusto frente a prefijos como "model.".
SPECIAL_LAYERS: Tuple[str, ...] = (
    "action_head",
    "encoder_robot_state_input_proj",
    "vae_encoder_robot_state_input_proj",
    "unnormalize_outputs",
    "normalize_inputs.observation.state",
    "normalize_targets",
)

HF_CHECKPOINT_REPO = "lerobot/act_aloha_sim_insertion_scripted"

# Cuántas dimensiones del state vector son joint positions (transferibles).
# AIC: [j1..j6, gripper, kp1u, kp1v, ...] → primeras 6 = joints UR5e.
# Aloha: [arm1_j1..arm1_j7, arm2_j1..arm2_j7] → primeras 6 = joints arm1.
N_TRANSFERABLE_STATE_DIMS = 6


def smart_partial_init_state_proj(
    aloha_weight: torch.Tensor,           # [512, 14]
    aloha_bias: torch.Tensor,             # [512]
    aic_state_dim: int = 32,
    n_transferable: int = N_TRANSFERABLE_STATE_DIMS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inicializa la matriz de proyección state→features de AIC ([512, 32])
    copiando las primeras `n_transferable` columnas desde Aloha, y usando
    Xavier random para las restantes.
    """
    dim_model, _ = aloha_weight.shape
    aic_weight = torch.empty(dim_model, aic_state_dim, dtype=aloha_weight.dtype)
    nn.init.xavier_uniform_(aic_weight)
    aic_weight[:, :n_transferable] = aloha_weight[:, :n_transferable]
    aic_bias = aloha_bias.clone()
    return aic_weight, aic_bias


def smart_small_init_action_head(
    policy_layer: nn.Linear,
    scale: float = 0.01,
) -> None:
    """
    Inicializa action_head con magnitud pequeña.
    Predicciones iniciales ~0 → evita explosión de loss en los primeros steps.
    """
    with torch.no_grad():
        nn.init.normal_(policy_layer.weight, mean=0.0, std=scale)
        if policy_layer.bias is not None:
            nn.init.zeros_(policy_layer.bias)


def _find_submodule(root: nn.Module, name: str) -> Optional[nn.Module]:
    """Busca un módulo por nombre exacto del segmento final en la jerarquía."""
    for mod_name, mod in root.named_modules():
        segment = mod_name.split(".")[-1] if mod_name else ""
        if segment == name:
            return mod
    return None


def freeze_vision_backbone(policy: nn.Module) -> int:
    """
    Congela los parámetros de los ResNet18 backbone con requires_grad=False.

    IMPORTANTE: este efecto NO se persiste en safetensors. Si se usa lerobot-train,
    el congelamiento se pierde al recargar el checkpoint. Para lerobot-train usa
    lr_backbone=0.0 en el yaml. Para el training loop custom (train_act_fast.py)
    llama a esta función después de cargar el modelo.

    Retorna el número de tensores congelados.
    """
    frozen = 0
    for name, param in policy.named_parameters():
        if "backbone" in name:
            param.requires_grad_(False)
            frozen += 1
    return frozen


def load_pretrained_act_for_aic(
    config: ACTConfig,
    checkpoint_repo: str = HF_CHECKPOINT_REPO,
    smart_init: bool = True,
    freeze_backbone: bool = False,
    verbose: bool = True,
) -> ACTPolicy:
    """
    Descarga checkpoint desde HuggingFace Hub y carga pesos en una ACTPolicy
    de AIC, con smart partial init para las capas incompatibles.

    Args:
        config: ACTConfig con shapes de AIC.
        checkpoint_repo: ID del repo en HF Hub.
        smart_init: si True, hace partial transfer. Si False, init random
                    (fallback de seguridad si smart_init crashea por API).
        freeze_backbone: si True, llama a freeze_vision_backbone() al final.
                         Solo tiene efecto en uso directo — no persiste en
                         safetensors (ver docstring de freeze_vision_backbone).
        verbose: imprimir info de capas cargadas.

    Returns:
        ACTPolicy con pesos preentrenados (partial en las 2 capas raras).
    """
    ckpt_dir = snapshot_download(repo_id=checkpoint_repo)
    state_dict = load_file(f"{ckpt_dir}/model.safetensors")

    # Separar compatibles de especiales por substring (robusto ante prefijos).
    def _is_special(key: str) -> bool:
        return any(name in key for name in SPECIAL_LAYERS)

    compatible = {k: v for k, v in state_dict.items() if not _is_special(k)}
    n_special = sum(1 for k in state_dict if _is_special(k))

    # Instanciar ACTPolicy con shapes de AIC (pesos random por ahora).
    policy = ACTPolicy(config)

    # Cargar pesos compatibles. strict=False ignora missing/unexpected keys
    # (las capas de normalización faltan porque las fija el dataset en training).
    missing_keys, _ = policy.load_state_dict(compatible, strict=False)

    if verbose:
        print(f"  Capas compatibles cargadas : {len(compatible)}")
        print(f"  Capas especiales (skip)    : {n_special}")
        print(f"  Missing (normalization)    : {len(missing_keys)}")

    if not smart_init:
        if verbose:
            print("  smart_init=False — Xavier random para capas especiales")
        return policy

    # ── Smart partial init para TODAS las state projection layers ─────────
    # Incluye encoder_robot_state_input_proj y vae_encoder_robot_state_input_proj
    state_proj_names = [
        "encoder_robot_state_input_proj",
        "vae_encoder_robot_state_input_proj",
    ]
    for proj_name in state_proj_names:
        sp_w = next(
            (v for k, v in state_dict.items() if proj_name in k and k.endswith(".weight")),
            None,
        )
        sp_b = next(
            (v for k, v in state_dict.items() if proj_name in k and k.endswith(".bias")),
            None,
        )
        if sp_w is None or sp_b is None:
            if verbose:
                print(f"  SKIP {proj_name} (no hallado en Aloha — Xavier init)")
            continue

        aic_dim = config.input_shapes["observation.state"][0]
        new_w, new_b = smart_partial_init_state_proj(sp_w, sp_b, aic_state_dim=aic_dim)

        proj = _find_submodule(policy, proj_name)
        if proj is None:
            if verbose:
                print(f"  WARN {proj_name} no existe en ACTPolicy con esta config")
            continue

        with torch.no_grad():
            proj.weight.copy_(new_w)
            proj.bias.copy_(new_b)
        if verbose:
            print(f"  Smart partial init: {proj_name} {list(new_w.shape)}")

    # ── Smart small init: action_head ──────────────────────────────────────
    action_head = _find_submodule(policy, "action_head")
    if action_head is not None and isinstance(action_head, nn.Linear):
        smart_small_init_action_head(action_head)
        if verbose:
            print(f"  Small init: action_head {list(action_head.weight.shape)}")
    else:
        if verbose:
            print("  WARN: action_head no encontrado — init por defecto")

    if freeze_backbone:
        n_frozen = freeze_vision_backbone(policy)
        if verbose:
            print(f"  Backbone congelado: {n_frozen} tensores (solo en memoria)")

    return policy


if __name__ == "__main__":
    print("Smoke test: load_pretrained_act_for_aic ...")
    _config = ACTConfig(
        input_shapes={
            "observation.state": [32],
            "observation.images.center": [3, 480, 640],
            "observation.images.left": [3, 480, 640],
            "observation.images.right": [3, 480, 640],
        },
        output_shapes={"action": [6]},
        input_normalization_modes={
            "observation.state": "min_max",
            "observation.images.center": "mean_std",
            "observation.images.left": "mean_std",
            "observation.images.right": "mean_std",
        },
        output_normalization_modes={"action": "min_max"},
    )
    _policy = load_pretrained_act_for_aic(_config, verbose=True)
    _ah = _find_submodule(_policy, "action_head")
    _sp = _find_submodule(_policy, "encoder_robot_state_input_proj")
    assert _ah is not None, "action_head no encontrado en ACTPolicy"
    assert _sp is not None, "encoder_robot_state_input_proj no encontrado en ACTPolicy"
    assert _sp.weight.shape[1] == 32, f"state_proj esperado [512,32], got {_sp.weight.shape}"
    assert _ah.weight.shape[0] == 6, f"action_head esperado [6,512], got {_ah.weight.shape}"
    print(f"  action_head : {list(_ah.weight.shape)}")
    print(f"  state_proj  : {list(_sp.weight.shape)}")
    print("Smoke test OK")
