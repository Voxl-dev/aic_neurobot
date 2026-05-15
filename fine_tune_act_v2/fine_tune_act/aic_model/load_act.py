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
from typing import Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig


# Capas que requieren tratamiento especial (no se pueden copiar directo).
SPECIAL_LAYERS: Tuple[str, ...] = (
    "action_head",                       # smart small-init
    "encoder_robot_state_input_proj",    # partial copy de primeras 6 cols
    "unnormalize_outputs",               # bounds — los calcula el dataset
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


def load_pretrained_act_for_aic(
    config: ACTConfig,
    checkpoint_repo: str = HF_CHECKPOINT_REPO,
    smart_init: bool = True,
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
        verbose: imprimir info de capas cargadas.

    Returns:
        ACTPolicy con pesos preentrenados (partial en las 2 capas raras).
    """
    ckpt_dir = snapshot_download(repo_id=checkpoint_repo)
    state_dict = load_file(f"{ckpt_dir}/model.safetensors")

    compatible = {
        k: v for k, v in state_dict.items()
        if not k.startswith(SPECIAL_LAYERS)
    }
    special = {
    