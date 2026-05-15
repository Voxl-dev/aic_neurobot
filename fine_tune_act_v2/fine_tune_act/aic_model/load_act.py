"""
Load the Aloha ACT checkpoint and adapt it to the AIC dimensions.

The upstream LeRobot API changed in 0.5.x, so this module uses the current
`lerobot.policies.*` imports and performs the partial weight transfer manually.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


HF_CHECKPOINT_REPO = "lerobot/act_aloha_sim_insertion_human"
N_TRANSFERABLE_STATE_DIMS = 6

_SKIP_PREFIXES: Tuple[str, ...] = (
    "model.action_head",
    "model.encoder_robot_state_input_proj",
    "model.vae_encoder_robot_state_input_proj",
    "normalize_inputs",
    "normalize_targets",
    "unnormalize_outputs",
)


def smart_partial_init_state_proj(
    aloha_weight: torch.Tensor,
    aloha_bias: torch.Tensor,
    aic_state_dim: int = 32,
    n_transferable: int = N_TRANSFERABLE_STATE_DIMS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy transferable joint-state columns and initialize the new columns."""
    dim_model, _ = aloha_weight.shape
    aic_weight = torch.empty(dim_model, aic_state_dim, dtype=aloha_weight.dtype)
    nn.init.xavier_uniform_(aic_weight)
    aic_weight[:, :n_transferable] = aloha_weight[:, :n_transferable]
    return aic_weight, aloha_bias.clone()


def smart_small_init_action_head(policy_layer: nn.Linear, scale: float = 0.01) -> None:
    """Start the Cartesian action head close to zero for stable warmup."""
    with torch.no_grad():
        nn.init.normal_(policy_layer.weight, mean=0.0, std=scale)
        if policy_layer.bias is not None:
            nn.init.zeros_(policy_layer.bias)


def _load_matching_weights(policy: ACTPolicy, source_state: dict[str, torch.Tensor]) -> list[str]:
    target_state = policy.state_dict()
    compatible = {}
    skipped = []

    for key, tensor in source_state.items():
        if key.startswith(_SKIP_PREFIXES):
            skipped.append(key)
            continue
        if key in target_state and target_state[key].shape == tensor.shape:
            compatible[key] = tensor
        else:
            skipped.append(key)

    policy.load_state_dict(compatible, strict=False)
    return skipped


def _apply_smart_init(policy: ACTPolicy, source_state: dict[str, torch.Tensor]) -> None:
    target_state = policy.state_dict()

    for prefix in (
        "model.encoder_robot_state_input_proj",
        "model.vae_encoder_robot_state_input_proj",
    ):
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        if weight_key not in source_state or bias_key not in source_state:
            continue

        aic_state_dim = target_state[weight_key].shape[1]
        weight, bias = smart_partial_init_state_proj(
            source_state[weight_key],
            source_state[bias_key],
            aic_state_dim=aic_state_dim,
        )
        target_state[weight_key].copy_(weight)
        target_state[bias_key].copy_(bias)

    smart_small_init_action_head(policy.model.action_head)


def load_pretrained_act_for_aic(
    config: ACTConfig,
    checkpoint_repo: str = HF_CHECKPOINT_REPO,
    smart_init: bool = True,
    verbose: bool = True,
) -> ACTPolicy:
    """Return an ACTPolicy initialized from Aloha where dimensions match."""
    ckpt_dir = snapshot_download(
        repo_id=checkpoint_repo,
        allow_patterns=["model.safetensors", "config.json"],
    )
    source_state = load_file(f"{ckpt_dir}/model.safetensors")

    policy = ACTPolicy(config)
    skipped = _load_matching_weights(policy, source_state)

    if smart_init:
        _apply_smart_init(policy, source_state)

    if verbose:
        print(f"[load_act] Base checkpoint: {checkpoint_repo}")
        print(f"[load_act] Source tensors: {len(source_state)}")
        print(f"[load_act] Skipped/special tensors: {len(skipped)}")
        print("[load_act] Smart init applied to state projections and action head.")

    return policy
