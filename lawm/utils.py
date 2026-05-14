from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import Optimizer

from .model import LeastActionWorldModel


def make_time_grid(length: int, dt: float, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.arange(length, dtype=torch.float32, device=device) * float(dt)


def load_state_tensor(path: Path, state_dim: int, max_samples: Optional[int] = None) -> torch.Tensor:
    states = torch.load(path, map_location="cpu")
    if not isinstance(states, torch.Tensor):
        raise TypeError(f"{path} must contain a torch.Tensor")
    states = states.float()
    if states.ndim != 3:
        raise ValueError(f"{path} must contain a tensor with shape (B,T,D), got {tuple(states.shape)}")
    states = states[:, :, :state_dim]
    if max_samples is not None:
        states = states[:max_samples]
    return states


def make_state_weights(state_dim: int, device: torch.device | str) -> torch.Tensor:
    weights = torch.ones(state_dim, device=device)
    if state_dim >= 9:
        weights = torch.tensor(
            [1.01, 1.01, 1.01, 1.01, 0.1, 0.1, 0.1, 0.1, 0.1],
            dtype=torch.float32,
            device=device,
        )
    return weights


def weighted_state_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target).square() * weights.view(1, 1, -1)).mean()


def model_config(model: LeastActionWorldModel) -> Dict[str, object]:
    return {
        "state_dim": model.state_dim,
        "latent_dim": model.latent_dim,
        "context_dim": model.context_dim,
        "hidden_dim": model.hidden_dim,
        "depth": model.depth,
        "solver_iters": model.solver_iters,
        "solver_step_size": model.solver_step_size,
    }


def save_checkpoint(
    path: Path,
    model: LeastActionWorldModel,
    optimizer: Optional[Optimizer] = None,
    *,
    args: Optional[object] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": model_config(model),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if args is not None:
        payload["args"] = vars(args) if hasattr(args, "__dict__") else args
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device | str) -> LeastActionWorldModel:
    ckpt = torch.load(path, map_location=device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = LeastActionWorldModel(**config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
