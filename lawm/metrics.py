from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .model import LeastActionWorldModel
from .utils import load_checkpoint, load_state_tensor, make_time_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LaWM physical consistency metrics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-pt", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--state-dim", type=int, default=9)
    parser.add_argument(
        "--true-energy-mode",
        choices=["none", "kinetic", "translational_gravity"],
        default="translational_gravity",
        help="State-space energy to report alongside model-learned energy.",
    )
    parser.add_argument("--gravity", type=float, default=9.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def stationary_action_residual(model: LeastActionWorldModel, states: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    q_seq = model.q_sequence_from_state(states)
    residual = model.dynamics.stationary_action_residual(q_seq, ts)
    if residual.numel() == 0:
        return states.new_tensor(0.0)
    return residual.norm(dim=-1).mean()


def relative_energy_drift(
    model: LeastActionWorldModel,
    states: torch.Tensor,
    ts: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    energy = model.total_energy_from_state(states, ts)
    drift = (energy[:, -1] - energy[:, 0]).abs() / (energy[:, 0].abs() + eps)
    return drift.mean()


def energy_variation(
    model: LeastActionWorldModel,
    states: torch.Tensor,
    ts: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    energy = model.total_energy_from_state(states, ts)
    variation = (energy.max(dim=1).values - energy.min(dim=1).values) / (energy.mean(dim=1).abs() + eps)
    return variation.mean()


def state_space_energy(
    states: torch.Tensor,
    mode: str,
    gravity: float = 9.8,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    if states.shape[-1] < 4:
        raise ValueError("State-space energy requires at least [x, y, vx, vy] columns")
    vx = states[..., 2]
    vy = states[..., 3]
    energy = 0.5 * (vx.square() + vy.square())
    if mode == "translational_gravity":
        energy = energy + gravity * states[..., 1]
    elif mode != "kinetic":
        raise ValueError(f"Unknown true-energy-mode: {mode}")
    return energy


def relative_drift_from_energy(energy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ((energy[:, -1] - energy[:, 0]).abs() / (energy[:, 0].abs() + eps)).mean()


def variation_from_energy(energy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ((energy.max(dim=1).values - energy.min(dim=1).values) / (energy.mean(dim=1).abs() + eps)).mean()


def pis_norm_from_series(series: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = series.mean(dim=1).abs()
    std = series.std(dim=1, unbiased=False)
    return (1.0 / (1.0 + std / (mean + eps))).mean()


def compute_metrics(
    model: LeastActionWorldModel,
    states: torch.Tensor,
    ts: torch.Tensor,
    true_energy_mode: str = "translational_gravity",
    gravity: float = 9.8,
) -> Dict[str, object]:
    with torch.enable_grad():
        residual = stationary_action_residual(model, states, ts)
        drift = relative_energy_drift(model, states, ts)
        variation = energy_variation(model, states, ts)
    metrics: Dict[str, object] = {
        "stationary_action_residual": float(residual.detach().cpu()),
        "model_relative_energy_drift": float(drift.detach().cpu()),
        "model_relative_energy_variation": float(variation.detach().cpu()),
    }
    true_energy = state_space_energy(states, true_energy_mode, gravity=gravity)
    if true_energy is not None:
        state_drift = relative_drift_from_energy(true_energy)
        state_variation = variation_from_energy(true_energy)
        state_pis = pis_norm_from_series(true_energy)
        metrics.update(
            {
                "state_energy_mode": true_energy_mode,
                "state_relative_energy_drift": float(state_drift.detach().cpu()),
                "state_relative_energy_variation": float(state_variation.detach().cpu()),
                "state_energy_pis_norm": float(state_pis.detach().cpu()),
            }
        )
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    states = load_state_tensor(args.trajectory_pt, args.state_dim, args.max_samples).to(device)
    ts = make_time_grid(states.shape[1], args.dt, device)
    metrics = compute_metrics(
        model,
        states,
        ts,
        true_energy_mode=args.true_energy_mode,
        gravity=args.gravity,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
