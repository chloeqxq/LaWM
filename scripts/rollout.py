#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lawm.utils import load_checkpoint, make_time_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a trained physical-state LaWM checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--z0", type=str, required=True, help="Comma-separated initial state")
    parser.add_argument("--z1", type=str, default=None, help="Optional comma-separated second context state")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--out-pt", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_vector(text: str, dim: int, device: torch.device, name: str) -> torch.Tensor:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != dim:
        raise ValueError(f"{name} expected {dim} values, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, dim)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    z0 = parse_vector(args.z0, model.state_dim, device, "--z0")
    z1: Optional[torch.Tensor] = None
    if args.z1 is not None:
        z1 = parse_vector(args.z1, model.state_dim, device, "--z1")
    ts = make_time_grid(args.steps, args.dt, device)
    with torch.enable_grad():
        rollout = model(z0, ts, state1=z1).detach().cpu()
    if args.out_pt is not None:
        args.out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rollout, args.out_pt)
    print(json.dumps({"shape": list(rollout.shape), "first": rollout[0, 0].tolist(), "last": rollout[0, -1].tolist()}, indent=2))


if __name__ == "__main__":
    main()
