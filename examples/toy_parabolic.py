#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a toy parabolic state dataset for LaWM")
    parser.add_argument("--out", type=Path, default=Path("data/toy_parabolic.pt"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_toy_parabolic(samples: int, steps: int, dt: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(steps, dtype=torch.float32) * dt
    x0 = torch.empty(samples, 1).uniform_(-1.0, 1.0, generator=generator)
    y0 = torch.empty(samples, 1).uniform_(2.0, 4.0, generator=generator)
    vx0 = torch.empty(samples, 1).uniform_(0.5, 2.0, generator=generator)
    vy0 = torch.empty(samples, 1).uniform_(1.0, 4.0, generator=generator)
    gravity = 9.8

    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * gravity * t.square()
    vx = vx0.expand_as(x)
    vy = vy0 - gravity * t

    zeros = torch.zeros_like(x)
    size = torch.ones_like(x)
    area = torch.ones_like(x)
    return torch.stack([x, y, vx, vy, zeros, zeros, size, size, area], dim=-1)


def main() -> None:
    args = parse_args()
    states = make_toy_parabolic(args.samples, args.steps, args.dt, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, args.out)
    print(f"Saved {tuple(states.shape)} tensor to {args.out}")


if __name__ == "__main__":
    main()
