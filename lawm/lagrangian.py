from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(max(depth - 1, 0)):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(activation())
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class LatentDiscreteLagrangian(nn.Module):
    """Learned midpoint discrete Lagrangian.

    The continuous latent Lagrangian is parameterized as L(q, qdot) = T - V
    with a positive diagonal mass. The discrete Lagrangian applies the midpoint
    rule over one time interval.
    """

    def __init__(
        self,
        q_dim: int,
        context_dim: int = 0,
        hidden_dim: int = 128,
        depth: int = 3,
        min_mass: float = 1e-3,
    ) -> None:
        super().__init__()
        self.q_dim = int(q_dim)
        self.context_dim = int(context_dim)
        self.min_mass = float(min_mass)
        in_dim = self.q_dim + self.context_dim
        self.mass_net = mlp(in_dim, hidden_dim, self.q_dim, depth)
        self.potential_net = mlp(in_dim, hidden_dim, 1, depth)

        # Start near unit mass and zero potential for stable early rollouts.
        nn.init.zeros_(self.mass_net[-1].weight)
        nn.init.zeros_(self.mass_net[-1].bias)
        nn.init.zeros_(self.potential_net[-1].weight)
        nn.init.zeros_(self.potential_net[-1].bias)

    def _features(self, q: torch.Tensor, eta: Optional[torch.Tensor]) -> torch.Tensor:
        if self.context_dim <= 0:
            return q
        if eta is None:
            eta = q.new_zeros(q.shape[0], self.context_dim)
        if eta.shape[0] == 1 and q.shape[0] != 1:
            eta = eta.expand(q.shape[0], -1)
        return torch.cat([q, eta], dim=-1)

    def mass_diag(self, q: torch.Tensor, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        raw = self.mass_net(self._features(q, eta))
        return F.softplus(raw) + self.min_mass

    def potential(self, q: torch.Tensor, eta: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.potential_net(self._features(q, eta)).squeeze(-1)

    def continuous_lagrangian(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        eta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mass = self.mass_diag(q, eta)
        kinetic = 0.5 * (mass * v.square()).sum(dim=-1)
        return kinetic - self.potential(q, eta)

    def discrete_lagrangian(
        self,
        q0: torch.Tensor,
        q1: torch.Tensor,
        h: torch.Tensor | float,
        eta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(h):
            h = q0.new_tensor(float(h))
        h = h.to(device=q0.device, dtype=q0.dtype)
        midpoint = 0.5 * (q0 + q1)
        velocity = (q1 - q0) / h.clamp_min(1e-8)
        return h * self.continuous_lagrangian(midpoint, velocity, eta)

    def mass_conditioning_loss(
        self,
        q: torch.Tensor,
        eta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_q = q.reshape(-1, q.shape[-1])
        flat_eta = None if eta is None else eta.reshape(-1, eta.shape[-1])
        mass = self.mass_diag(flat_q, flat_eta)
        return mass.log().square().mean()
