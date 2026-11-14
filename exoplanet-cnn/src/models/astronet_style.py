"""Two-branch CNN inspired by Astronet (Shallue & Vanderburg 2018).

    global view (N, 1, 2001)                     local view (N, 1, 201)
            |                                            |
    5 x [Conv-ReLU, Conv-ReLU, MaxPool(5, s2)]   2 x [Conv-ReLU, Conv-ReLU, MaxPool(7, s2)]
        channels 16, 32, 64, 128, 256                channels 16, 32
            |                                            |
         Flatten                                      Flatten
            +------------------- concat -----------------+
                                  |
                 4 x [Linear(512) -> ReLU -> Dropout]
                                  |
                              Linear(1)
                                  |
              raw logit (N,); probability = sigmoid(logit)

Usage:
    python -m src.models.astronet_style      # print architecture and parameter count
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
from torch import nn

from src.models.baseline_cnn import count_parameters, parameter_table


def conv_branch(
    channels: Sequence[int],
    convs_per_block: int,
    kernel_size: int,
    pool_size: int,
    pool_stride: int,
    batch_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 1
    for out_ch in channels:
        for _ in range(convs_per_block):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        layers.append(nn.MaxPool1d(pool_size, pool_stride))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def flat_size(branch: nn.Module, length: int) -> int:
    was_training = branch.training
    branch.eval()
    with torch.no_grad():
        n = branch(torch.zeros(1, 1, length)).shape[1]
    branch.train(was_training)
    return n


class AstronetStyle(nn.Module):
    """`forward(global, local)` returns raw logits (for BCEWithLogitsLoss)."""

    def __init__(
        self,
        global_length: int = 2001,
        local_length: int = 201,
        global_channels: Sequence[int] = (16, 32, 64, 128, 256),
        local_channels: Sequence[int] = (16, 32),
        kernel_size: int = 5,
        global_pool: tuple[int, int] = (5, 2),
        local_pool: tuple[int, int] = (7, 2),
        convs_per_block: int = 2,
        hidden: Sequence[int] = (512, 512, 512, 512),
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")

        self.global_branch = conv_branch(
            global_channels, convs_per_block, kernel_size, *global_pool, batch_norm
        )
        self.local_branch = conv_branch(
            local_channels, convs_per_block, kernel_size, *local_pool, batch_norm
        )
        try:
            self.global_dim = flat_size(self.global_branch, global_length)
            self.local_dim = flat_size(self.local_branch, local_length)
        except RuntimeError as exc:
            raise ValueError("Input length too short for the configured pooling") from exc

        layers: list[nn.Module] = []
        in_dim = self.global_dim + self.local_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        if x_global.dim() == 2:
            x_global = x_global.unsqueeze(1)
        if x_local.dim() == 2:
            x_local = x_local.unsqueeze(1)
        features = torch.cat([self.global_branch(x_global), self.local_branch(x_local)], dim=1)
        return self.classifier(features).squeeze(-1)

    def logits(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        return self(x_global, x_local)

    @torch.no_grad()
    def predict_proba(self, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x_global, x_local))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Show AstronetStyle architecture and parameter count.")
    p.add_argument("--global-length", type=int, default=2001)
    p.add_argument("--local-length", type=int, default=201)
    p.add_argument("--batch-norm", action="store_true")
    args = p.parse_args(argv)

    model = AstronetStyle(args.global_length, args.local_length, batch_norm=args.batch_norm)
    print(model)
    print()
    print(parameter_table(model))
    branch_params = {
        "global_branch": count_parameters(model.global_branch),
        "local_branch": count_parameters(model.local_branch),
        "classifier": count_parameters(model.classifier),
    }
    print()
    for name, n in branch_params.items():
        print(f"{name:<14} {n:>11,}")
    print(f"\nFlattened features: global {model.global_dim} + local {model.local_dim}"
          f" = {model.global_dim + model.local_dim}")
    out = model.eval().predict_proba(
        torch.randn(4, 1, args.global_length), torch.randn(4, 1, args.local_length)
    )
    print(f"Output shape for batch of 4: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
