"""Baseline 1D CNN for exoplanet transit classification on the global view.

    input (N, 1, 2001)
    4 x [Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d]
    Flatten -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
    output (N,) raw logit; planet probability = sigmoid(logit) via predict_proba

Usage:
    python -m src.models.baseline_cnn          # print architecture and parameter count
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
from torch import nn


def conv_block(
    in_channels: int, out_channels: int, kernel_size: int, pool_size: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool_size),
    )


class BaselineCNN(nn.Module):
    """Global-view CNN. `forward` returns raw logits (for BCEWithLogitsLoss)."""

    def __init__(
        self,
        input_length: int = 2001,
        channels: Sequence[int] = (16, 32, 64, 128),
        kernel_size: int = 5,
        pool_size: int = 4,
        hidden: Sequence[int] = (256, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length")

        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks.append(conv_block(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, input_length)).numel()
        if flat_dim == 0:
            raise ValueError(f"input_length {input_length} is too short for {len(channels)} pools")

        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = flat_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

        self.input_length = input_length
        self.flat_dim = flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x)).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def parameter_table(model: nn.Module) -> str:
    rows = [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()]
    width = max(len(name) for name, _, _ in rows)
    lines = [f"{'parameter':<{width}}  {'shape':<14}  {'count':>9}", "-" * (width + 27)]
    lines += [f"{name:<{width}}  {str(shape):<14}  {n:>9,}" for name, shape, n in rows]
    lines += ["-" * (width + 27), f"{'total':<{width}}  {'':<14}  {count_parameters(model):>9,}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Show BaselineCNN architecture and parameter count.")
    p.add_argument("--input-length", type=int, default=2001)
    args = p.parse_args(argv)

    model = BaselineCNN(input_length=args.input_length)
    print(model)
    print()
    print(parameter_table(model))
    print(f"\nFlattened feature size: {model.flat_dim}")
    out = model.eval()(torch.randn(4, 1, args.input_length))
    print(f"Output shape for batch of 4: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
