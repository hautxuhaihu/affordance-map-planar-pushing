"""三模型：MeanBaseline, LinearBaseline, OutcomeMLP。"""

import torch
import torch.nn as nn


class MeanBaseline(nn.Module):
    """输出训练集标签均值，作为最低基准。"""

    def __init__(self, target_mean: torch.Tensor):
        super().__init__()
        self.register_buffer("target_mean", target_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_mean.expand(x.shape[0], -1)


class LinearBaseline(nn.Module):
    """线性回归 baseline: Linear(input_dim, 3)。"""

    def __init__(self, input_dim: int = 10, output_dim: int = 3):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OutcomeMLP(nn.Module):
    """小型 MLP: Linear(input_dim,64)->ReLU->Linear(64,64)->ReLU->Linear(64,3)。"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
