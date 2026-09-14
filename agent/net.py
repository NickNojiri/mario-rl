"""Q-networks. Inputs arrive as uint8; normalization happens here, as late as possible."""
from __future__ import annotations

import copy

import torch
from torch import nn


class MarioNet(nn.Module):
    """Nature-DQN CNN with a frozen target copy. forward(x, model="online"|"target")."""

    def __init__(self, in_channels: int, n_actions: int, frame_size: int = 84):
        super().__init__()
        if frame_size != 84:
            raise ValueError(f"MarioNet expects 84x84 frames, got {frame_size}")
        self.online = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, model: str = "online") -> torch.Tensor:
        if x.dtype != torch.uint8:
            raise TypeError(f"expected uint8 observations, got {x.dtype}")
        x = x.float() / 255.0
        return self.online(x) if model == "online" else self.target(x)

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
