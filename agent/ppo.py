"""PPO over tile-grid observations."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from env.tiles import COLS, ROWS, VOCAB


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class TilePolicy(nn.Module):
    """Embed tile ids -> small CNN over the 13x16 grid -> concat extras -> policy logits and value."""

    def __init__(self, n_actions: int, stack: int, n_extras: int, emb_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, emb_dim)
        self.conv = nn.Sequential(
            _init(nn.Conv2d(stack * emb_dim, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * ((ROWS + 1) // 2) * ((COLS + 1) // 2)
        self.fc = nn.Sequential(_init(nn.Linear(conv_out + n_extras, 256)), nn.ReLU())
        self.pi = _init(nn.Linear(256, n_actions), gain=0.01)
        self.v = _init(nn.Linear(256, 1), gain=1.0)

    def forward(self, tiles: torch.Tensor, extras: torch.Tensor):
        b, s, h, w = tiles.shape
        x = self.embed(tiles.long())  # [B, S, H, W, E]
        x = x.permute(0, 1, 4, 2, 3).reshape(b, s * x.shape[-1], h, w)
        x = self.fc(torch.cat([self.conv(x), extras], dim=1))
        return self.pi(x), self.v(x).squeeze(-1)


def compute_gae(rewards, values, last_values, terminated, truncated, trunc_values, gamma, lam):
    """Generalized advantage estimation over a [T, N] rollout.

    terminated: the future is zero. truncated: the episode was cut short, so bootstrap from the value of the
    real final observation (trunc_values), not from the next row, which already belongs to a new episode.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        done = terminated[t] | truncated[t]
        next_v = last_values if t == T - 1 else values[t + 1]
        next_v = np.where(done, np.where(terminated[t], 0.0, trunc_values[t]), next_v)
        delta = rewards[t] + gamma * next_v - values[t]
        last_gae = delta + gamma * lam * (~done) * last_gae
        adv[t] = last_gae
    return adv, adv + values


class PPOAgent:
    def __init__(self, cfg, n_actions: int, n_extras: int):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = torch.device(cfg.device)
        self.net = TilePolicy(n_actions, cfg.stack, n_extras).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.global_step = 0
        self.updates = 0

    def _tensors(self, obs: dict):
        return (torch.as_tensor(obs["tiles"], device=self.device),
                torch.as_tensor(obs["extras"], device=self.device))

    @torch.no_grad()
    def act(self, obs: dict, greedy: bool = False, generator: torch.Generator | None = None):
        logits, value = self.net(*self._tensors(obs))
        if greedy:
            action = logits.argmax(dim=1)
        else:
            action = torch.multinomial(torch.softmax(logits, dim=1), 1, generator=generator).squeeze(1)
        logp = torch.log_softmax(logits, dim=1).gather(1, action.unsqueeze(1)).squeeze(1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def value(self, obs: dict) -> np.ndarray:
        return self.net(*self._tensors(obs))[1].cpu().numpy()

    def update(self, batch: dict) -> dict:
        c = self.cfg
        n = len(batch["actions"])
        t = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
        mb_size = n // c.minibatches
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, mb_size * c.minibatches, mb_size):
                idx = perm[start:start + mb_size]
                logits, values = self.net(t["tiles"][idx], t["extras"][idx])
                logp_all = torch.log_softmax(logits, dim=1)
                logp = logp_all.gather(1, t["actions"][idx].long().unsqueeze(1)).squeeze(1)
                entropy = -(logp_all.exp() * logp_all).sum(1).mean()

                adv = t["advantages"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = logp - t["logp"][idx]
                ratio = log_ratio.exp()
                policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - c.clip, 1 + c.clip)).mean()
                value_loss = 0.5 * (values - t["returns"][idx]).pow(2).mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip).float().mean().item())
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
        self.updates += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        var_y = np.var(batch["returns"])
        out["explained_variance"] = float(1 - np.var(batch["returns"] - batch["values"]) / var_y) if var_y > 0 else 0.0
        return out

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step, "updates": self.updates,
            "config": self.cfg.to_dict(), "learning_hash": self.cfg.learning_hash(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
        }, path)

    def load(self, path: str | Path, allow_config_change: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt["learning_hash"] != self.cfg.learning_hash() and not allow_config_change:
            raise ValueError(f"learning config changed since checkpoint ({ckpt['learning_hash']} -> "
                             f"{self.cfg.learning_hash()}); pass --allow-config-change if intentional")
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step, self.updates = ckpt["global_step"], ckpt["updates"]
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
