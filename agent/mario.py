"""Double-DQN agent: act / cache / learn, plus checkpoints that actually resume."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from agent.buffer import FrameReplayBuffer
from agent.net import MarioNet
from config import Config


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    return start + (step / decay_steps) * (end - start)


class Mario:
    def __init__(self, cfg: Config, n_actions: int):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = torch.device(cfg.device)
        self.net = MarioNet(cfg.stack, n_actions, cfg.frame_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=cfg.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.buffer = FrameReplayBuffer(cfg.buffer_size, (cfg.frame_size, cfg.frame_size), cfg.stack, seed=cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.curr_step = 0
        self.episode = 0

    @property
    def exploration_rate(self) -> float:
        c = self.cfg
        return linear_epsilon(self.curr_step, c.eps_start, c.eps_end, c.eps_decay_steps)

    def act(self, state: np.ndarray, epsilon: float | None = None, rng: np.random.Generator | None = None) -> int:
        """Epsilon-greedy. Pass epsilon/rng for eval; training uses the schedule and advances curr_step."""
        training = epsilon is None
        eps = self.exploration_rate if training else epsilon
        rng = rng or self.rng
        if rng.random() < eps:
            action = int(rng.integers(self.n_actions))
        else:
            with torch.no_grad():
                x = torch.from_numpy(state).unsqueeze(0).to(self.device)
                action = int(self.net(x, "online").argmax(dim=1).item())
        if training:
            self.curr_step += 1
        return action

    def cache(self, state, action, reward, next_state, terminated: bool, truncated: bool):
        self.buffer.add(state, action, reward * self.cfg.reward_scale, next_state, terminated, truncated)

    def learn(self) -> tuple[float, float] | None:
        """Returns (mean_q, loss) when an update ran, else None."""
        c = self.cfg
        if self.curr_step % c.sync_every == 0:
            self.net.sync_target()
        if len(self.buffer) < c.burnin or self.curr_step % c.learn_every != 0:
            return None

        state, action, reward, next_state, terminated = (t.to(self.device) for t in self.buffer.sample(c.batch_size))
        td_est = self.net(state, "online").gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            best_next = self.net(next_state, "online").argmax(dim=1, keepdim=True)
            next_q = self.net(next_state, "target").gather(1, best_next).squeeze(-1)
            td_tgt = reward + c.gamma * (1.0 - terminated) * next_q
        assert td_est.shape == td_tgt.shape == (c.batch_size,), (td_est.shape, td_tgt.shape)

        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.online.parameters(), c.max_grad_norm)
        self.optimizer.step()
        return td_est.mean().item(), loss.item()

    # --- checkpoints ---
    def save(self, path: str | Path, save_buffer: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "curr_step": self.curr_step,
                "episode": self.episode,
                "exploration_rate": self.exploration_rate,
                "config": self.cfg.to_dict(),
                "learning_hash": self.cfg.learning_hash(),
                "rng": {
                    "agent": self.rng.bit_generator.state,
                    "buffer": self.buffer.rng.bit_generator.state,
                    "python": random.getstate(),
                    "numpy_global": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                },
            },
            path,
        )
        if save_buffer:
            self.buffer.save(buffer_path(path))
        elif buffer_path(path).exists():
            buffer_path(path).unlink()  # never let load() pair these weights with an older buffer

    def load(self, path: str | Path, allow_config_change: bool = False) -> bool:
        """Restore everything. Returns True if a replay buffer was restored too."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt["learning_hash"] != self.cfg.learning_hash() and not allow_config_change:
            raise ValueError(
                f"learning config changed since checkpoint ({ckpt['learning_hash']} -> {self.cfg.learning_hash()}); "
                "pass --allow-config-change if intentional"
            )
        self.net.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.curr_step = ckpt["curr_step"]
        self.episode = ckpt["episode"]
        rng = ckpt["rng"]
        self.rng.bit_generator.state = rng["agent"]
        self.buffer.rng.bit_generator.state = rng["buffer"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy_global"])
        torch.set_rng_state(rng["torch"])
        bp = buffer_path(path)
        if bp.exists():
            self.buffer.load(bp)
            return True
        return False


def buffer_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_name(ckpt_path.stem + "_buffer.npz")
