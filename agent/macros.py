"""Macro actions (options) for single-env use: eval, demo, GIFs, diagnostics.

Training expands macros inside the rollout loop instead (train_ppo.py), so no env waits on another.
"""
from __future__ import annotations

from env.tiles import MACRO_SETS


class MacroStepper:
    """step(policy_action): joypad actions pass through; a macro holds its buttons for its length."""

    def __init__(self, env, actions: str):
        self.env = env
        self.macros = MACRO_SETS.get(actions, [])
        self.n_joy = env.n_actions

    def expand(self, policy_action: int) -> tuple[int, int]:
        """(joypad action, number of agent steps)."""
        if policy_action < self.n_joy:
            return policy_action, 1
        _, joy, steps = self.macros[policy_action - self.n_joy]
        return joy, steps

    def step(self, policy_action: int):
        joy, steps = self.expand(int(policy_action))
        total = 0.0
        for _ in range(steps):
            obs, reward, terminated, truncated, info = self.env.step(joy)
            total += reward
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
