"""Run configuration. One dataclass, two presets, and a hash over the fields that change learning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    # --- environment ---
    world: int = 1
    stage: int = 1
    actions: str = "right_only"  # "right_only" | "simple"
    skip: int = 4
    frame_size: int = 84
    stack: int = 4
    no_progress_steps: int = 150  # truncate if max x_pos hasn't improved in this many agent steps
    noop_max: int = 30  # random 0..noop_max agent steps of NOOP after reset

    # --- learning ---
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 32
    buffer_size: int = 100_000
    burnin: int = 10_000  # min transitions in buffer before learning
    learn_every: int = 3
    sync_every: int = 10_000
    reward_scale: float = 1 / 15  # applied to summed skip-frame reward before storing
    max_grad_norm: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 1_000_000
    seed: int = 0

    # --- run bookkeeping (not part of the learning hash) ---
    total_steps: int = 3_000_000
    save_every: int = 100_000
    snapshot_every: int = 250_000  # keep a separate weights-only checkpoint so any point in the run can be evaluated
    save_buffer: bool = False
    log_every_episodes: int = 20
    device: str = "cpu"
    torch_threads: int = 0  # 0 = torch default

    def learning_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in BOOKKEEPING_FIELDS}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


BOOKKEEPING_FIELDS = frozenset(
    {"total_steps", "save_every", "snapshot_every", "save_buffer", "log_every_episodes", "device", "torch_threads"}
)

# Small enough to finish in about a minute on CPU, but every code path fires:
# learning starts, target sync runs several times, checkpoint (with buffer) is written.
SMOKE = Config(
    buffer_size=5_000,
    burnin=200,
    sync_every=500,
    eps_decay_steps=1_500,
    total_steps=2_000,
    save_every=1_000,
    snapshot_every=1_000,
    save_buffer=True,
    log_every_episodes=1,
)

FULL = Config()

PRESETS = {"smoke": SMOKE, "full": FULL}


def get_preset(name: str, **overrides) -> Config:
    return replace(PRESETS[name], **overrides)


# ============================================================================ PPO on tile grids, many stages
@dataclass(frozen=True)
class PPOConfig:
    # --- environment ---
    train_stages: tuple = ()  # empty -> env.tiles.TRAIN_STAGES
    test_stages: tuple = ()  # empty -> env.tiles.TEST_STAGES
    actions: str = "simple"
    skip: int = 4
    stack: int = 4
    no_progress_steps: int = 200
    # Random start delay. Every reset emulates up to noop_max idle steps while the other 11 envs wait, so
    # 30 cost ~35% rollout speed (628 vs 990 steps/s); 8 keeps start variety at 856 steps/s.
    # eval_stages.py still evaluates with 30.
    noop_max: int = 8
    coin_reward: float = 15.0  # raw game units; ~1 agent step of full-speed running
    flag_reward: float = 150.0

    # --- learning ---
    n_envs: int = 12
    rollout_len: int = 256
    epochs: int = 4
    minibatches: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 2.5e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 1 / 15
    seed: int = 0

    # --- bookkeeping ---
    total_steps: int = 20_000_000
    save_every: int = 250_000
    snapshot_every: int = 1_000_000
    device: str = "cpu"
    torch_threads: int = 8  # learning phase (envs idle): 8 threads 1.58 s/update vs 2.20 s at 4
    rollout_threads: int = 1  # inference while 12 env workers run: 1 thread avoids contention

    def env_kwargs(self, stages) -> dict:
        return dict(stages=list(stages), actions=self.actions, skip=self.skip, stack=self.stack,
                    no_progress_steps=self.no_progress_steps, noop_max=self.noop_max,
                    coin_reward=self.coin_reward, flag_reward=self.flag_reward)

    def resolved_stages(self) -> tuple[list, list]:
        from env.tiles import TEST_STAGES, TRAIN_STAGES
        return list(self.train_stages or TRAIN_STAGES), list(self.test_stages or TEST_STAGES)

    def learning_hash(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in PPO_BOOKKEEPING_FIELDS}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})


PPO_BOOKKEEPING_FIELDS = frozenset(
    {"total_steps", "save_every", "snapshot_every", "device", "torch_threads", "rollout_threads"})

PPO_PRESETS = {
    # Every code path in about a minute: vec env, truncation bootstrap, updates, checkpoint, snapshot.
    "ppo_smoke": PPOConfig(train_stages=("1-1", "1-2"), test_stages=("1-3",), n_envs=4, rollout_len=64,
                           minibatches=4, total_steps=2_048, save_every=512, snapshot_every=1_024,
                           no_progress_steps=40, torch_threads=2, rollout_threads=1),
    "ppo_full": PPOConfig(),
    # One-hour run for fast iteration: measured 520 steps/s (12 envs, after speed fixes) x 3300 s; snapshots
    # every ~15 min (520 x 900). Evals running alongside will stretch the wall-clock time somewhat.
    "ppo_1h": PPOConfig(total_steps=1_716_000, snapshot_every=468_000, save_every=100_000),
}


def get_ppo_preset(name: str, **overrides) -> PPOConfig:
    return replace(PPO_PRESETS[name], **overrides)
