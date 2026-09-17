"""Run N TileMarioEnvs in worker processes. Episodes auto-reset; the finished episode's last observation
is returned in info["final_obs"] so PPO can bootstrap truncated episodes correctly."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _worker(conn, env_kwargs: dict):
    from env.tiles import TileMarioEnv  # import inside the child process

    env = TileMarioEnv(**env_kwargs)
    env.prebuild()
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                conn.send((obs, info))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info["final_obs"] = obs
                    obs, _ = env.reset()
                conn.send((obs, reward, terminated, truncated, info))
            elif cmd == "close":
                break
    finally:
        env.close()
        conn.close()


def stack_obs(obs_list: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


class SubprocVecEnv:
    def __init__(self, n_envs: int, env_kwargs: dict):
        ctx = mp.get_context("forkserver")  # torch is already loaded in the parent; don't fork its threads
        self.n_envs = n_envs
        self._conns, self._procs = [], []
        for _ in range(n_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, env_kwargs), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)

    def reset(self, seed: int) -> dict:
        for i, c in enumerate(self._conns):
            c.send(("reset", seed + i))
        return stack_obs([c.recv()[0] for c in self._conns])

    def step(self, actions):
        for c, a in zip(self._conns, actions):
            c.send(("step", int(a)))
        results = [c.recv() for c in self._conns]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (stack_obs(list(obs)), np.asarray(rewards, dtype=np.float32), np.asarray(terminated),
                np.asarray(truncated), list(infos))

    def close(self):
        for c in self._conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
