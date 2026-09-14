"""Import test + probe of raw env API behaviour. Run before writing/changing env/make_env.py."""
import time
import warnings

import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch

print("numpy", np.__version__, "gym", gym.__version__, "torch", torch.__version__)

base = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="rgb_array")
print("wrapper chain:", base)
print("max_episode_steps:", base.spec.max_episode_steps)

env = JoypadSpace(base, [["right"], ["right", "A"]])
try:
    out = env.reset(seed=0)
    print("JoypadSpace.reset(seed=0) ok ->", type(out))
except TypeError as e:
    print("JoypadSpace.reset(seed=0) FAILS:", e)
    out = env.reset()
print("reset() returns:", type(out), len(out) if isinstance(out, tuple) else "", getattr(out[0] if isinstance(out, tuple) else out, "shape", None))

step_out = env.step(0)
print("step() arity:", len(step_out), "obs dtype:", step_out[0].dtype, "info keys:", sorted(step_out[-1]))

# Hold NOOP-ish 'right' into a wall is hard to guarantee; instead stand still with an action set containing NOOP
# and run until the episode ends, recording why.
env2 = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True, render_mode="rgb_array"), SIMPLE_MOVEMENT)
env2.reset()
steps, t0, rew = 0, time.time(), []
while True:
    obs, r, term, trunc, info = env2.step(0)  # NOOP: Mario stands still, clock runs out
    rew.append(r)
    steps += 1
    if term or trunc:
        break
dt = time.time() - t0
print(f"NOOP episode: frames={steps} terminated={term} truncated={trunc} time={info['time']} life={info['life']} "
      f"x_pos={info['x_pos']} fps={steps/dt:.0f}")
print("reward range seen:", min(rew), max(rew), "sum:", sum(rew))

# NumPy 2 failure mode check (documentation of the pin): uint8 * 256 under legacy promotion
print("uint8(3)*0x100 =", np.uint8(3) * 0x100, type(np.uint8(3) * 0x100))
