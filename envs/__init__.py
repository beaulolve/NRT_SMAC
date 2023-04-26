from functools import partial
from .starcraft2.starcraft2 import StarCraft2Env
from .multiagentenv import MultiAgentEnv

import sys
import os

from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
}


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
