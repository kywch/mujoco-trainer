# from pdb import set_trace as T

# import functools

import numpy as np
import gymnasium


def wrap_env(env, gamma):
    # env = gymnasium.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = EpisodeStats(env)
    env = gymnasium.wrappers.ClipAction(env)
    env = gymnasium.wrappers.NormalizeObservation(env)
    env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


def cleanrl_env_creator(env_name, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gymnasium.make(env_name, render_mode="rgb_array")
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gymnasium.make(env_name)
        env = wrap_env(env, gamma)
        return env

    return thunk


def pufferl_env_creator(env_name):
    # Make a list of env fns, with the first one having the video capture
    pass


### Put the wrappers here, for now
class EpisodeStats(gymnasium.Wrapper):
    """Wrapper for Gymnasium environments that stores
    episodic returns and lengths in infos"""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset()

    def reset(self, seed=None, options=None):
        self.info = dict(episode_return=0, episode_length=0)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        self.info["episode_return"] += reward
        self.info["episode_length"] += 1

        if terminated or truncated:
            for k, v in self.info.items():
                info[k] = int(v)

        return observation, reward, terminated, truncated, info
