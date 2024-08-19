# from pdb import set_trace as T

import functools

import numpy as np
import gymnasium

import pufferlib.emulation
import pufferlib.postprocess


def single_env_creator(
    env_name, run_name, capture_video, gamma, idx=None, pufferl=False, center_reward=True
):
    if capture_video and idx == 0:
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gymnasium.make(env_name)
    env = EpisodeStats(env, center_reward=center_reward)
    env = pufferlib.postprocess.ClipAction(env)  # NOTE: this changed actions space
    env = gymnasium.wrappers.NormalizeObservation(env)
    env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    if pufferl is True:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return env


def cleanrl_env_creator(env_name, run_name, capture_video, gamma, idx):
    kwargs = {
        "env_name": env_name,
        "run_name": run_name,
        "capture_video": capture_video,
        "gamma": gamma,
        "idx": idx,
        "pufferl": False,
    }
    return functools.partial(single_env_creator, **kwargs)


def pufferl_env_creator(env_name, run_name, args_dict):
    env_fns = []
    default_kwargs = {
        "env_name": env_name,
        "run_name": run_name,
        "capture_video": args_dict["capture_video"],
        "gamma": args_dict["train"]["gamma"],
        "pufferl": True,
    }
    for idx in range(args_dict["train"]["num_envs"]):
        env_fns.append(functools.partial(single_env_creator, **{**default_kwargs, "idx": idx}))
    return env_fns


### Put the wrappers here, for now
class EpisodeStats(gymnasium.Wrapper):
    """Wrapper for Gymnasium environments that stores
    episodic returns and lengths in infos"""

    def __init__(self, env, center_reward=False):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset()

        self.center_reward = center_reward
        self.total_reward = 0
        self.total_steps = 0

    def reset(self, seed=None, options=None):
        self.info = dict(episode_return=0, episode_length=0)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # For reward centering, see https://arxiv.org/abs/2405.09999
        self.total_reward += reward
        self.total_steps += 1
        average_reward = self.total_reward / self.total_steps

        self.info["episode_return"] += reward
        self.info["episode_length"] += 1
        self.info["average_reward"] = average_reward

        if terminated or truncated:
            for k, v in self.info.items():
                info[k] = v

        if self.center_reward:
            reward -= average_reward

        return observation, reward, terminated, truncated, info
