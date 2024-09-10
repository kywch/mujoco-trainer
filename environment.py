# from pdb import set_trace as T
import time
import functools
from collections import deque

import numpy as np
import gymnasium

import pufferlib.emulation
import pufferlib.postprocess

import updated_envs  # noqa  # register the mujoco v5 envs


def single_env_creator(
    env_name,
    capture_video,
    video_dir=None,
    idx=None,
    norm_obs=False,  # to be removed
    pufferl=False,
    # gymnasium provided norm_reward
    rms_norm_reward=True,
    rms_norm_reward_gamma=0.99,
    # simpler norm_reward
    simp_norm_reward=False,  # just subtracting a const
    simp_norm_reward_bias=0.0,
):
    # Allow video capture only during eval
    if capture_video is True and idx == 0:
        if video_dir is None:
            video_dir = f"{env_name}_{int(time.time())}"
        env = gymnasium.make(env_name, render_mode="rgb_array", width=240, height=240)
        env = gymnasium.wrappers.RecordVideo(env, f"videos/{video_dir}")
    else:
        env = gymnasium.make(env_name)

    # Episode stats have the returns before normalization
    env = EpisodeStats(env)
    env = pufferlib.postprocess.ClipAction(env)  # NOTE: this changed actions space

    # TODO: If not necessary, comment out the below
    if norm_obs is True:
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

    # TODO: RMS norm reward is slow. If simple norm works, use that
    if simp_norm_reward is True:
        env = SimpleNormalizeReward(env)  # , bias=simp_norm_reward_bias)

    elif rms_norm_reward is True:
        env = RMSNormalizeReward(env, gamma=rms_norm_reward_gamma)
        env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    if pufferl is True:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env)

    return env


def cleanrl_env_creator(env_name, capture_video, gamma, idx):
    kwargs = {
        "env_name": env_name,
        "capture_video": capture_video,
        "idx": idx,
        "norm_obs": True,
        "pufferl": False,
        "rms_norm_reward": True,
        "rms_norm_reward_gamma": gamma,
        "simp_norm_reward": False,
    }
    return functools.partial(single_env_creator, **kwargs)


def pufferl_env_creator(env_name, args_dict, **env_kwargs):
    capture_video = args_dict["mode"] == "video"

    if "simp_norm_reward" in args_dict["env"]:
        use_simp_norm_reward = args_dict["env"]["simp_norm_reward"]
    else:
        use_simp_norm_reward = False

    default_kwargs = {
        "env_name": env_name,
        "capture_video": capture_video,
        "pufferl": True,
        "rms_norm_reward": not use_simp_norm_reward,
        "rms_norm_reward_gamma": args_dict["train"]["gamma"],
        "simp_norm_reward": use_simp_norm_reward,
    }
    default_kwargs.update(env_kwargs)

    if capture_video is False:
        return functools.partial(single_env_creator, **default_kwargs)

    # NOTE: When capturing videos, we need to create multiple envs with one video env
    return [
        functools.partial(single_env_creator, **{**default_kwargs, "idx": idx})
        for idx in range(args_dict["train"]["num_envs"])
    ]


### Put the wrappers here, for now
class EpisodeStats(gymnasium.Wrapper):
    """Wrapper for Gymnasium environments that stores
    episodic returns and lengths in infos"""

    def __init__(self, env, traj_history_len=100):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.episode_results = deque(maxlen=30)
        self.reached_horizon = deque(maxlen=30)
        self.total_reward = 0
        self.total_steps = 0

        # # For early termination
        # self.traj_rewards = deque(maxlen=traj_history_len)
        # self.early_terminiation_lookback = traj_history_len

        self.reset()

    def reset(self, seed=None, options=None):
        self.info = dict(episode_return=0, episode_length=0)
        # self.traj_rewards.clear()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # For video
        info["raw_reward"] = reward

        self.total_reward += reward
        self.total_steps += 1
        average_reward = self.total_reward / self.total_steps

        self.info["episode_return"] += reward
        self.info["episode_length"] += 1

        # # Check for early termination
        # # NOTE: In cheetah, agents were keep getting negative rew, leading to policy collapse?
        # self.traj_rewards.append(reward)
        # if (
        #     len(self.traj_rewards) >= self.early_terminiation_lookback
        #     and sum(self.traj_rewards) < 0
        # ):
        #     terminated = True

        if terminated or truncated:
            self.episode_results.append(self.info["episode_return"])
            info["last30episode_return"] = np.mean(self.episode_results)
            info["average_reward"] = average_reward

            for k, v in self.info.items():
                info[k] = v

        return observation, reward, terminated, truncated, info


class SimpleNormalizeReward(gymnasium.Wrapper):
    def __init__(self, env, scale=0.1, traj_history_len=100):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.reward_scale = scale

        # To get average reward
        self.sum_reward_raw = 0
        self.sum_reward_norm = 0
        self.total_steps = 1  # to avoid division by zero

        self.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        self.total_steps += 1
        self.sum_reward_raw += reward

        # TODO: try exponential moving average with alpha as a hyperparameter
        norm_rew = (reward - self.sum_reward_raw / self.total_steps) * self.reward_scale
        self.sum_reward_norm += norm_rew

        if terminated:
            norm_rew = -1.0

        if terminated or truncated:
            info["normalized_reward"] = self.sum_reward_norm / self.total_steps

        return observation, norm_rew, terminated, truncated, info


### Replace below with simple rew normalizations
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RMSNormalizeReward(gymnasium.core.Wrapper, gymnasium.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gymnasium.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        # To track rewards
        self.total_reward = 0
        self.total_steps = 0

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]

        # For logging rewards
        self.total_reward += rews
        self.total_steps += 1

        if terminateds or truncateds:
            infos["normalized_reward"] = self.total_reward / self.total_steps

        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
