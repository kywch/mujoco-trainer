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
    env = EpisodeStats(env)
    env = pufferlib.postprocess.ClipAction(env)  # NOTE: this changed actions space
    env = gymnasium.wrappers.NormalizeObservation(env)
    env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = NormalizeReward(env, gamma=gamma)
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


class NormalizeReward(gymnasium.core.Wrapper, gymnasium.utils.RecordConstructorArgs):
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
