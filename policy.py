import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.models
import pufferlib.frameworks.cleanrl
from pufferlib.pytorch import layer_init


# This replaces gymnasium's NormalizeObservation wrapper
# NOTE: Tried BatchNorm1d with momentum=None, but the policy did not learn. Check again later.
class RunningNorm(nn.Module):
    def __init__(self, shape, epsilon=1e-5, clip=10.0):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.count = 1
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            weight = 1 / self.count
            self.running_mean = self.running_mean * (1 - weight) + mean * weight
            self.running_var = self.running_var * (1 - weight) + var * weight
            self.count += 1

        return torch.clamp(
            (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon),
            -self.clip,
            self.clip,
        )


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=64, hidden_size=64, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Debug(pufferlib.models.Default):
    def __init__(self, env, hidden_size=64):
        super().__init__(env, hidden_size)


# Difference between CleanRL and PufferRL policies is that
# CleanRL has seperate actor and critic networks, while
# PufferRL has a single encoding network that is shared between
# the actor and critic networks
class Policy(torch.nn.Module):
    def __init__(self, env, hidden_size=64):
        super().__init__()
        self.is_continuous = True

        obs_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)

        self.obs_encoder = nn.Sequential(
            RunningNorm(obs_size),
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.actor_decoder_mean = layer_init(nn.Linear(hidden_size, action_size), std=0.01)
        self.actor_decoder_logstd = nn.Parameter(torch.zeros(1, action_size))

        self.value_head = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    # NOTE: these are for the LSTM wrapper, which may NOT work as intended
    def encode_observations(self, observations):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        observations = observations.float()
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        return self.obs_encoder(observations), None

    def decode_actions(self, hidden, lookup, concat=True):
        """Decodes a batch of hidden states into continuous actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        value = self.value_head(hidden)

        mean = self.actor_decoder_mean(hidden)
        logstd = self.actor_decoder_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = torch.distributions.Normal(mean, std)
        # batch = hidden.shape[0]

        return probs, value


class SepCriticLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=64, hidden_size=64, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

    def forward(self, x, state):
        x = x.float()
        hidden, _, state = super().forward(x, state)
        value = self.policy.critic(x.reshape(-1, self.policy.obs_size))
        return hidden, value, state


class SepCriticPolicy(torch.nn.Module):
    def __init__(self, env, hidden_size=64):
        super().__init__()
        self.is_continuous = True

        self.obs_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)

        self.actor_encoder = nn.Sequential(
            RunningNorm(self.obs_size),
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.actor_decoder_mean = layer_init(nn.Linear(hidden_size, action_size), std=0.01)
        self.actor_decoder_logstd = nn.Parameter(torch.zeros(1, action_size))

        self.critic = nn.Sequential(
            RunningNorm(self.obs_size),
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward(self, observations):
        observations = observations.float()
        hidden, lookup = self.encode_observations(observations)
        actions, _ = self.decode_actions(hidden, lookup)
        value = self.critic(observations)
        return actions, value

    def encode_observations(self, observations):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        batch_size = observations.shape[0]
        observations = observations.view(batch_size, -1)
        return self.actor_encoder(observations), None

    def decode_actions(self, hidden, lookup, concat=True):
        """Decodes a batch of hidden states into continuous actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        # value = self.value_head(hidden)

        mean = self.actor_decoder_mean(hidden)
        logstd = self.actor_decoder_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = torch.distributions.Normal(mean, std)
        # batch = hidden.shape[0]

        # NOTE: Value is computed elsewhere. The below mean is just a hack to get the shape right
        return probs, mean[:, 0]


# TODO: test 128 width, with the lstm
class CleanRLPolicy(pufferlib.frameworks.cleanrl.Policy):
    def __init__(self, envs, hidden_size=64):
        super().__init__(policy=None)  # Just to get the right init
        self.is_continuous = True

        self.obs_size = np.array(envs.single_observation_space.shape).prod()
        action_size = np.prod(envs.single_action_space.shape)

        self.obs_norm = RunningNorm(self.obs_size)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor_decoder_mean = layer_init(nn.Linear(hidden_size, action_size), std=0.01)
        self.actor_decoder_logstd = nn.Parameter(torch.zeros(1, action_size))

    def get_value(self, x):
        x = x.float()
        x = self.obs_norm(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float()
        x = self.obs_norm(x)
        batch = x.shape[0]

        encoding = self.actor_encoder(x)
        action_mean = self.actor_decoder_mean(encoding)
        action_logstd = self.actor_decoder_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        logits = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = logits.sample()  # .view(batch, -1)

        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)

        # NOTE: entropy can go negative, when std is small (e.g. 0.1)
        logits_entropy = logits.entropy().sum(1)  # .view(batch, -1).sum(1)

        return action, log_probs, logits_entropy, self.critic(x)


class RegCritPolicy(CleanRLPolicy):
    def __init__(self, envs, hidden_size=64):
        super().__init__(envs, hidden_size=hidden_size)

        # Learn to walk in 20 min: https://arxiv.org/abs/2208.07860
        # Used LayerNorm to regularize the critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
