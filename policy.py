import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.models
import pufferlib.frameworks.cleanrl
from pufferlib.pytorch import layer_init


# class Recurrent(pufferlib.models.LSTMWrapper):
#     def __init__(self, env, policy, input_size=128, hidden_size=128, num_layers=1):
#         super().__init__(env, policy, input_size, hidden_size, num_layers)


class Debug(pufferlib.models.Default):
    def __init__(self, env, hidden_size=64):
        super().__init__(env, hidden_size)


class CleanRLPolicy(pufferlib.frameworks.cleanrl.Policy):
    def __init__(self, envs, hidden_size=64):
        super().__init__(policy=None)  # Just to get the right init
        self.is_continuous = True

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor_encoder = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_size)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor_decoder_mean = layer_init(
            nn.Linear(hidden_size, np.prod(envs.single_action_space.shape)), std=0.01
        )
        self.actor_decoder_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        x = x.float()
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float()
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


# class Policy(nn.Module):
#     def __init__(self, env, hidden_size=128):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), hidden_size)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, hidden_size)),
#             nn.Tanh(),
#         )

#         self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
#         if self.is_continuous:
#             self.decoder_mean = layer_init(
#                 nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
#             )
#             self.decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))
#         else:
#             self.decoders = torch.nn.ModuleList(
#                 [layer_init(torch.nn.Linear(hidden_size, n)) for n in env.single_action_space.nvec]
#             )

#         self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)

#     def forward(self, observations):
#         hidden, lookup = self.encode_observations(observations)
#         actions, value = self.decode_actions(hidden, lookup)
#         return actions, value

#     def encode_observations(self, observations):
#         """Encodes a batch of observations into hidden states. Assumes
#         no time dimension (handled by LSTM wrappers)."""
#         batch_size = observations.shape[0]
#         observations = observations.view(batch_size, -1)
#         return self.encoder(observations.float()), None

#     def decode_actions(self, hidden, lookup, concat=True):
#         """Decodes a batch of hidden states into (multi)discrete actions.
#         Assumes no time dimension (handled by LSTM wrappers)."""
#         value = self.value_head(hidden)

#         if self.is_continuous:
#             mean = self.decoder_mean(hidden)
#             logstd = self.decoder_logstd.expand_as(mean)
#             std = torch.exp(logstd)
#             probs = torch.distributions.Normal(mean, std)
#             # batch = hidden.shape[0]
#             return probs, value
#         else:
#             actions = [dec(hidden) for dec in self.decoders]
#             return actions, value
