import torch
from tensordict.nn import CudaGraphModule

from train import parse_args, make_policy
import environment
import policy

import pufferlib
import pufferlib.vector

torch.set_float32_matmul_precision("high")

# This should show '2.4.0' and include 'inductor' in the list of available backends.
print(torch.__version__)
print(torch.compiler.list_backends())


args, env_name = parse_args(config="config/debug_cuda.toml")

env_creator = environment.pufferl_env_creator(env_name, args)
policy_cls = getattr(policy, args["base"]["policy_name"])
vec = pufferlib.vector.Serial

vecenv = pufferlib.vector.make(
    env_creator,
    env_kwargs=args["env"],
    num_envs=args["train"]["num_envs"],
    num_workers=args["train"]["num_workers"],
    batch_size=args["train"]["env_batch_size"],
    zero_copy=args["train"]["zero_copy"],
    backend=vec,
)

agent = make_policy(vecenv.driver_env, policy_cls, None, args)

# Try compiling the entire model
compiled_agent = torch.compile(agent, mode=args["train"]["compile_mode"])

obs = torch.randn(1, agent.obs_size, device=args["train"]["device"])
action, log_prob, entropy, value = compiled_agent.get_action_and_value(obs)
print("Compilation successful!")

# Compiling the forward fn
compiled_forward = torch.compile(
    agent.forward, backend="inductor", mode=args["train"]["compile_mode"]
)
for _ in range(10):
    action, log_prob, entropy, value = compiled_forward(obs)
print("Compilation the function successful!")

# Wrap the function with cudagraphs
cudagraph_forward = CudaGraphModule(agent.forward)
for _ in range(10):
    torch.compiler.cudagraph_mark_step_begin()
    action, log_prob, entropy, value = cudagraph_forward(obs)
print("Capturing the cudagraphs successful!")

# NOTE: This will cause an error
# Have not figured out how to use both cudagraphs and inductor
compiled_cudagraph_forward = CudaGraphModule(compiled_forward)
for _ in range(10):
    torch.compiler.cudagraph_mark_step_begin()
    action, log_prob, entropy, value = compiled_cudagraph_forward(obs)
print("Using both successful!")
