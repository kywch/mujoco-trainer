import argparse
import tomllib
import signal
import random
import uuid
import time
import ast
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich.traceback import install

import clean_pufferl
import environment
import policy

from utils import init_wandb

# Rich tracebacks
install(show_locals=False)

# Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args["policy"])
    if isinstance(policy, pufferlib.frameworks.cleanrl.Policy) or isinstance(
        policy, pufferlib.frameworks.cleanrl.RecurrentPolicy
    ):
        pass
    elif rnn_cls is not None:
        policy = rnn_cls(env, policy, **args["rnn"])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args["train"]["device"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training arguments for gymnasium mujoco", add_help=False
    )
    parser.add_argument("-c", "--config", default="config/debug.toml")
    parser.add_argument(
        "-e",
        "--env-name",
        type=str,
        default="Ant-v5",
        help="Name of specific environment to run",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="sweep",
        choices="train video sweep autotune profile".split(),
    )
    # parser.add_argument("--eval-model-path", type=str, default=None)
    parser.add_argument(
        "--eval-model-path", type=str, default="experiments/Ant-v5-380ce758/model_000136.pt"
    )

    parser.add_argument(
        "--baseline", action="store_true", help="Pretrained baseline where available"
    )
    parser.add_argument(
        "-v",
        "--vec",
        type=str,
        default="serial",
        choices=["serial", "multiprocessing"],
    )
    parser.add_argument(
        "--exp-id", "--exp-name", type=str, default=None, help="Resume from experiment"
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat the training with different seeds"
    )
    parser.add_argument("-d", "--device", type=str, default=None)
    # parser.add_argument("--capture-video", action="store_true", help="Capture videos")

    args = parser.parse_known_args()[0]

    # Load config file
    if not os.path.exists(args.config):
        raise Exception(f"Config file {args.config} not found")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    for section in config:
        for key in config[section]:
            argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=config[section][key])

    # Override config with command line arguments
    parsed = parser.parse_args().__dict__
    args = {"env": {}, "policy": {}, "rnn": {}}
    env_name = parsed.pop("env_name")
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:  # noqa
            prev[subkey] = value

    run_name = f"{env_name}_{args['train']['seed']}_{int(time.time())}"

    return args, env_name, run_name


def train(args, env_creator, policy_cls, rnn_cls, wandb=None, skip_dash=False):
    if args["vec"] == "serial":
        vec = pufferlib.vector.Serial
    elif args["vec"] == "multiprocessing":
        vec = pufferlib.vector.Multiprocessing
    else:
        raise ValueError("Invalid --vector (serial/multiprocessing).")

    vecenv = pufferlib.vector.make(
        env_creator,
        env_kwargs=args["env"],
        num_envs=args["train"]["num_envs"],
        num_workers=args["train"]["num_workers"],
        batch_size=args["train"]["env_batch_size"],
        zero_copy=args["train"]["zero_copy"],
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    train_config = pufferlib.namespace(
        **args["train"],
        env=env_name,
        exp_id=args["exp_id"] or env_name + "-" + str(uuid.uuid4())[:8],
    )
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb, skip_dash=skip_dash)

    try:
        while data.global_step < train_config.total_timesteps:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)

        uptime = data.profile.uptime

        # TODO: If we CARBS reward param, we should have a separate venenv ready for eval
        # Run evaluation to get the average stats
        stats = []
        data.vecenv.async_reset(seed=int(time.time()))
        num_eval_epochs = train_config.eval_timesteps // train_config.batch_size
        for _ in range(1 + num_eval_epochs):  # extra data for sweeps
            stats.append(clean_pufferl.evaluate(data)[0])

    except Exception as e:  # noqa
        uptime, stats = 0, []
        import traceback

        traceback.print_exc()

    clean_pufferl.close(data)
    return stats, uptime


### CARBS Sweeps
def sweep_carbs(args, env_name, env_creator, policy_cls, rnn_cls):
    import wandb
    from utils import init_carbs, carbs_runner_fn

    if not os.path.exists("carbs_checkpoints"):
        os.system("mkdir carbs_checkpoints")

    carbs = init_carbs(args, num_random_samples=20)

    sweep_id = wandb.sweep(
        sweep=args["sweep"],
        project="carbs",
    )

    def train_fn(args, wandb):
        return train(args, env_creator, policy_cls, rnn_cls, wandb=wandb, skip_dash=True)

    # Run sweep
    wandb.agent(
        sweep_id,
        carbs_runner_fn(args, env_name, carbs, sweep_id, train_fn),
        count=args["train"]["num_sweeps"],
    )


if __name__ == "__main__":
    args, env_name, run_name = parse_args()
    run_name = "pufferl_" + run_name

    if args["device"] is not None:
        args["train"]["device"] = args["device"]

    # Load env binding and policy
    env_creator = environment.pufferl_env_creator(env_name, run_name, args)
    policy_cls = getattr(policy, args["base"]["policy_name"])
    rnn_cls = None
    if "rnn_name" in args["base"]:
        rnn_cls = getattr(policy, args["base"]["rnn_name"])

    # Process mode
    if args["mode"] == "train":
        assert args["repeat"] > 0, "Repeat count must be positive"
        if args["repeat"] > 1:
            args["track"] = True
            assert args["wandb_group"] is not None, "Repeating requires a wandb group"
        wandb = None

        for i in range(args["repeat"]):
            if i > 0:
                # Generate a new 8-digit seed
                args["train"]["seed"] = random.randint(10_000_000, 99_999_999)

            if args["track"]:
                wandb = init_wandb(args, run_name, id=i)
            train(args, env_creator, policy_cls, rnn_cls, wandb=wandb)

    elif args["mode"] == "video":
        # Single env
        args["train"]["num_envs"] = 1
        args["train"]["num_workers"] = 1
        args["train"]["env_batch_size"] = 1

        clean_pufferl.rollout(
            env_creator[0],
            args["env"],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            model_path=args["eval_model_path"],
            device=args["train"]["device"],
        )

    elif args["mode"] == "sweep":
        env_creator = environment.pufferl_env_creator(env_name, run_name, args)
        sweep_carbs(args, env_name, env_creator, policy_cls, rnn_cls)

    elif args["mode"] == "autotune":
        pufferlib.vector.autotune(env_creator, batch_size=args["train"]["env_batch_size"])

    elif args["mode"] == "profile":
        # TODO: Profile gymnasium wrappers
        import cProfile

        args["train"]["total_timesteps"] = 10_000
        cProfile.run("train(args, env_creator, policy_cls, rnn_cls)", "stats.profile")
        import pstats
        from pstats import SortKey

        p = pstats.Stats("stats.profile")
        p.sort_stats(SortKey.TIME).print_stats(10)
