import os
import ast
import time
import tomllib
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training arguments for myosuite", add_help=False)
    parser.add_argument("-c", "--config", default="config_debug.toml")
    parser.add_argument(
        "-e",
        "--env-name",
        type=str,
        default="Hopper-v4",
        help="Name of specific environment to run",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices="train eval evaluate sweep sweep-carbs autotune profile".split(),
    )
    parser.add_argument("--eval-model-path", type=str, default=None)
    parser.add_argument(
        "--baseline", action="store_true", help="Pretrained baseline where available"
    )
    parser.add_argument(
        "--vec",
        "--vector",
        "--vectorization",
        type=str,
        default="serial",
        choices=["serial", "multiprocessing"],
    )
    parser.add_argument(
        "--exp-id", "--exp-name", type=str, default=None, help="Resume from experiment"
    )
    parser.add_argument("--wandb-project", type=str, default="myosuite")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
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
