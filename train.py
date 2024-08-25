import signal
import uuid
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich.traceback import install

import clean_pufferl
import environment
import policy

from utils import parse_args, init_wandb

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


### CARBS Sweeps
def sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls):
    from math import log, ceil, floor
    import numpy as np

    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam

    # from carbs import ParamDictType
    from carbs import Param

    def closest_power(x):
        possible_results = floor(log(x, 2)), ceil(log(x, 2))
        return int(2 ** min(possible_results, key=lambda z: abs(x - 2**z)))

    def closest_multiple(x, base):
        return base * ceil(x / base)

    def carbs_param(
        group,
        name,
        space,
        wandb_params,
        mmin=None,
        mmax=None,
        search_center=None,
        is_integer=False,
        rounding_factor=1,
    ):
        wandb_param = wandb_params[group]["parameters"][name]
        if "values" in wandb_param:
            values = wandb_param["values"]
            mmin = min(values)
            mmax = max(values)

        if mmin is None:
            mmin = float(wandb_param["min"])
        if mmax is None:
            mmax = float(wandb_param["max"])
        if search_center is None:
            search_center = (
                int(wandb_param["search_center"])
                if is_integer
                else float(wandb_param["search_center"])
            )

        if space == "log":
            Space = LogSpace
        elif space == "linear":
            Space = LinearSpace
        elif space == "logit":
            Space = LogitSpace
            assert mmin == 0
            assert mmax == 1
            assert search_center is not None
        else:
            raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

        return Param(
            name=f"{group}-{name}",
            space=Space(min=mmin, max=mmax, is_integer=is_integer, rounding_factor=rounding_factor),
            search_center=search_center,
        )

    if not os.path.exists("checkpoints"):
        os.system("mkdir checkpoints")

    target_metric = args["sweep"]["metric"]["name"].split("/")[-1]
    sweep_param = args["sweep"]["parameters"]
    seeds = args["carbs"]["search_center"]
    # wandb_env_params = sweep_param['env']['parameters']
    # wandb_policy_params = sweep_param['policy']['parameters']

    # total_timesteps: Must be hardcoded and match wandb sweep space for now
    param_spaces = []
    if "total_timesteps" in sweep_param["train"]["parameters"]:
        time_param = sweep_param["train"]["parameters"]["total_timesteps"]
        min_timesteps = time_param["min"]
        param_spaces.append(
            carbs_param(
                "train",
                "total_timesteps",
                "log",
                sweep_param,
                search_center=min_timesteps,
                is_integer=True,
            )
        )

    # TODO: lines are too many, refactor
    # params seem to be too many -- hold num_envs, max_grad_norm
    param_spaces += [
        # carbs_param(
        #     "train",
        #     "num_envs",
        #     "linear",
        #     sweep_param,
        #     is_integer=True,
        #     search_center=seeds["num_envs"],
        # ),
        carbs_param(
            "train", "learning_rate", "log", sweep_param, search_center=seeds["learning_rate"]
        ),
        carbs_param("train", "gamma", "logit", sweep_param, search_center=seeds["gamma"]),
        carbs_param("train", "gae_lambda", "logit", sweep_param, search_center=seeds["gae_lambda"]),
        carbs_param(
            "train",
            "update_epochs",
            "log",
            sweep_param,
            is_integer=True,
            search_center=seeds["update_epochs"],
        ),
        carbs_param("train", "clip_coef", "logit", sweep_param, search_center=seeds["clip_coef"]),
        carbs_param("train", "vf_coef", "linear", sweep_param, search_center=seeds["vf_coef"]),
        carbs_param(
            "train", "vf_clip_coef", "logit", sweep_param, search_center=seeds["vf_clip_coef"]
        ),
        # carbs_param(
        #     "train", "max_grad_norm", "linear", sweep_param, search_center=seeds["max_grad_norm"]
        # ),
        carbs_param("train", "ent_coef", "log", sweep_param, search_center=seeds["ent_coef"]),
        carbs_param(
            "train",
            "batch_size",
            "log",
            sweep_param,
            is_integer=True,
            search_center=seeds["batch_size"],
        ),
        carbs_param(
            "train",
            "minibatch_size",
            "log",
            sweep_param,
            is_integer=True,
            search_center=seeds["minibatch_size"],
        ),
        carbs_param(
            "train",
            "bptt_horizon",
            "log",
            sweep_param,
            is_integer=True,
            search_center=seeds["bptt_horizon"],
        ),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
        num_random_samples=2,  # random sampling doesn't seem to work
        # NOTE: play with these to vary the training steps
        min_pareto_cost_fraction=0.3,
        max_suggestion_cost=600,  # Shoot for 10 mins
    )
    carbs = CARBS(carbs_params, param_spaces)

    def run_sweep_session():
        print("--------------------------------------------------------------------------------")
        print("Starting a new session...")
        print("--------------------------------------------------------------------------------")
        wandb = init_wandb(args, env_name, id=args["exp_id"])
        wandb.config.__dict__["_locked"] = {}

        print("Getting suggestion...")
        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print("CARBS suggestion:", suggestion)
        train_suggestion = {
            k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("train-")
        }
        # Correcting critical parameters before updating
        for key in ["batch_size", "minibatch_size", "bptt_horizon"]:
            train_suggestion[key] = closest_power(train_suggestion[key])
        # args["train"]["num_envs"] = closest_power(train_suggestion["num_envs"])  # 16, 32, 64
        args["train"].update(train_suggestion)

        env_suggestion = {k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("env-")}
        args["env"].update(env_suggestion)

        args["track"] = True
        wandb.config.update({"train": args["train"]}, allow_val_change=True)

        print("Train config:", wandb.config.train)
        # print(wandb.config.env)
        # print(wandb.config.policy)
        stats, uptime, is_success = {}, 0, False
        try:
            stats, uptime = train(args, make_env, policy_cls, rnn_cls, wandb, skip_dash=True)
            is_success = len(stats) > 0
        except Exception as e:  # noqa
            import traceback

            traceback.print_exc()

        # NOTE: What happens if training fails?
        """
        A run should be reported as a failure if the hyperparameters suggested by CARBS 
        caused the failure, for example a batch size that is too large that caused an OOM failure. 
        If a failure occurs that is not related to the hyperparameters, it is better to forget 
        the suggestion or retry it. Report a failure by making an ObservationInParam with is_failure=True
        """
        observed_value = [s[target_metric] for s in stats if target_metric in s]
        if len(observed_value) > 0:
            observed_value = np.mean(observed_value)
        else:
            observed_value = 0

        print(f"\n\nTrain success: {is_success}, Observed value: {observed_value}\n\n")
        obs_out = carbs.observe(  # noqa
            ObservationInParam(
                input=orig_suggestion,
                output=observed_value,
                cost=uptime,
                is_failure=not is_success,
            )
        )

    # For debugging
    # run_sweep_session()

    # Run sweep
    import wandb

    sweep_id = wandb.sweep(
        sweep=args["sweep"],
        project="carbs",
    )
    wandb.agent(sweep_id, run_sweep_session, count=args["train"]["num_sweeps"])


def train(args, env_creator, policy_cls, rnn_cls, wandb=None, skip_dash=False):
    if args["vec"] == "serial":
        vec = pufferlib.vector.Serial
    elif args["vec"] == "multiprocessing":
        vec = pufferlib.vector.Multiprocessing
    else:
        raise ValueError("Invalid --vector (serial/multiprocessing).")

    env_args = None
    env_kwargs = args["env"]
    if args["capture_video"] is True:
        assert isinstance(env_creator, list), "Video capture requires the env_creator to be a list"
        num_envs = args["train"]["num_envs"]
        assert len(env_creator) == num_envs
        env_args = [[]] * num_envs
        env_kwargs = [args["env"]] * num_envs

    vecenv = pufferlib.vector.make(
        env_creator,
        env_args=env_args,
        env_kwargs=env_kwargs,
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

            # Or, the time budget is up
            if data.profile.uptime > train_config.train_time_budget:
                break

            clean_pufferl.train(data)

        uptime = data.profile.uptime

        # Run evaluation to get the average stats
        stats = []
        num_eval_epochs = train_config.eval_timesteps // train_config.batch_size
        for _ in range(1 + num_eval_epochs):  # extra data for sweeps
            stats.append(clean_pufferl.evaluate(data)[0])

    except Exception as e:  # noqa
        uptime, stats = 0, []
        import traceback

        traceback.print_exc()

    clean_pufferl.close(data)
    return stats, uptime


if __name__ == "__main__":
    args, env_name, run_name = parse_args()
    run_name = "pufferl_" + run_name

    # Load env binding and policy
    env_creator = environment.pufferl_env_creator(env_name, run_name, args)
    policy_cls = getattr(policy, args["base"]["policy_name"])
    rnn_cls = None
    if "rnn_name" in args["base"]:
        rnn_cls = getattr(policy, args["base"]["rnn_name"])

    # Process mode
    if args["mode"] == "train":
        wandb = None
        if args["track"]:
            wandb = init_wandb(args, run_name)
        train(args, env_creator, policy_cls, rnn_cls, wandb=wandb)

    elif args["mode"] in ("eval", "evaluate"):
        clean_pufferl.rollout(
            env_creator,
            args["env"],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            model_path=args["eval_model_path"],
            device=args["train"]["device"],
        )

    elif args["mode"] == "sweep-carbs":
        args["capture_video"] = False
        env_creator = environment.pufferl_env_creator(env_name, run_name, args)
        sweep_carbs(args, env_name, env_creator, policy_cls, rnn_cls)

    elif args["mode"] == "autotune":
        pufferlib.vector.autotune(env_creator, batch_size=args["train"]["env_batch_size"])

    elif args["mode"] == "profile":
        import cProfile

        cProfile.run("train(args, env_creator, policy_cls, rnn_cls)", "stats.profile")
        import pstats
        from pstats import SortKey

        p = pstats.Stats("stats.profile")
        p.sort_stats(SortKey.TIME).print_stats(10)
