import sys
import numpy as np
from loguru import logger

from train import parse_args
from clean_pufferl import seed_everything
from utils import init_carbs, carbs_runner_fn

RANDOM_SEED = 42
NUM_SUGGESTIONS = 20

# NOTE: When samples >= 3, we get the repeated random samples
NUM_RANDOM_SAMPLES = 2  # 10


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="ERROR")

    # Use the default config to test CARBS
    args, _, _ = parse_args()
    carbs = init_carbs(args, num_random_samples=NUM_RANDOM_SAMPLES)

    rng = np.random.default_rng(RANDOM_SEED)

    def dummy_train_fn(args, wandb):
        # seeding like clean pufferl
        seed_everything(args["train"]["seed"], args["train"]["torch_deterministic"])

        stats = [{"episode_return": rng.integers(1, 100)}]
        cost = args["train"]["total_timesteps"] // 10000

        return stats, cost

    carbs_runner = carbs_runner_fn(
        args, "test", carbs, "test", train_fn=dummy_train_fn, disable_wandb=True
    )

    for i in range(NUM_SUGGESTIONS):
        carbs_runner()
        print("CARBS state:", carbs.get_state_dict(), "\n")

    # # Test the wandb sweep
    # import wandb
    # sweep_id = wandb.sweep(
    #     sweep=args["sweep"],
    #     project="carbs",
    # )

    # # Run sweep
    # wandb.agent(
    #     sweep_id,
    #     carbs_runner_fn(args, "test", carbs, sweep_id, dummy_train_fn),
    #     count=NUM_SUGGESTIONS,
    # )
