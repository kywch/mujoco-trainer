import sys
import numpy as np
from loguru import logger
from carbs import ObservationInParam
from utils import parse_args, init_carbs

RANDOM_SEED = 42
NUM_SUGGESTIONS = 20
NUM_RANDOM_SAMPLES = 10


def generate_positive_int(np_random, low, high):
    return np_random.integers(low, high, endpoint=True)


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="ERROR")

    # Use the default config to test CARBS
    args, _, _ = parse_args()
    carbs = init_carbs(args, num_random_samples=NUM_RANDOM_SAMPLES)

    rng = np.random.default_rng(RANDOM_SEED)

    for i in range(NUM_SUGGESTIONS):
        suggestion = carbs.suggest().suggestion
        print(suggestion)
        print()

        cost = suggestion["train-total_timesteps"] * 100

        carbs.observe(
            ObservationInParam(
                input=suggestion, output=generate_positive_int(rng, 0, 100), cost=cost
            )
        )

        print("CARBS state:", carbs.get_state_dict())
