import os
import random
import numpy as np

RANDOM_STATE = 42

def fix_all_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)