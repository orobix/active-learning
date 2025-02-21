from typing import Any

import numpy as np


def query_random(classifier: Any, X_pool: np.ndarray, n_instances: int = 1) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to queried/added to be labeled pool at random.

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        n_instances: number of instances to be selected

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    random_idx = np.random.choice(range(X_pool.shape[0]), size=n_instances, replace=False)
    random_instances = X_pool[random_idx, :]
    return random_idx, random_instances
