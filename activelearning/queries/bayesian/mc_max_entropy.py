from typing import Any

import numpy as np
from modAL.utils.selection import multi_argmax

from activelearning.queries.bayesian.mc_bald import get_mc_preds


def mc_max_entropy(classifier: Any, X_pool: np.ndarray, n_instances: int = 1) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to queried/added to be labeled pool with monte carlo dropout max entropy.

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        n_instances: number of instances to be selected

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    predictions = get_mc_preds(classifier, X_pool)

    mean = np.mean(predictions, axis=0)

    epsilon = 1e-10  # this avoids errors with log(0)
    # entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=1)

    selected_idx, _ = multi_argmax(entropy, n_instances=n_instances)
    return (
        np.array(selected_idx),
        X_pool[np.array(selected_idx),],
    )
