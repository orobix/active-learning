from typing import Any

import numpy as np
from modAL.density import information_density
from modAL.utils.selection import multi_argmax


def query_density(
    classifier: Any, X_pool: np.ndarray, n_instances: int = 1, metric: str = "euclidean"
) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to be queried/added to the labeled set
    via the information matrix provided by the modAL package.
    The selected instances are the ones with the highest information density.

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        n_instances: number of instances to be selected
        metric: euclidean or cosine. Metric for the calculation of the information density

    Returns:
        List of two:
        The indexes of the selected instances, and the selected instances
    """
    utility = information_density(X_pool, metric)

    idxs, _ = multi_argmax(utility, n_instances=n_instances)
    return idxs, X_pool[idxs]
