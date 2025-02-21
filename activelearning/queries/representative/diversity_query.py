from typing import Any

import numpy as np
from modAL.utils.selection import multi_argmax
from sklearn.metrics.pairwise import cosine_similarity


def diversity_measure(X_pool: np.ndarray, X_start: np.ndarray | None) -> np.ndarray:
    """Calculates the diversity of each instance in the pool set from the instances
    in the labeled set, measured as the sum of the angles between the feature vector
    of the unlabeled and the feature vectors of all the labeled points.
    Angle is calculated with cosine similarity.

    Args:
        X_pool: pool of unlabeled instances
        X_start: set of labeled instances

    Returns:
        A vector with the diversity measure for each instance
    """
    div_matrix = np.clip(cosine_similarity(X_pool, X_start), -1.0, 1.0)
    div_matrix_angles = np.arccos(div_matrix)
    div_matrix_angles = div_matrix_angles.sum(axis=1)  # sum over rows

    # div_matrix is the sum of the angles between the vector and the other vectors calculated by cosine similarity
    return div_matrix_angles


def query_diversity(
    classifier: Any, X_pool: np.ndarray, X_start: np.ndarray, n_instances: int = 1
) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to queried/added to be labeled pool
    via a diversity measure. The selected instances are the most diverse from
    the labeled set.

    Note:
        If X_start == None (no labeled data is available), a random sample of instances is returned.
        Useful for the first iteration of the querying cycle

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        X_start: set of labeled instances
        n_instances: number of instances to be selected

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    if X_start is None:
        random_idx = np.random.choice(range(X_pool.shape[0]), size=n_instances, replace=False)
        random_instances = X_pool[random_idx, :]
        return random_idx, random_instances
    else:
        utility = diversity_measure(X_pool, X_start)
        return multi_argmax(utility, n_instances=n_instances)
