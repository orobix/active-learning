import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from activelearning.queries.representative.probcover_query import estimate_delta


def stream_query_probcover(
    X_pool: np.ndarray, X_start: np.ndarray | None, start_len: int = 50, n_classes: int = 2, n_instances: int = 1
) -> tuple[list, np.ndarray, bool]:
    """Chooses whether to query a point from stream based sampling, by calculating if it is already covered.
    Distance threshold is necessary for the decision.

    Note:
        If X_start == None (no labeled data is available), the point is queried.
        Useful for the first iteration of the querying cycle

    Args:
        X_pool: pool of unlabeled instances. One point is randomly sampled to simulate the stream
        X_start: set of labeled instances
        start_len: number of instances to sample randomly at the beginning
        n_classes: number of classes of the data
        n_instances: necessary argument for integration with cycle_AL, but isn't used. Number of instances is always 1

    Returns:
        The index of the sampled instance, the sampled instance, and whether or not it is to be kept
    """
    # sample stream point from pool
    new_instance_idx = np.random.randint(0, X_pool.shape[0])
    new_instance = X_pool[new_instance_idx]

    # check new_instance shape
    new_instance = new_instance.reshape(1, -1)

    if (
        X_start is None or X_start.shape[0] < start_len
    ):  # need at least one point per class, or can't calculate clusters
        return new_instance_idx, new_instance, True
    else:
        # Calculate distance between X_pool (unlabeled) and X_train (labeled)
        distances_to_train = euclidean_distances(X_pool, X_start)

        # estimate delta
        delta = estimate_delta(X_start, K=n_classes, alpha=0.95)

        # check if distacne is farther than delta from all labeled points (uncovered points)
        if (distances_to_train > delta).all():
            is_uncovered = True
        else:
            is_uncovered = False

        if is_uncovered:
            kept = True
        else:
            kept = False
        return new_instance_idx, new_instance, kept
