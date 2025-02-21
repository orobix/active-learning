import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def stream_query_coreset(
    X_pool: np.ndarray, X_start: np.ndarray | None, start_len: int = 50, quantile: float = 0.5, n_instances: int = 1
) -> tuple[list, np.ndarray, bool]:
    """Chooses whether to query a point from stream based sampling,
    by calculating if it has enough distance from the labeled set.
    Distance threshold is necessary for the decision.

    Note:
        If X_start == None (no labeled data is available), the point is queried.
        Useful for the first iteration of the querying cycle

    Args:
        X_pool: pool of unlabeled instances. One point is randomly sampled to simulate the stream
        X_start: set of labeled instances
        start_len: number of instances to randomly sample at the start, if X_start isn't passed
        quantile: quantile of the distances in the labeled pool to be used as threshold for the decision on the query
        n_instances: necessary argument for integration with cycle_AL, but isn't used. Number of instances is always 1

    Returns:
        List of three:
        The index of the sampled instance, the sampled instance, and whether or not it is to be kept
    """
    # sample stream point from pool
    new_instance_idx = np.random.randint(0, X_pool.shape[0])
    new_instance = X_pool[new_instance_idx]

    # check new_instance shape
    new_instance = new_instance.reshape(1, -1)

    if X_start is None or X_start.shape[0] < start_len:  # need at least 2 points, or can't calculate distances
        return new_instance_idx, new_instance, True
    else:
        # calculate minimum distances of labeled instances from eachother
        labeled_dist = euclidean_distances(X_start, X_start)
        labeled_dist = np.sort(labeled_dist, axis=1)[:, 1]  # index 1 because otherwise minimum is 0
        # calculate threshold as quantile of distances
        dist_threshold = np.quantile(labeled_dist, quantile)

        # calculate min distance of new instance from labeled set
        utility = euclidean_distances(new_instance, X_start).min(axis=1)

        if utility > dist_threshold:
            kept = True
        else:
            kept = False
        return new_instance_idx, new_instance, kept
