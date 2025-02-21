import numpy as np

from activelearning.queries.representative.diversity_query import diversity_measure


def stream_query_diversity(
    X_pool: np.ndarray, X_start: np.ndarray | None, start_len: int = 50, quantile: float = 0.5, n_instances: int = 1
) -> tuple[list, np.ndarray, bool]:
    """Chooses whether to query a point from stream based sampling,
    by calculating if it's diverse enough from the labeled set.
    Diversity threshold is necessary for the decision,
    and is set as a quantile of distances in labeled set.

    Note:
        If X_start == None (no labeled data is available), the point is queried.
        Useful for the first iteration of the querying cycle

    Args:
        X_pool: pool of unlabeled instances. One point is randomly sampled to simulate the stream
        X_start: set of labeled instances
        start_len: number of instances to sample randomly at the beginning
        quantile: quantile of the distances in the labeled pool to be used as threshold for the decision on the query
        n_instances: necessary argument for integration with cycle_AL, but isn't used. Number of instances is always 1

    Returns:
        The index of the sampled instance, the sampled instance, and whether or not it is to be kept
    """
    # sample stream point from pool
    new_instance_idx = np.random.randint(0, X_pool.shape[0])
    new_instance = X_pool[new_instance_idx]

    # check new_instance shape
    new_instance = new_instance.reshape(1, -1)

    if X_start is None or X_start.shape[0] < start_len:
        return new_instance_idx, new_instance, True
    else:
        # calculate diversity of labeled instances from eachother
        labeled_div = diversity_measure(X_start, X_start)
        # calculate threshold as quantile of distances
        div_threshold = np.quantile(labeled_div, quantile)

        # calculate diversity of new instance from labeled set
        utility = diversity_measure(new_instance, X_start)

        if utility > div_threshold:
            kept = True
        else:
            kept = False
        return new_instance_idx, new_instance, kept
