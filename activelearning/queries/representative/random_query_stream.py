import numpy as np


def stream_query_random(X_pool: np.array, perc: float = 0.5, n_instances: int = 1) -> tuple[list, np.ndarray, bool]:
    """Chooses whether to query a point from stream based sampling, at random.

    Args:
        X_pool: pool of unlabeled instances. One point is randomly sampled to simulate the stream
        perc: probability of accepting the value
        n_instances: necessary argument for integration with cycle_AL, but isn't used. Number of instances is always 1

    Returns:
        The index of the sampled instance, the sampled instance, and whether or not it is to be kept
    """
    # sample stream point from pool
    new_instance_idx = np.random.randint(0, X_pool.shape[0])
    new_instance = X_pool[new_instance_idx]

    # check new_instance shape
    new_instance = new_instance.reshape(1, -1)

    rand = np.random.uniform(0, 1)

    if rand >= perc:
        return new_instance_idx, new_instance, True
    else:
        return new_instance_idx, new_instance, False
