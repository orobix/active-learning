import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def query_probcover(
    classifier,
    X_pool: np.ndarray,
    X_start: np.ndarray | None,
    start_len: int = 50,
    n_instances: int = 1,
    delta: float = 0.5,
) -> tuple[list, np.ndarray]:
    """Query function implementing the ProbCover algorithm for query selection.
    Calculates distances of unlabeled points to labeled points,
    and eliminates those that are already covered (too close).
    For the remaining points, it calculates distances between them and chooses to query the point that has the most
    neighbors (neighbors determined as having a distance under a threshold delta).

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        X_start: set with already selected instances
        start_len: number of instances to sample randomly at the beginning
        n_instances: number of instances to be selected
        delta: distance threshold

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    if X_start is None or X_start.shape[0] < start_len:  # random if there is no labeled set
        random_idx = np.random.choice(range(X_pool.shape[0]), size=n_instances, replace=False)
        random_instances = X_pool[random_idx, :]
        return random_idx, random_instances
    else:
        # calculate distances between X_pool and X_train
        distances_to_train = euclidean_distances(X_pool, X_start)
        # mark points in X_pool that are farther than delta from all labeled points (uncovered points)
        uncovered = np.all(distances_to_train > delta, axis=1)
        # filter X_pool to only uncovered points
        X_pool_uncovered = X_pool[uncovered]

        # check if there's still some uncovered point or reduce delta
        while X_pool_uncovered.shape[0] == 0:
            delta = delta / 2  # if no more uncovered samples, reduce delta
            uncovered = np.all(distances_to_train > delta, axis=1)
            X_pool_uncovered = X_pool[uncovered]

        uncovered_indices = np.where(uncovered)[0]  # original indices in X_pool

        # find uncovered points with neighbors in the range delta
        pool_distances = euclidean_distances(X_pool_uncovered, X_pool_uncovered)
        adjacency_matrix = (pool_distances <= delta).astype(int)

        query_idxs = []
        # iteratively select the points with the most neighbors in the range delta
        for _ in range(min(n_instances, len(X_pool_uncovered))):
            n_neighbors = np.sum(adjacency_matrix, axis=1)

            if n_neighbors.size == 0:  # if no neighbors, select at random
                break
            else:
                selected_index = np.argmax(n_neighbors)
                query_idxs.append(uncovered_indices[selected_index])

            # remove the selected point and its neighbors (covered points)
            neighbors = adjacency_matrix[selected_index].astype(bool)

            # delete covered points from adjacency_matrix and uncovered_indices
            adjacency_matrix = np.delete(adjacency_matrix, np.where(neighbors), axis=0)
            adjacency_matrix = np.delete(adjacency_matrix, np.where(neighbors), axis=1)
            uncovered_indices = np.delete(uncovered_indices, np.where(neighbors))

        return query_idxs, X_pool[query_idxs, :]


def is_ball_pure(D: np.ndarray, i: int, delta: float, labels: np.ndarray) -> bool:
    """Check if a ball of radius delta contains only points of the same label (pure ball).

    Args:
        D (np.array): matrix of pairwise distance of all points in X_pool
        i (int): index of point that is center of the ball
        delta (float): radius of the ball
        labels (np.array): labels of the points, used to check purity

    Returns:
        (bool): whether ball is pure (1) or not (0)
    """
    points_in_ball = D[i] <= delta
    labels_in_ball = labels[points_in_ball]

    return len(set(labels_in_ball)) == 1


def estimate_delta(X_pool: np.ndarray, K: int, alpha: float = 0.95) -> float:
    """Estimates delta for probcover query. Delta is the radius size of the balls.
    The estimation is the largest radius so that at least 95% of the balls (one ball for data point) is pure.
    Pure means that it contains only points with the same label.
    Labels are obtained with k-means clustering where K is the number of true classes.

    Args:
        X_pool: unlabeled pool of instances
        K: real number of classes, used to obtain pseudo labels via kmeans clustering
        alpha: percentage of balls that has to be pure to accept delta

    Returns:
        delta: chosen value of delta. if no value obtains the requested purity, the smallest value is returned
    """
    kmeans_mod = KMeans(n_clusters=K)
    kmeans_mod.fit(X_pool)
    labels = kmeans_mod.labels_  # use clusters as pseudolabels

    D = euclidean_distances(X_pool)
    n_points = len(labels)

    delta_max = np.median(D)
    delta_min = np.unique(D)[1]  # second smallest value, as smallest is 0
    delta_step = -(delta_max - delta_min) / 100  # try 100 values
    deltas = np.arange(delta_max, delta_min, delta_step)

    for delta in deltas:
        pure_balls_count = sum(is_ball_pure(D, i, delta, labels) for i in range(n_points))
        pure_balls_percentage = pure_balls_count / n_points

        if pure_balls_percentage >= alpha:
            return delta

    return delta_min  # if purity goal not reached, return min delta
