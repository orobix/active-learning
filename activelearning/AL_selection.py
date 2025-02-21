import numpy as np

from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.representative.diversity_query import query_diversity
from activelearning.queries.representative.probcover_query import query_probcover
from activelearning.utils.AL_utils import set_query_args


def selection_AL(
    X_train: np.ndarray | None,
    X_pool: np.ndarray,
    query_strategy,
    n_instances: int = 1,
    n_queries: int = 50,
    K: int = 3,
    metric: str = "euclidean",
    og_idx: bool = True,
) -> list[np.ndarray, list | None]:
    """Executes Active Learning for selection of instances, without evaluating a classifier.

    Inputs:
        X_train: initial selected data
            can be "None" if starting without labeled data
        X_pool: initial pool of data
        query_strategy: one the implemented query strategies imported from the queries folder.
        n_instances: number of instances to be queried at each iteration
        n_queries: number of queries to be performed

        K: number of clusters, only if query strategy is k-means
        metric: density metric, only if query strategy is density

        og_idx: bool, whether to store the original indexes of selected instances

    Outputs:
        selected_instances: set of selected instances
        If og_idx is true, also returns the list of original indexes
        If og_idx is false, returns None as second value
    """
    if og_idx is True:
        og_idxs_arr = np.array(range(X_pool.shape[0])).reshape(
            -1,
        )
        selected_og_idxs = np.empty(shape=(0,), dtype=int)
    selected_instances = np.empty(shape=(0, X_pool.shape[1]))  # this avoids problems when there is no start set

    query_args = set_query_args(None, query_strategy, X_pool, None, X_train, n_instances, K, metric, None, None)
    # iterates for n_queries
    for _ in range(n_queries):
        if X_pool.shape[0] - n_instances < 0:
            # if there's less instances left, sample all of them
            n_instances = X_pool.shape[0]
            query_args["n_instances"] = n_instances

        query_idx, _ = query_strategy(None, **query_args)  # selects instances

        if og_idx is True:  # store original index of selected item
            selected_og_idxs = np.concatenate(
                [
                    selected_og_idxs,
                    np.array(og_idxs_arr[query_idx]).reshape(
                        -1,
                    ),
                ]
            )
            og_idxs_arr = np.delete(og_idxs_arr, query_idx)

        if X_train is None:
            X_train = np.empty(shape=(0, X_pool.shape[1]))

        # append new instances
        X_train = np.vstack([X_train, X_pool[query_idx].reshape(n_instances, -1)])
        selected_instances = np.vstack([selected_instances, X_pool[query_idx].reshape(n_instances, -1)])
        # remove queried instance from the pool
        X_pool = np.delete(X_pool, query_idx, axis=0)

        # update input dictionary for the query function
        query_args["X_pool"] = X_pool
        if query_strategy in (query_coreset, query_diversity, query_probcover):
            query_args["X_start"] = X_train

    if og_idx is False:
        return selected_instances, None
    else:
        return selected_instances, selected_og_idxs.tolist()
