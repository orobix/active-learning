import logging
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from skorch import NeuralNetClassifier
from torch import nn

from activelearning.queries.informative.entropy_query_stream import stream_query_entropy
from activelearning.queries.informative.margin_query_stream import stream_query_margin
from activelearning.queries.informative.uncertainty_query_stream import stream_query_uncertainty
from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.representative.coreset_query_stream import stream_query_coreset
from activelearning.queries.representative.diversity_query_stream import stream_query_diversity
from activelearning.queries.representative.probcover_query_stream import stream_query_probcover
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.random_query_stream import stream_query_random
from activelearning.utils.AL_utils import (
    ValidClassifiers,
    score_accuracy,
    set_committee,
    set_learner,
    set_query_args,
    set_query_names,
)
from activelearning.utils.skorch_nnet import (
    reshapedVGG,
)


def cycle_AL(
    X_train: np.ndarray | None,
    y_train: np.ndarray | None,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: Any,
    query_strategy: Callable,
    committee_classifiers: list | None = None,
    n_instances: int = 1,
    batch_size: int = 0,
    start_len: int = 50,
    K: int = 3,
    metric: str = "euclidean",
    input_dim: int = 512,
    output_dim: int = 10,
    max_epochs: int = 50,
    quantile: float = 0.5,
    goal_acc: float = 0.90,
    acc: str = "test",
) -> tuple[list, list]:
    """Executes Active Learning on provided data with the specified strategy and parameters.

    Note:
        If batch_size is specified, then it become stream based batch sampling.
        At every iteration, saves a batch of points, performs the query strategy on this batch, then discards the
        points that are not chosen.

    Inputs:
        X_train: initial labeled data
        y_train: labels for X_train
            both can be "None" if starting without labeled data
        X_pool: initial unlabeled data
        y_pool: labels for X_pool
        X_test: test data
        y_test_ lebels for test data
        classifier: model type used for classification
            supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
        committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
        query_strategy: one the implemented query strategies imported from the queries folder.
        n_instances: number of instances to be queried at each iteration
        batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
        start_len: for stream based sampling, number of points to be kept as initial set

        K: number of clusters, only if query strategy is k-means
        metric: density metric, only if query strategy is density
        quantile: quantile of distances for stream based diversity query threshold

        input_dim: size of inputs, in case a nnet classifier is passed
        output_dim: number of classes, in case a nnet classifier is passed
        max_epochs: max number of epochs for the training of a nnet at every iteration
        goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
        acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

    Outputs:
        acc_list: list of accuracy scores, one for each iteration
        inst_list: keeps track of how many instances have been sampled after each iteration
    """
    if committee_classifiers is None:
        committee_classifiers = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check if is stream based batch sampling or not
    if batch_size == 0:
        is_stream = False
    else:
        is_stream = True

    # declare learner as ActiveLearner class
    valid_classifiers = [c.value for c in ValidClassifiers]
    if classifier in valid_classifiers:
        learner = set_learner(classifier, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device)
    else:
        # in this case the classifier is assumed to be a non-default sklearn classifier
        # if the model is wrongly specified, activelearner will give an error
        learner = ActiveLearner(
            estimator=classifier, query_strategy=query_strategy, X_training=X_train, y_training=y_train
        )

    if committee_classifiers != []:
        committee = set_committee(
            committee_classifiers, query_strategy, X_train, y_train, input_dim, output_dim, max_epochs, device
        )

    if is_stream is True:
        # initialize first batch
        X_pool_full, y_pool_full = X_pool, y_pool
        if X_pool_full.shape[0] < batch_size:
            batch_size = X_pool_full.shape[0]
        batch_idxs = np.random.choice(range(X_pool_full.shape[0]), size=batch_size, replace=False)
        X_pool, y_pool = X_pool_full[batch_idxs], y_pool[batch_idxs]
        X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
        y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

    # initialize query args dict if empty
    query_args = set_query_args(
        learner.estimator,
        query_strategy,
        X_pool,
        y_pool,
        learner.X_training,
        n_instances,
        K,
        metric,
        quantile,
        start_len,
    )

    n_instances = query_args["n_instances"]
    if X_train is None:
        instances_sampled = 0
    else:
        instances_sampled = X_train.shape[0]

    # init acc_list and inst_list, and calculate starting values if there's a starting set
    accuracy = 0
    inst_list = []
    acc_list = []
    if X_train is not None:
        accuracy = score_accuracy(learner, acc, X_pool, y_pool, X_test, y_test)
        acc_list.append(accuracy)
        inst_list.append(instances_sampled)

    # iterates until goal accuracy is reached or all pool is sampled
    while accuracy < goal_acc and X_pool.shape[0] > 0:
        if X_pool.shape[0] - n_instances < 0:
            # if there's less instances left, sample all of them
            n_instances = X_pool.shape[0]
            query_args["n_instances"] = n_instances

        stream_queries = [
            stream_query_diversity,
            stream_query_coreset,
            stream_query_probcover,
            stream_query_random,
            stream_query_uncertainty,
            stream_query_margin,
            stream_query_entropy,
        ]
        if query_strategy in stream_queries:  # stream based query
            query_idx, _, kept = query_strategy(**query_args)
        elif committee_classifiers != []:  # committee based query
            query_idx, _ = committee.query(**query_args)
            kept = True
        else:  # pool based query
            query_idx, _ = learner.query(**query_args)
            kept = True

        if kept is True:
            if n_instances == 1:
                shape = n_instances
            else:
                shape = len(query_idx)
            if classifier == "nnet_bo":  # re-instantiate the classifier
                # just initializing doesn't work
                learner.estimator = NeuralNetClassifier(
                    reshapedVGG(num_classes=output_dim),
                    max_epochs=max_epochs,
                    criterion=nn.CrossEntropyLoss,
                    optimizer=torch.optim.Adam,
                    train_split=None,
                    verbose=0,
                    device=device,
                )
            learner.teach(
                X_pool[query_idx].reshape(shape, -1),
                y_pool[query_idx].reshape(
                    shape,
                ),
                only_new=False,
            )  # appends instances to labeled set
            if committee_classifiers != []:
                committee.teach(
                    X_pool[query_idx].reshape(shape, -1),
                    y_pool[query_idx].reshape(
                        shape,
                    ),
                )

        if is_stream is False:
            # remove queried instance from the pool
            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)
        else:  # stream based, prepare next batch
            if X_pool_full.shape[0] < batch_size:
                batch_size = X_pool_full.shape[0]
            batch_idxs = np.random.choice(X_pool_full.shape[0], size=batch_size, replace=False)

            X_pool, y_pool = X_pool_full[batch_idxs], y_pool_full[batch_idxs]
            X_pool_full = np.delete(X_pool_full, batch_idxs, axis=0)
            y_pool_full = np.delete(y_pool_full, batch_idxs, axis=0)

        # update input dictionary for the query function
        query_args["X_pool"] = X_pool
        # update other parameters of query strategy
        if "X_start" in query_args:
            query_args["X_start"] = learner.X_training
        if "classifier" in query_args:
            query_args["classifier"] = learner.estimator

        if kept is True:
            accuracy = score_accuracy(learner, acc, X_pool, y_pool, X_test, y_test)
            acc_list.append(accuracy)
            instances_sampled += n_instances
            inst_list.append(instances_sampled)

    return acc_list, inst_list


def strategy_comparison(
    X_train: np.ndarray | None,
    y_train: np.ndarray | None,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: Any,
    query_strategies: list | None = None,
    committee_classifiers: list | None = None,
    n_instances: list | None = None,
    batch_size: int = 0,
    start_len: int = 50,
    K: int = 3,
    metric: str = "euclidean",
    quantile: float = 0.5,
    input_dim: int = 512,
    output_dim: int = 10,
    max_epochs: int = 50,
    goal_acc: float = 0.90,
    acc: str = "test",
) -> list[pd.DataFrame]:
    """Executes Active Learning on provided data with many strategies and parameters, and returns a data frame
    for comparing results.

    Inputs:
        X_train: initial labeled data
        y_train: labels for X_train
            both can be "None" if starting without labeled data
        X_pool: initial unlabeled data
        y_pool: labels for X_pool
        X_test: test data
        y_test_ lebels for test data
        classifier: model type used for classification
            supports "randomforest", "nnet" and the other values in ValidClassifiers, or an already instantiated model
        query_strategies: list containing the query strategies to try, from the ones implemented in the
            queries folder. Default is None, which compares three standard strategies (coreset, uncertainty and random)
        committee_classifiers: list of classifiers to be used for the committee. If None, no committee is used
        n_instances: list of parameters for the number of instances to be queried at each iteration.
            If longer than one, all queries are tried with each parameter.
        batch_size: if left at 0 is ignored, otherwise the size of the batch for stream based batch sampling
        start_len: for stream based sampling, number of points to be accepted as initial set

        K: number of clusters, only if one of the query strategies is k-means
        metric: density metric, only if one of the query strategies is density
        quantile: quantile of distances for stream based diversity query threshold
        input_dim: size of inputs, in case a nnet classifier is passed
        output_dim: number of classes, in case a nnet classifier is passed
        max_epochs: max number of epochs for the training of a nnet at every iteration

        goal_acc: querying continues until this accuracy is reached, or all the pool has been sampled
        acc: whether to asses the classification accuracy on test set, remaining pool, or joint.

    Outputs:
        scores_list: list of data frames containing the results, accuracy and number of instances for each
            query strategy. Number of dataframes corresponds to number of parameters to try specified
            in n_instances
    """
    # set default values for None-arguments
    if query_strategies is None:
        query_strategies = [query_coreset, uncertainty_sampling, query_random]
    if committee_classifiers is None:
        committee_classifiers = []
    if n_instances is None:
        n_instances = [1]

    # column names for output data frame
    col_names = set_query_names(query_strategies)
    # duplicate columns: one for accuracy, one for instances
    col_names = [item for item in col_names for _ in range(2)] * len(n_instances)

    i = 1  # counter combinations
    j = 0  # counter columns

    scores_list = []

    for n_inst in n_instances:
        scores = pd.DataFrame()
        for query in query_strategies:
            logging.info(f"---computing combination {i}")
            accs, insts = cycle_AL(
                X_train=X_train,
                y_train=y_train,
                X_pool=X_pool,
                y_pool=y_pool,
                X_test=X_test,
                y_test=y_test,
                classifier=classifier,
                query_strategy=query,
                committee_classifiers=committee_classifiers,
                n_instances=n_inst,
                batch_size=batch_size,
                start_len=start_len,
                K=K,
                metric=metric,
                quantile=quantile,
                input_dim=input_dim,
                output_dim=output_dim,
                max_epochs=max_epochs,
                goal_acc=goal_acc,
                acc=acc,
            )

            col_names[j + 1] += " inst"

            df_temp = pd.DataFrame({col_names[j]: accs, col_names[j + 1]: insts})
            scores = pd.concat([scores, df_temp], axis=1)

            i += 1
            j += 2
        scores_list.append(scores)
    logging.info("Done.")

    return scores_list


def plot_results(
    scores_list: list, n_instances: list, tot_samples: int, figsize: tuple = (13, 8), goal_acc: int = 0.9
) -> None:
    """Plots scores from strategy_comparison.

    Inputs:
        scores_list: output from strategy_comparison. List of dataframes containing scores
        n_instances: list of parameters used for n_instances. Determines number of plots, one for each parameter
        tot_samples: number of samples in original data, for scale in the graph
        figsize: optional, tuple determining graph size
        goal_acc: accuracy of the full data classifier, displays a yellow line

    Outputs:
        One plot for each parameter choice of n_instances. X axis is number of instances sampled, Y axis is accuracy.
        One line on the plot for each query strategy
    """
    ngraphs = len(scores_list)
    fig, axarr = plt.subplots(ngraphs, 1, figsize=figsize)

    for i in range(ngraphs):
        scores = scores_list[i]
        row = i
        for j in range(0, scores.shape[1], 2):
            if ngraphs == 1:
                plot_j = axarr
            else:
                plot_j = axarr[row]
            plot_j.plot(scores.iloc[:, j + 1], scores.iloc[:, j], label=scores.columns[j])
            plot_j.legend()
            plot_j.grid(True)
            plot_j.set_title(f"instances per iter: {n_instances[i]}", fontsize=10)
            x_ticks = range(0, tot_samples, n_instances[i])
            plot_j.set_xticks(x_ticks)
            plot_j.axhline(y=goal_acc, color="y", linestyle="--")  # goal acc
            plot_j.set_xlabel("instances")
            plot_j.set_ylabel("accuracy")

    fig.tight_layout()
    fig.suptitle(f"Accuracy of different strategies until {goal_acc} accuracy is reached", y=1.02)
