from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import torch
from modAL.disagreement import consensus_entropy_sampling, max_disagreement_sampling, vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skorch import NeuralNetClassifier
from torch import nn

from activelearning.queries.bayesian.mc_bald import mc_bald
from activelearning.queries.bayesian.mc_max_entropy import mc_max_entropy
from activelearning.queries.bayesian.mc_max_meanstd import mc_max_meanstd
from activelearning.queries.bayesian.mc_max_varratios import mc_max_varratios
from activelearning.queries.informative.entropy_query_stream import stream_query_entropy
from activelearning.queries.informative.margin_query_stream import stream_query_margin
from activelearning.queries.informative.uncertainty_query_stream import stream_query_uncertainty
from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.representative.coreset_query_stream import stream_query_coreset
from activelearning.queries.representative.density_query import query_density
from activelearning.queries.representative.diversity_query import query_diversity
from activelearning.queries.representative.diversity_query_stream import stream_query_diversity
from activelearning.queries.representative.kmeans_query import query_kmeans, query_kmeans_foreach
from activelearning.queries.representative.probcover_query import estimate_delta, query_probcover
from activelearning.queries.representative.probcover_query_stream import stream_query_probcover
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.random_query_stream import stream_query_random
from activelearning.utils.skorch_nnet import (
    DropoutNeuralNetClassifier,
    SimpleNeuralNetClassifier,
    reshapedVGG,
)


class ValidClassifiers(Enum):
    """Valid classifiers for cycle_AL."""

    randomforest: str = "randomforest"
    nnet: str = "nnet"
    kneighbors: str = "kneighbors"
    nnet_bo: str = "nnet_bo"
    nnet_dropout: str = "nnet_dropout"


def set_learner(
    classifier: str,
    query_strategy: Callable,
    X_train: np.ndarray | None,
    y_train: np.ndarray,
    input_dim: int,
    output_dim: int,
    max_epochs: int,
    device: str,
) -> ActiveLearner:
    """Instantiates Active Learner class from modAL when classifier is a valid string.

    Args:
        classifier: one of the valid strings indicating a default classifier
        query_strategy: one of the valid query strategies from modAL or this repo
        X_train: starting training data; can be None
        y_train: strating data labels; can be None
        input_dim: input size for neural networks
        output_dim: output size for neural networks
        max_epochs: max epochs of training for neural networks
        device: cpu or cuda for neural networks

    Returns:
        learner (ActiveLearner): learner class from modAL
    """
    valid_classifiers = [c.value for c in ValidClassifiers]
    if classifier not in valid_classifiers:
        raise ValueError("Invalid value for classifier")

    if classifier == "randomforest":
        learner = ActiveLearner(
            estimator=RandomForestClassifier(), query_strategy=query_strategy, X_training=X_train, y_training=y_train
        )
    elif classifier == "nnet":
        learner = ActiveLearner(
            estimator=NeuralNetClassifier(
                SimpleNeuralNetClassifier(input_dim=input_dim, output_dim=output_dim),
                max_epochs=max_epochs,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                train_split=None,
                verbose=0,
                device=device,
            ),
            query_strategy=query_strategy,
            X_training=X_train,
            y_training=y_train,
        )
    elif classifier == "nnet_dropout":
        learner = ActiveLearner(
            estimator=NeuralNetClassifier(
                DropoutNeuralNetClassifier(input_dim=input_dim, output_dim=output_dim),
                max_epochs=max_epochs,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                train_split=None,
                verbose=0,
                device=device,
            ),
            query_strategy=query_strategy,
            X_training=X_train,
            y_training=y_train,
        )
    elif classifier == "nnet_bo":
        learner = ActiveLearner(
            estimator=NeuralNetClassifier(
                reshapedVGG(num_classes=output_dim),
                max_epochs=max_epochs,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                train_split=None,
                verbose=0,
                device=device,
            ),
            query_strategy=query_strategy,
            X_training=X_train,
            y_training=y_train,
        )
    elif classifier == "kneighbors":
        learner = ActiveLearner(
            estimator=KNeighborsClassifier(n_neighbors=5),
            query_strategy=query_strategy,
            X_training=X_train,
            y_training=y_train,
        )

    return learner


def set_committee(
    committee_classifiers: list,
    query_strategy: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    output_dim: int,
    max_epochs: int,
    device: str,
) -> list[Any]:
    """Sets list of committee classifiers.

    Args:
        committee_classifiers: list containing strings (ValidClassifiers) or sklearn classifiers
        query_strategy: one of the valid query strategies from modAL or this repo
        X_train: starting training data; can be None
        y_train: strating data labels; can be None
        input_dim: input size for neural networks
        output_dim: output size for neural networks
        max_epochs: max epochs of training for neural networks
        device: cpu or cuda for neural networks

    Returns:
        committee_list: list of sklearn classifiers
    """
    committee_list = []
    valid_classifiers = [c.value for c in ValidClassifiers]

    for model in committee_classifiers:
        if model in valid_classifiers:
            cmt_learner = set_learner(
                model, uncertainty_sampling, X_train, y_train, input_dim, output_dim, max_epochs, device
            )
        else:
            cmt_learner = ActiveLearner(estimator=model, X_training=X_train, y_training=y_train)
        committee_list.append(cmt_learner)

    committee = Committee(learner_list=committee_list, query_strategy=query_strategy)

    return committee


def set_query_args(
    classifier: Any,
    query_strategy: Callable,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_start: np.ndarray | None,
    n_instances: int,
    K: int,
    metric: str,
    quantile: float,
    start_len: int,
) -> dict:
    """Sets the query_args dictionary containing parameters for the selected query strategy.

    Args:
        classifier (sklearn classifier): estimator of the ActiveLearner
        query_strategy: one of the valid query strategies from modAL or this repo
        X_pool: pool of unlabeled data
        y_pool: true labels of the unlabeled data
        X_start: labeled data
        n_instances: number of points queried at each AL iteration
        K: number of clusters for k-means query
        metric: metric for density query
        quantile: parameter for stream based query strategies
        start_len: number of initial samples to keep in stream based sampling

    Returns:
        query_args (dict): dictionary of query_parameters
    """
    if query_strategy == query_kmeans_foreach:
        query_args = {"X_pool": X_pool, "K": K, "n_instances": n_instances}
    elif query_strategy == query_density:
        query_args = {"X_pool": X_pool, "n_instances": n_instances, "metric": metric}
    elif query_strategy == query_diversity:
        query_args = {"X_pool": X_pool, "X_start": X_start, "n_instances": n_instances}
    elif query_strategy == query_random:
        query_args = {"X_pool": X_pool, "n_instances": n_instances}
    elif query_strategy == query_coreset:
        query_args = {"X_pool": X_pool, "X_start": X_start, "n_instances": n_instances}
    elif query_strategy == query_probcover:
        delta = estimate_delta(X_pool, K=len(np.unique(y_pool)))  # estimate delta for probcover
        query_args = {"X_pool": X_pool, "X_start": X_start, "n_instances": n_instances, "delta": delta}
    elif query_strategy in (stream_query_diversity, stream_query_coreset):
        query_args = {
            "X_pool": X_pool,
            "X_start": X_start,
            "n_instances": n_instances,
            "quantile": quantile,
            "start_len": start_len,
        }
    elif query_strategy == stream_query_probcover:
        n_classes = len(np.unique(y_pool))
        query_args = {
            "X_pool": X_pool,
            "X_start": X_start,
            "n_instances": n_instances,
            "n_classes": n_classes,
            "start_len": start_len,
        }
    elif query_strategy == stream_query_random:
        query_args = {"X_pool": X_pool, "n_instances": n_instances, "perc": quantile}
    # information based queries
    elif query_strategy in (uncertainty_sampling, margin_sampling, entropy_sampling):
        query_args = {"X_pool": X_pool, "n_instances": n_instances}
    elif query_strategy in (stream_query_uncertainty, stream_query_margin, stream_query_entropy):
        query_args = {
            "classifier": classifier,
            "X_pool": X_pool,
            "X_start": X_start,
            "n_instances": n_instances,
            "threshold": quantile,
            "start_len": start_len,
        }
    # committee queries
    elif (
        query_strategy in (vote_entropy_sampling, consensus_entropy_sampling)
        or (query_strategy in (max_disagreement_sampling, mc_bald))
        or (query_strategy in (mc_max_entropy, mc_max_varratios, mc_max_meanstd))
    ):
        query_args = {"X_pool": X_pool, "n_instances": n_instances}

    return query_args


def score_accuracy(
    learner, acc: str, X_pool: np.ndarray, y_pool: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Scores the accuracy of the classifier.

    Args:
        learner (ActiveLearner): learner from modAL class to be scored
        acc: whether to score the learner on test set, remaining pool or both
        X_pool: unlabeled data pool
        y_pool: labels of pool
        X_test: test set
        y_test: labels of the test set

    Returns:
        accuracy (float): accuracy score of the learner's classifier
    """
    if acc not in ["joint", "pool", "test"]:
        raise ValueError("acc parameter is not valid. Should be one of 'joint', 'pool' or 'test'")

    if acc == "joint":
        accuracy_pool = learner.score(X_pool, y_pool)
        accuracy_test = learner.score(X_test, y_test)
        accuracy_joint = (accuracy_pool * X_pool.shape[0] + accuracy_test * X_test.shape[0]) / (
            X_pool.shape[0] + X_test.shape[0]
        )
        accuracy = accuracy_joint
    elif acc == "test":
        accuracy_test = learner.score(X_test, y_test)
        accuracy = accuracy_test
    elif acc == "pool":
        if X_pool.shape[0] == 0:
            accuracy_pool = 0
        else:
            accuracy_pool = learner.score(X_pool, y_pool)
        accuracy = accuracy_pool

    return accuracy


def set_query_names(query_strategies: list) -> list[str]:
    """Sets string names for the query strategies.

    Args:
        query_strategies: list of query strategies

    Returns:
        col_names (list): list of query strategies' names
    """
    col_names = []
    for q in query_strategies:
        if q == query_kmeans_foreach:
            col_names.append("kmeans")
        elif q == query_kmeans:
            col_names.append("kmeans naive")
        elif q == query_density:
            col_names.append("density")
        elif q == query_diversity:
            col_names.append("diversity")
        elif q == query_coreset:
            col_names.append("coreset")
        elif q == query_probcover:
            col_names.append("probcover")
        elif q == query_random:
            col_names.append("random")
        # stream representation
        elif q == stream_query_diversity:
            col_names.append("streamdiv")
        elif q == stream_query_coreset:
            col_names.append("streamcore")
        elif q == stream_query_probcover:
            col_names.append("streamprob")
        elif q == stream_query_random:
            col_names.append("streamrandom")
        # information based
        elif q == uncertainty_sampling:
            col_names.append("uncertainty")
        elif q == margin_sampling:
            col_names.append("margin")
        elif q == entropy_sampling:
            col_names.append("entropy")
        # stream uncertainty
        elif q == stream_query_uncertainty:
            col_names.append("streamunc")
        elif q == stream_query_margin:
            col_names.append("streammargin")
        elif q in (stream_query_entropy, stream_query_entropy):
            col_names.append("streamentropy")
        # committee
        elif q == vote_entropy_sampling:
            col_names.append("voteentropy")
        elif q == consensus_entropy_sampling:
            col_names.append("consensus")
        elif q == max_disagreement_sampling:
            col_names.append("disagreement")
        # bayesian optimization (MC dropout)
        elif q == mc_bald:
            col_names.append("BALD")
        elif q == mc_max_entropy:
            col_names.append("MC max entropy")
        elif q == mc_max_varratios:
            col_names.append("MC max VarRatios")
        elif q == mc_max_meanstd:
            col_names.append("MC max meanstd")
        # not implemented
        else:
            raise ValueError("Received a not valid query strategy")

    return col_names
