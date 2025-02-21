import numpy as np
import pytest
import torch
from modAL import ActiveLearner
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
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
from activelearning.queries.representative.kmeans_query import query_kmeans_foreach
from activelearning.queries.representative.probcover_query import query_probcover
from activelearning.queries.representative.probcover_query_stream import stream_query_probcover
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.random_query_stream import stream_query_random
from activelearning.utils.skorch_nnet import reshapedVGG


@pytest.fixture
def pool_based_query_strategy(request):
    """Calls for fixture parameter."""
    return request.param


@pytest.mark.parametrize(
    "pool_based_query_strategy, correct_query_kwargs",
    [
        (
            query_random,
            {"classifier": None, "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), "n_instances": 2},
        ),
        (
            query_coreset,
            {
                "classifier": None,
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "X_start": None,
                "n_instances": 2,
            },
        ),
        (
            query_probcover,
            {
                "classifier": None,
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "X_start": None,
                "n_instances": 2,
            },
        ),
        (
            query_density,
            {"classifier": None, "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), "n_instances": 2},
        ),
        (
            query_diversity,
            {
                "classifier": None,
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "X_start": None,
                "n_instances": 2,
            },
        ),
        (
            query_kmeans_foreach,
            {
                "classifier": None,
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "K": 2,
                "n_instances": 2,
            },
        ),
        (
            uncertainty_sampling,
            {
                "classifier": RandomForestClassifier(),
                "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "n_instances": 2,
            },
        ),
        (
            margin_sampling,
            {
                "classifier": RandomForestClassifier(),
                "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "n_instances": 2,
            },
        ),
        (
            entropy_sampling,
            {
                "classifier": RandomForestClassifier(),
                "X": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "n_instances": 2,
            },
        ),
    ],
    indirect=["pool_based_query_strategy"],
)
def test_pool_based_query(pool_based_query_strategy, correct_query_kwargs):
    """Tests pool based query strategies."""
    query_idx, query_instances = pool_based_query_strategy(**correct_query_kwargs)

    if pool_based_query_strategy not in (uncertainty_sampling, margin_sampling, entropy_sampling):
        X_pool = correct_query_kwargs["X_pool"]
    else:
        X_pool = correct_query_kwargs["X"]
    n_instances = correct_query_kwargs["n_instances"]

    # selected indexes are valid?
    assert all(idx in range(X_pool.shape[0]) for idx in query_idx), "Returned index out of range"

    # number of instances is correct?
    assert len(query_idx) == n_instances, "Incorrect number of indices returned"
    assert query_instances.shape[0] == n_instances, "Incorrect number of instances returned"

    # are selected instances not duplicated?
    assert len(query_idx) == len(set(query_idx)), "Selected indexes are duplicated"

    ## Edge cases

    # works when n_instances = 1?
    correct_query_kwargs["n_instances"] = 1
    query_idx, query_instances = pool_based_query_strategy(**correct_query_kwargs)

    assert len(query_idx) == 1, "Incorrect number of indices returned for n_instances=1"
    assert query_instances.shape[0] == 1, "Incorrect number of instances returned for n_instances=1"

    # works when n_instances equals the pool size?
    correct_query_kwargs["n_instances"] = X_pool.shape[0]
    query_idx, query_instances = pool_based_query_strategy(**correct_query_kwargs)

    assert len(query_idx) == X_pool.shape[0], "Incorrect number of indices returned for full pool"
    assert query_instances.shape[0] == X_pool.shape[0], "Incorrect number of instances returned for full pool"


@pytest.fixture
def stream_based_query_strategy(request):
    """Calls for fixture parameter."""
    return request.param


@pytest.mark.parametrize(
    "stream_based_query_strategy, correct_stream_kwargs",
    [
        (
            stream_query_random,
            {
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            },
        ),
        (stream_query_probcover, {"X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), "X_start": None}),
        (stream_query_coreset, {"X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), "X_start": None}),
        (stream_query_diversity, {"X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), "X_start": None}),
        (
            stream_query_uncertainty,
            {
                "classifier": RandomForestClassifier(),
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "X_start": None,
            },
        ),
        (
            stream_query_margin,
            {
                "classifier": RandomForestClassifier(),
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "X_start": None,
            },
        ),
        (
            stream_query_entropy,
            {
                "classifier": RandomForestClassifier(),
                "X_pool": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                "X_start": None,
            },
        ),
    ],
    indirect=["stream_based_query_strategy"],
)
def test_stream_based_query(stream_based_query_strategy, correct_stream_kwargs):
    """Tests pool based query strategies."""
    query_idx, query_instances, kept = stream_based_query_strategy(**correct_stream_kwargs)

    X_pool = correct_stream_kwargs["X_pool"]

    # correct data types?
    assert isinstance(query_idx, int) and isinstance(query_instances, np.ndarray) and isinstance(kept, bool)

    # selected index is valid?
    assert query_idx in range(X_pool.shape[0]), "Returned index out of range"

    # number of instances is correct?
    assert query_instances.shape[0] == 1, "Incorrect number of instances returned"


@pytest.fixture
def bayesian_query_strategy(request):
    """Calls for fixture parameter."""
    return request.param


@pytest.mark.parametrize(
    "bayesian_query_strategy, correct_bayes_kwargs",
    [
        (mc_bald, {"X_pool": np.zeros((10, 150528)).astype(np.float32), "n_instances": 2}),
        (mc_max_entropy, {"X_pool": np.random.randn(10, 150528).astype(np.float32), "n_instances": 2}),
        (mc_max_meanstd, {"X_pool": np.random.randn(10, 150528).astype(np.float32), "n_instances": 2}),
        (mc_max_varratios, {"X_pool": np.random.randn(10, 150528).astype(np.float32), "n_instances": 2}),
    ],
    indirect=["bayesian_query_strategy"],
)
def test_bayesian_query(bayesian_query_strategy, correct_bayes_kwargs):
    """Tests pool based query strategies."""
    X_pool = correct_bayes_kwargs["X_pool"]
    n_instances = correct_bayes_kwargs["n_instances"]

    learner = ActiveLearner(
        estimator=NeuralNetClassifier(
            reshapedVGG(num_classes=2),
            max_epochs=1,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            verbose=0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        query_strategy=bayesian_query_strategy,
        X_training=np.zeros((3, 150528)).astype(np.float32),
        y_training=np.array([0, 1, 0]).reshape((3,)),
    )
    print(X_pool.shape)

    query_idx, query_instances = mc_bald(learner, **correct_bayes_kwargs)

    # selected indexes are valid?
    assert all(idx in range(X_pool.shape[0]) for idx in query_idx), "Returned index out of range"

    # number of instances is correct?
    assert len(query_idx) == n_instances, "Incorrect number of indices returned"
    assert query_instances.shape[0] == n_instances, "Incorrect number of instances returned"

    # are selected instances not duplicated?
    assert len(query_idx) == len(set(query_idx)), "Selected indexes are duplicated"

    ## Edge cases

    # works when n_instances = 1?
    correct_bayes_kwargs["n_instances"] = 1
    query_idx, query_instances = mc_bald(learner, **correct_bayes_kwargs)
    assert len(query_idx) == 1, "Incorrect number of indices returned for n_instances=1"
    assert query_instances.shape[0] == 1, "Incorrect number of instances returned for n_instances=1"

    # works when n_instances equals the pool size?
    correct_bayes_kwargs["n_instances"] = X_pool.shape[0]
    query_idx, query_instances = mc_bald(learner, **correct_bayes_kwargs)

    assert len(query_idx) == X_pool.shape[0], "Incorrect number of indices returned for full pool"
    assert query_instances.shape[0] == X_pool.shape[0], "Incorrect number of instances returned for full pool"
