from unittest.mock import MagicMock

import numpy as np
import pytest
from modAL import ActiveLearner, Committee
from modAL.disagreement import consensus_entropy_sampling, max_disagreement_sampling, vote_entropy_sampling
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skorch import NeuralNetClassifier

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
from activelearning.utils.AL_utils import score_accuracy, set_committee, set_learner, set_query_args, set_query_names


@pytest.fixture
def mock_query_strategy():
    """Fixture for mocking query strategy."""
    return MagicMock()


@pytest.mark.parametrize(
    "classifier, X_train, correct_class",
    [
        ("randomforest", np.random.rand(10, 5).astype(np.float32), RandomForestClassifier),
        ("kneighbors", np.random.rand(10, 5).astype(np.float32), KNeighborsClassifier),
        ("nnet", np.random.rand(10, 5).astype(np.float32), NeuralNetClassifier),
        ("nnet_dropout", np.random.rand(10, 5).astype(np.float32), NeuralNetClassifier),
        ("nnet_bo", np.random.rand(10, 150528).astype(np.float32), NeuralNetClassifier),
    ],
)
def test_set_learner(classifier, X_train, correct_class, mock_query_strategy):
    """Tests set_learner function."""
    y_train = np.random.randint(0, 2, size=10)

    learner = set_learner(
        classifier=classifier,
        query_strategy=mock_query_strategy,
        X_train=X_train,
        y_train=y_train,
        input_dim=5,
        output_dim=2,
        max_epochs=10,
        device="cpu",
    )

    # is output of the correct class?
    assert isinstance(learner, ActiveLearner), "Wrong class for learner"
    # is classifier of the correct class?
    assert isinstance(learner.estimator, correct_class), "Wrong class for estimator"


def test_set_learner_invalid_classifier(mock_query_strategy):
    """Tests set_learner function with invalid classifier string."""
    X_train = np.random.rand(10, 5).astype(np.float32)
    y_train = np.random.randint(0, 2, size=10)

    # raises error with invalid classifier?
    with pytest.raises(ValueError, match="Invalid value for classifier"):
        set_learner(
            classifier="invalid_classifier",
            query_strategy=mock_query_strategy,
            X_train=X_train,
            y_train=y_train,
            input_dim=5,
            output_dim=2,
            max_epochs=10,
            device="cpu",
        )


@pytest.mark.parametrize(
    "committee_classifiers, X_train, expected_classes",
    [
        (
            ["randomforest", "kneighbors"],
            np.random.rand(10, 5).astype(np.float32),
            [RandomForestClassifier, KNeighborsClassifier],
        ),
        (
            ["nnet", "nnet_dropout"],
            np.random.rand(10, 5).astype(np.float32),
            [NeuralNetClassifier, NeuralNetClassifier],
        ),
        (["nnet_bo"], np.random.rand(10, 150528).astype(np.float32), [NeuralNetClassifier]),
        (
            [RandomForestClassifier(), KNeighborsClassifier()],
            np.random.rand(10, 5).astype(np.float32),
            [RandomForestClassifier, KNeighborsClassifier],
        ),
    ],
)
def test_set_committee(committee_classifiers, X_train, expected_classes, mock_query_strategy):
    """Tests set_committee function."""
    y_train = np.random.randint(0, 2, size=10)

    committee = set_committee(
        committee_classifiers=committee_classifiers,
        query_strategy=mock_query_strategy,
        X_train=X_train,
        y_train=y_train,
        input_dim=5,
        output_dim=2,
        max_epochs=10,
        device="cpu",
    )

    # is output of the correct class?
    assert isinstance(committee, Committee), "Output is not a Committee"
    # are all the classifiers of the correct class?
    for learner, expected_class in zip(committee.learner_list, expected_classes, strict=True):
        assert isinstance(learner, ActiveLearner), "Learner in committee is not an ActiveLearner"
        assert isinstance(learner.estimator, expected_class), f"Estimator in learner is not of type {expected_class}"


@pytest.fixture
def mock_classifier():
    """Fixture for mocking a classifier."""
    return MagicMock()


@pytest.mark.parametrize(
    "query_strategy, expected_keys",
    [
        (query_kmeans_foreach, {"X_pool", "K", "n_instances"}),
        (query_density, {"X_pool", "n_instances", "metric"}),
        (query_diversity, {"X_pool", "X_start", "n_instances"}),
        (query_random, {"X_pool", "n_instances"}),
        (query_coreset, {"X_pool", "X_start", "n_instances"}),
        (query_probcover, {"X_pool", "X_start", "n_instances", "delta"}),
        (stream_query_diversity, {"X_pool", "X_start", "n_instances", "quantile", "start_len"}),
        (stream_query_coreset, {"X_pool", "X_start", "n_instances", "quantile", "start_len"}),
        (stream_query_probcover, {"X_pool", "X_start", "n_instances", "n_classes", "start_len"}),
        (stream_query_random, {"X_pool", "n_instances", "perc"}),
        (uncertainty_sampling, {"X_pool", "n_instances"}),
        (margin_sampling, {"X_pool", "n_instances"}),
        (entropy_sampling, {"X_pool", "n_instances"}),
        (stream_query_uncertainty, {"classifier", "X_pool", "X_start", "n_instances", "threshold", "start_len"}),
        (stream_query_margin, {"classifier", "X_pool", "X_start", "n_instances", "threshold", "start_len"}),
        (stream_query_entropy, {"classifier", "X_pool", "X_start", "n_instances", "threshold", "start_len"}),
        (vote_entropy_sampling, {"X_pool", "n_instances"}),
        (consensus_entropy_sampling, {"X_pool", "n_instances"}),
        (max_disagreement_sampling, {"X_pool", "n_instances"}),
        (mc_bald, {"X_pool", "n_instances"}),
        (mc_max_entropy, {"X_pool", "n_instances"}),
        (mc_max_varratios, {"X_pool", "n_instances"}),
        (mc_max_meanstd, {"X_pool", "n_instances"}),
    ],
)
def test_set_query_args(mock_classifier, query_strategy, expected_keys):
    """Tests set_query_args function."""
    X_pool = np.random.rand(10, 5).astype(np.float32)
    y_pool = np.random.randint(0, 2, size=10)
    X_start = np.random.rand(5, 5).astype(np.float32)
    n_instances = 5
    K = 3
    metric = "euclidean"
    quantile = 0.5
    start_len = 3

    query_args = set_query_args(
        classifier=mock_classifier,
        query_strategy=query_strategy,
        X_pool=X_pool,
        y_pool=y_pool,
        X_start=X_start,
        n_instances=n_instances,
        K=K,
        metric=metric,
        quantile=quantile,
        start_len=start_len,
    )

    # query_args keys are correct?
    assert set(query_args.keys()) == expected_keys, f"Expected keys: {expected_keys}, but got: {set(query_args.keys())}"

    # query arguments values are correct?
    if "X_pool" in expected_keys:
        assert np.array_equal(query_args["X_pool"], X_pool), "X_pool does not match"
    if "n_instances" in expected_keys:
        assert query_args["n_instances"] == n_instances, "n_instances does not match"
    if "K" in expected_keys:
        assert query_args["K"] == K, "K does not match"
    if "metric" in expected_keys:
        assert query_args["metric"] == metric, "metric does not match"
    if "quantile" in expected_keys:
        assert query_args["quantile"] == quantile, "quantile does not match"
    if "start_len" in expected_keys:
        assert query_args["start_len"] == start_len, "start_len does not match"
    if "delta" in expected_keys:
        assert "delta" in query_args, "delta is missing from query_args"
    if "n_classes" in expected_keys:
        assert query_args["n_classes"] == len(np.unique(y_pool)), "n_classes does not match"


@pytest.fixture
def mock_learner():
    """Fixture for mocking an ActiveLearner."""
    learner = MagicMock()
    # mocked method of ActiveLearner
    learner.score.side_effect = lambda X, y: np.mean(X.sum(axis=1) + y)
    return learner


@pytest.mark.parametrize(
    "acc, X_pool_size, X_test_size",
    [
        ("joint", 10, 5),
        ("test", 10, 5),
        ("pool", 10, 5),
    ],
)
def test_score_accuracy(mock_learner, acc, X_pool_size, X_test_size):
    """Tests score_accuracy function."""
    X_pool = np.random.rand(X_pool_size, 5).astype(np.float32)
    y_pool = np.random.randint(0, 2, size=X_pool_size)
    X_test = np.random.rand(X_test_size, 5).astype(np.float32)
    y_test = np.random.randint(0, 2, size=X_test_size)

    accuracy = score_accuracy(mock_learner, acc, X_pool, y_pool, X_test, y_test)

    # is accuracy a float?
    assert isinstance(accuracy, float), "Accuracy is not a float value"


@pytest.mark.parametrize(
    "query_strategies, expected_names",
    [
        ([query_kmeans_foreach], ["kmeans"]),
        ([query_density], ["density"]),
        ([query_diversity], ["diversity"]),
        ([query_coreset], ["coreset"]),
        ([query_probcover], ["probcover"]),
        ([query_random], ["random"]),
        ([stream_query_diversity], ["streamdiv"]),
        ([stream_query_coreset], ["streamcore"]),
        ([stream_query_probcover], ["streamprob"]),
        ([stream_query_random], ["streamrandom"]),
        ([uncertainty_sampling], ["uncertainty"]),
        ([margin_sampling], ["margin"]),
        ([entropy_sampling], ["entropy"]),
        ([stream_query_uncertainty], ["streamunc"]),
        ([stream_query_margin], ["streammargin"]),
        ([stream_query_entropy], ["streamentropy"]),
        ([vote_entropy_sampling], ["voteentropy"]),
        ([consensus_entropy_sampling], ["consensus"]),
        ([max_disagreement_sampling], ["disagreement"]),
        ([mc_bald], ["BALD"]),
        ([mc_max_entropy], ["MC max entropy"]),
        ([mc_max_varratios], ["MC max VarRatios"]),
        ([mc_max_meanstd], ["MC max meanstd"]),
    ],
)
def test_set_query_names_valid(query_strategies, expected_names):
    """Tests set_query_names function with valid inputs."""
    result = set_query_names(query_strategies)
    assert result == expected_names, f"Expected {expected_names}, but got {result}"


def test_set_query_names_invalid():
    """Tests set_query_names function with invalid input."""
    invalid_strategy = MagicMock(name="invalid_strategy")
    with pytest.raises(ValueError, match="Received a not valid query strategy"):
        set_query_names([invalid_strategy])
