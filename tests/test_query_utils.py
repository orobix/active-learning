import numpy as np
import pytest
import torch
from modAL import ActiveLearner
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from skorch import NeuralNetClassifier
from torch import nn

from activelearning.queries.bayesian.mc_bald import enable_dropout_layers, get_mc_preds
from activelearning.queries.representative.diversity_query import diversity_measure
from activelearning.queries.representative.kmeans_query import kmeans_min_distances
from activelearning.queries.representative.probcover_query import estimate_delta, is_ball_pure


def test_diversity_measure():
    """Tests diversity measure calculation."""
    X_pool = np.array([[1, 0], [0, 1]])
    X_start = np.array([[1, 1], [-1, -1]])

    result = diversity_measure(X_pool, X_start)
    expected = np.array(
        [np.arccos(1 / np.sqrt(2)) + np.arccos(-1 / np.sqrt(2)), np.arccos(1 / np.sqrt(2)) + np.arccos(-1 / np.sqrt(2))]
    )

    assert result.shape == expected.shape, "Wrong output shape"
    assert np.all(result == expected), "Wrong output"


def test_kmeans_min_distances():
    """Tests k-means min distances calculation."""
    X_pool = np.array([[0, 0], [1, 1], [2, 2], [10, 10]])
    K = 2

    result_1 = kmeans_min_distances(K, X_pool, for_each=False, random_state=123)
    result_2 = kmeans_min_distances(K, X_pool, for_each=True, random_state=123)

    assert result_1.shape == (4,) and result_2.shape == (4, 2), "Wrong output shape"

    # all values are the same
    X_pool = np.ones((10, 2))
    K = 3
    result = kmeans_min_distances(K, X_pool, for_each=True, random_state=123)

    assert np.all(result == 0), "Wrong result when all values are the same"


def test_estimate_delta():
    """Tests delta estimation for probcover."""
    # is ball purity threshold met?
    X_pool = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])  # Two clusters
    K = 2
    alpha = 0.95

    delta = estimate_delta(X_pool, K, alpha)
    D = euclidean_distances(X_pool)
    labels = KMeans(n_clusters=K).fit(X_pool).labels_

    pure_balls_count = sum(is_ball_pure(D, i, delta, labels) for i in range(len(labels)))
    pure_balls_percentage = pure_balls_count / len(labels)
    assert pure_balls_percentage >= alpha, "Delta does not meet the required purity threshold"

    ## Edge case
    # as many clusters as there are data points
    K = 1

    delta = estimate_delta(X_pool, K, alpha)
    D = euclidean_distances(X_pool)
    labels = KMeans(n_clusters=K).fit(X_pool).labels_

    pure_balls_count = sum(is_ball_pure(D, i, delta, labels) for i in range(len(labels)))
    pure_balls_percentage = pure_balls_count / len(labels)
    assert pure_balls_percentage == 1, "All balls should be pure when each has its cluster"


def test_is_ball_pure():
    """Tests the is_ball_pure function."""
    D = np.array([[0, 0.5, 1.5, 2], [0.5, 0, 1, 1.5], [1.5, 1, 0, 0.5], [2, 1.5, 0.5, 0]])
    labels = np.array([1, 1, 2, 2])

    # outputis correct?
    assert is_ball_pure(D, i=0, delta=0.6, labels=labels) == True, "Pure ball output is False"  # noqa: E712
    assert is_ball_pure(D, i=2, delta=0.6, labels=labels) == True, "Pure ball output is False"  # noqa: E712
    assert is_ball_pure(D, i=1, delta=1.5, labels=labels) == False, "Impure ball output is True"  # noqa: E712
    assert is_ball_pure(D, i=3, delta=2.0, labels=labels) == False, "Impure ball output is True"  # noqa: E712


class TestModel(nn.Module):
    """Dummy model with dropout layer."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout()
        self.layer2 = nn.Linear(10, 2)

    def forward(self, x):
        """Forward passes."""
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


@pytest.fixture
def model():
    """Wraps the test model in skorch."""
    return NeuralNetClassifier(TestModel)


def test_enable_dropout_layers(model):
    """Tests enabling of dropout layers."""
    model.initialize()
    model.module_.eval()

    enable_dropout_layers(model)

    for layer in model.module_.modules():
        if isinstance(layer, nn.Dropout):
            assert layer.training is True, "Dropout layer is not activated"


def test_get_mc_preds():
    """Tests MC predictions."""
    X_pool = np.random.randn(10, 10).astype(np.float32)
    X_start = np.random.randn(10, 10).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).reshape((10,))

    model = TestModel()
    learner = ActiveLearner(
        estimator=NeuralNetClassifier(
            model,
            max_epochs=1,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=0,
        ),
        X_training=X_start,
        y_training=labels,
    )

    n_cycles = 5
    predictions = get_mc_preds(learner, X_pool, n_cycles)

    # is shape of the output correct?
    assert np.all(predictions.shape == (n_cycles, X_pool.shape[0], 2)), "Incorrect shape of predictions"

    # are all predictions probabilities? (0<x<1)
    assert np.all(predictions >= 0) and np.all(predictions <= 1), "Predictions have value out of probability range"

    # is sum of softmax probabilities approximately 1?
    sum_probs = predictions.sum(axis=2)
    assert np.allclose(sum_probs, 1)
