import numpy as np
import pandas as pd
import pytest
from modAL.disagreement import consensus_entropy_sampling
from modAL.uncertainty import uncertainty_sampling
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from activelearning.AL_cycle import cycle_AL, strategy_comparison
from activelearning.AL_selection import selection_AL
from activelearning.queries.bayesian.mc_bald import mc_bald
from activelearning.queries.informative.uncertainty_query_stream import stream_query_uncertainty
from activelearning.queries.representative.coreset_query import query_coreset
from activelearning.queries.representative.coreset_query_stream import stream_query_coreset
from activelearning.queries.representative.random_query import query_random
from activelearning.queries.representative.random_query_stream import stream_query_random


@pytest.mark.parametrize(
    "classifier, query_strategy, n_instances",
    [
        ("randomforest", query_random, 1),
        ("randomforest", query_random, 2),
        ("randomforest", query_coreset, 1),
        ("randomforest", query_coreset, 2),
        ("randomforest", uncertainty_sampling, 1),
        ("randomforest", query_coreset, 2),
        ("nnet", query_random, 1),
        ("nnet", query_random, 2),
        ("nnet", query_coreset, 1),
        ("nnet", query_coreset, 2),
        ("nnet", uncertainty_sampling, 1),
        ("nnet", query_coreset, 2),
        (RandomForestClassifier(), query_random, 1),
        (RandomForestClassifier(), query_random, 1),
    ],
)
def test_cycle_AL(classifier, query_strategy, n_instances):
    """Tests cycle_AL for pool based queries."""
    X_train, y_train = make_classification(
        n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123
    )
    X_pool, y_pool = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)
    X_test, y_test = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)

    ## test without a starting set
    if classifier != "nnet" and query_strategy != uncertainty_sampling:  # this combination requires starting set
        acc_list, inst_list = cycle_AL(
            X_train=None,
            y_train=None,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            classifier=classifier,
            query_strategy=query_strategy,
            n_instances=n_instances,
            max_epochs=1,
        )

        # are outputs not empty?
        assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

        # are accuracies probability values (0<x<1) and are instances integers?
        assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
        assert isinstance(inst_list[-1], int) and inst_list[-1] > 0, "Instance count should be positive"

    ## test with a starting set
    acc_list, inst_list = cycle_AL(
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        query_strategy=query_strategy,
        n_instances=n_instances,
        max_epochs=1,
    )

    # are outputs not empty?
    assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

    # are accuracies probability values (0<x<1) and are instances integers?
    assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
    assert isinstance(inst_list[-1], int) and inst_list[-1] > 0, "Instance count should be positive"


@pytest.mark.parametrize(
    "classifier, query_strategy, n_instances, committee_classifiers",
    [
        ("randomforest", query_random, 1, ["randomforest", "randomforest"]),
        ("randomforest", query_random, 2, ["nnet", "nnet"]),
        ("randomforest", consensus_entropy_sampling, 1, ["randomforest", "randomforest"]),
        ("randomforest", consensus_entropy_sampling, 2, ["nnet", "nnet"]),
        ("nnet", query_random, 1, ["randomforest", "randomforest"]),
        ("nnet", query_random, 2, ["nnet", "nnet"]),
        ("nnet", consensus_entropy_sampling, 1, ["randomforest", "randomforest"]),
        ("nnet", consensus_entropy_sampling, 2, ["nnet", "nnet"]),
        (RandomForestClassifier(), query_random, 1, ["randomforest", "randomforest"]),
        (RandomForestClassifier(), consensus_entropy_sampling, 1, ["nnet", "nnet"]),
    ],
)
def test_cycle_AL_committee(classifier, query_strategy, n_instances, committee_classifiers):
    """Tests cycle_AL for pool based queries."""
    X_train, y_train = make_classification(
        n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123
    )
    X_pool, y_pool = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)
    X_test, y_test = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)

    ## test with a starting set
    acc_list, inst_list = cycle_AL(
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        query_strategy=query_strategy,
        n_instances=n_instances,
        max_epochs=1,
    )

    # are outputs not empty?
    assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

    # are accuracies probability values (0<x<1) and are instances integers?
    assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
    assert isinstance(inst_list[-1], int) and inst_list[-1] > 0, "Instance count should be positive"


@pytest.mark.parametrize(
    "classifier, query_strategy, n_instances",
    [("nnet_bo", query_random, 1), ("nnet_bo", query_random, 2), ("nnet_bo", mc_bald, 1), ("nnet_bo", mc_bald, 2)],
)
def test_cycle_AL_bayesian(classifier, query_strategy, n_instances):
    """Tests cycle_AL for pool based queries."""
    X_train, y_train = make_classification(
        n_samples=10, n_features=150528, n_classes=10, n_informative=5, random_state=123
    )
    X_pool, y_pool = make_classification(
        n_samples=10, n_features=150528, n_classes=10, n_informative=5, random_state=123
    )
    X_test, y_test = make_classification(
        n_samples=10, n_features=150528, n_classes=10, n_informative=5, random_state=123
    )

    ## test with a starting set
    acc_list, inst_list = cycle_AL(
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_pool=X_pool.astype(np.float32),
        y_pool=y_pool,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        classifier=classifier,
        query_strategy=query_strategy,
        n_instances=n_instances,
        max_epochs=1,
    )

    # are outputs not empty?
    assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

    # are accuracies probability values (0<x<1) and are instances integers?
    assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
    assert isinstance(inst_list[-1], int) and inst_list[-1] > 0, "Instance count should be positive"


@pytest.mark.parametrize(
    "classifier, query_strategy, n_instances",
    [
        ("randomforest", query_random, 1),
        ("randomforest", query_random, 2),
        ("randomforest", query_coreset, 1),
        ("randomforest", query_coreset, 2),
        ("randomforest", uncertainty_sampling, 1),
        ("randomforest", query_coreset, 2),
        ("nnet", query_random, 1),
        ("nnet", query_random, 2),
        ("nnet", query_coreset, 1),
        ("nnet", query_coreset, 2),
        ("nnet", uncertainty_sampling, 1),
        ("nnet", query_coreset, 2),
        (RandomForestClassifier(), query_random, 1),
        (RandomForestClassifier(), query_random, 1),
    ],
)
def test_cycle_AL_stream_batch(classifier, query_strategy, n_instances):
    """Tests cycle_AL in batch stream scenario."""
    X_train, y_train = make_classification(
        n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123
    )
    X_pool, y_pool = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)
    X_test, y_test = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)

    ## test without a starting set
    if classifier != "nnet" and query_strategy != uncertainty_sampling:  # this combination requires starting set
        acc_list, inst_list = cycle_AL(
            X_train=None,
            y_train=None,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            classifier=classifier,
            query_strategy=query_strategy,
            n_instances=n_instances,
            max_epochs=1,
            batch_size=2,
        )

        # are outputs not empty?
        assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

        # are accuracies probability values (0<x<1) and are instances integers?
        assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
        assert isinstance(inst_list[-1], int) and inst_list[-1] >= 0, "Instance count should be positive"

    ## test with a starting set
    acc_list, inst_list = cycle_AL(
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        query_strategy=query_strategy,
        n_instances=n_instances,
        max_epochs=1,
        batch_size=2,
    )

    # are outputs not empty?
    assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

    # are accuracies probability values (0<x<1) and are instances integers?
    assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
    assert isinstance(inst_list[-1], int) and inst_list[-1] >= 0, "Instance count should be positive"


@pytest.mark.parametrize(
    "classifier, query_strategy, n_instances",
    [
        ("randomforest", stream_query_random, 1),
        ("randomforest", stream_query_coreset, 1),
        ("randomforest", stream_query_uncertainty, 1),
        ("nnet", stream_query_random, 1),
        ("nnet", stream_query_coreset, 1),
        ("nnet", stream_query_uncertainty, 1),
        (RandomForestClassifier(), stream_query_random, 1),
        (RandomForestClassifier(), stream_query_uncertainty, 1),
    ],
)
def test_cycle_AL_stream(classifier, query_strategy, n_instances):
    """Tests cycle_AL for stream based queries."""
    X_train, y_train = make_classification(
        n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123
    )
    X_pool, y_pool = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)
    X_test, y_test = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)

    ## test without a starting set
    if classifier != "nnet" and query_strategy != uncertainty_sampling:  # this combination requires starting set
        acc_list, inst_list = cycle_AL(
            X_train=None,
            y_train=None,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            classifier=classifier,
            query_strategy=query_strategy,
            n_instances=n_instances,
            max_epochs=1,
        )

        # are outputs not empty?
        assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

        # are accuracies probability values (0<x<1) and are instances integers?
        assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
        assert isinstance(inst_list[-1], int) and inst_list[-1] >= 0, "Instance count should be positive"

    ## test with a starting set
    acc_list, inst_list = cycle_AL(
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        query_strategy=query_strategy,
        n_instances=n_instances,
        max_epochs=1,
    )

    # are outputs not empty?
    assert len(acc_list) > 0 and len(inst_list) > 0, "Output is empty lists"

    # are accuracies probability values (0<x<1) and are instances integers?
    assert acc_list[-1] >= 0.0 and acc_list[-1] <= 1.0, "Accuracy should be between 0 and 1"
    assert isinstance(inst_list[-1], int) and inst_list[-1] >= 0, "Instance count should be positive"


@pytest.mark.parametrize(
    "classifier, query_strategies, n_instances",
    [
        ("randomforest", [query_random], [1, 2]),
        ("randomforest", [query_coreset, uncertainty_sampling], [1, 2]),
        ("nnet", [query_random], [1, 2]),
        ("nnet", [query_coreset, uncertainty_sampling], [1, 2]),
        (RandomForestClassifier(), [query_random], [1, 2]),
        (RandomForestClassifier(), [uncertainty_sampling, query_coreset], [1, 2]),
    ],
)
def test_strategy_comparison(classifier, query_strategies, n_instances):
    """Tests strategy_comparison for pool based queries."""
    X_train, y_train = make_classification(
        n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123
    )
    X_pool, y_pool = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)
    X_test, y_test = make_classification(n_samples=10, n_features=512, n_classes=10, n_informative=5, random_state=123)

    scores = strategy_comparison(
        X_train=X_train,
        y_train=y_train,
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        query_strategies=query_strategies,
        n_instances=n_instances,
        max_epochs=5,
        goal_acc=0.9,
    )

    # output is in the expected form?
    assert len(scores) == len(n_instances)
    assert len(scores[0].columns) == 2 * len(query_strategies), "Scores is missing columns"
    assert isinstance(scores[0], pd.DataFrame), "Elements in scores should be DataFrames"


@pytest.mark.parametrize(
    "query_strategy, n_instances",
    [
        (query_random, 1),
        (query_random, 2),
        (query_coreset, 1),
        (query_coreset, 2),
    ],
)
def test_selection_AL(query_strategy, n_instances):
    """Tests selection_AL function."""
    X_train = np.random.randn(10, 512)
    X_pool = np.random.randn(10, 512)

    ## test with a starting set
    train, idxs = selection_AL(
        X_train=X_train,
        X_pool=X_pool,
        query_strategy=query_strategy,
        n_instances=n_instances,
        n_queries=2,
        og_idx=True,
    )

    # are outputs in the correct form?
    assert len(train) > 0 and len(idxs) > 0, "Output is empty lists with starting set"
    assert train.shape[0] == len(idxs), "Data and idxs have different lengths with starting set"
    assert isinstance(idxs[-1], int) and idxs[-1] >= 0, "Indexes should be positive with starting set"

    ## test without a starting set
    train, idxs = selection_AL(
        X_train=None,
        X_pool=X_pool,
        query_strategy=query_strategy,
        n_instances=n_instances,
        n_queries=2,
        og_idx=True,
    )

    # are outputs in the correct form?
    assert len(train) > 0 and len(idxs) > 0, "Output is empty lists without starting set"
    assert train.shape[0] == len(idxs), "Data and idxs have different lengths without starting set"
    assert isinstance(idxs[-1], int) and idxs[-1] >= 0, "Indexes should be positive without starting set"
