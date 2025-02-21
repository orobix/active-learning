from typing import Any

import numpy as np
import skorch
import torch
from modAL.utils.selection import multi_argmax


def mc_bald(classifier: Any, X_pool: np.ndarray, n_instances: int = 1) -> tuple[list, np.ndarray]:
    """Chooses a number of instances to queried/added to be labeled pool with monte carlo dropout BALD criterion.

    Args:
        classifier: the classifier the active learning model uses. This isn't called in the function,
        but it's necessary to adhere to the expected signature of query strategies in modAL
        X_pool: pool of unlabeled instances
        n_instances: number of instances to be selected

    Returns:
        The indexes of the selected instances, and the selected instances
    """
    predictions = get_mc_preds(classifier, X_pool)

    mean = np.mean(predictions, axis=0)

    epsilon = 1e-10
    # entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=1)

    # mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(np.sum(-predictions * np.log(predictions + epsilon), axis=-1), axis=0)

    selected_idx, _ = multi_argmax(mutual_info, n_instances=n_instances)
    return (
        np.array(selected_idx),
        X_pool[np.array(selected_idx),],
    )


def enable_dropout_layers(model: skorch.NeuralNetClassifier) -> None:
    """Function to activate the dropout layers.

    Args:
        model: skorch wrapped pytorch neaural net classifier
    """
    for m in model.module_.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def get_mc_preds(model: skorch.NeuralNetClassifier, X_pool: np.ndarray, n_cycles: int = 50) -> np.ndarray:
    """Function to get the softmax predictions of mc dropout.

    Args:
        model: skorch wrapped pytorch neural net classifier
        X_pool: pool of unlabeled data
        n_cycles: number of forward passes through the model with dropout activated

    Returns:
        predictions (np.ndarray): vector of shape (n_cycles, n_samples, n_classes) with the predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.estimator
    model.module_.eval()
    softmax = torch.nn.Softmax(dim=1)
    enable_dropout_layers(model)

    # predictions of mc forward passes
    for i in range(n_cycles):
        with torch.no_grad():
            pred_y = model.module_(torch.tensor(X_pool).to(device)).cpu()
            prob_y = softmax(pred_y)
            if i == 0:  # if first iteration, setup empty array
                n_classes = prob_y.shape[1]
                predictions = np.empty((n_cycles, X_pool.shape[0], n_classes))
        predictions[i, :, :] = np.array(prob_y)

    model.module_.eval()
    return predictions
