import pytest
import torch
from skorch import NeuralNetClassifier

from activelearning.utils.skorch_nnet import (
    DropoutNeuralNetClassifier,
    SimpleNeuralNetClassifier,
    TorchModel,
    createNNClassifier,
    reshapedVGG,
)


@pytest.mark.parametrize(
    "model_class, input_dim, output_dim, input_shape, expected_output_shape",
    [
        (SimpleNeuralNetClassifier, 512, 10, (1, 512), (1, 10)),
        (DropoutNeuralNetClassifier, 512, 10, (1, 512), (1, 10)),
        (TorchModel, 28, 2, (1, 3, 28, 28), (1, 2)),
        (reshapedVGG, 10, 10, (1, 150528), (1, 10)),
    ],
)
def test_model_forward_pass(model_class, input_dim, output_dim, input_shape, expected_output_shape):
    """Tests the forward pass of all the defined models."""
    if model_class == reshapedVGG:
        model = model_class(num_classes=output_dim)
    else:
        model = model_class(input_dim=input_dim, output_dim=output_dim)
    input_tensor = torch.randn(input_shape)

    # forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # is output shape correct?
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"


@pytest.mark.parametrize(
    "model_class, params_to_check",
    [
        (SimpleNeuralNetClassifier, ["fc_layers.0.weight"]),
        (DropoutNeuralNetClassifier, ["fc_layers.0.weight", "fc_layers.3.weight"]),
        (TorchModel, ["convs.0.weight", "fcs.0.weight"]),
        (reshapedVGG, ["vgg.features.0.weight", "vgg.classifier.0.weight"]),
    ],
)
def test_model_parameters(model_class, params_to_check):
    """Tests if the model contains the expected parameters."""
    if model_class == reshapedVGG:
        model = model_class(num_classes=10)
    else:
        model = model_class(input_dim=512, output_dim=10)

    model_params = dict(model.named_parameters())

    # are model parameters correct?
    for param in params_to_check:
        assert param in model_params, f"Parameter {param} not found in {model_class.__name__}"


def check_model_params(model, input_dim, output_dim, channels=3):
    """Function that checks model parameters are correct."""
    # Ensure the model is an instance of NeuralNetClassifier
    assert isinstance(model, NeuralNetClassifier)

    # Check the model's module type (which is the classifier)
    model_class = model.module.__class__

    # Check input/output dimensions for the relevant models
    if model_class == SimpleNeuralNetClassifier:
        assert model.module.fc_layers[0].in_features == input_dim
        assert model.module.fc_layers[2].out_features == output_dim
    elif model_class == DropoutNeuralNetClassifier:
        assert model.module.fc_layers[0].in_features == input_dim
        assert model.module.fc_layers[3].out_features == output_dim
    elif model_class == TorchModel:
        assert model.module.input_dim == input_dim
        assert model.module.output_dim == output_dim
        assert model.module.channels == channels
    elif model_class == reshapedVGG:
        assert model.module.vgg.classifier[-1].out_features == output_dim


@pytest.mark.parametrize(
    "classifier, input_dim, output_dim, channels",
    [
        (SimpleNeuralNetClassifier, 512, 10, 3),
        (DropoutNeuralNetClassifier, 512, 10, 3),
        (TorchModel, 28, 2, 3),
        (reshapedVGG, 224, 10, 3),
    ],
)
def test_createNNClassifier(classifier, input_dim, output_dim, channels):
    """Test the createNNClassifier function."""
    model = createNNClassifier(
        input_dim=input_dim, output_dim=output_dim, channels=channels, classifier=classifier, max_epochs=50
    )

    # is model of the correct class?
    assert isinstance(model, NeuralNetClassifier), "Returned model is of wrong class"

    model_class = model.module.__class__

    # are input and output dimensions correct?
    if model_class == SimpleNeuralNetClassifier:
        assert model.module.fc_layers[0].in_features == input_dim, "Wrong input dimension"
        assert model.module.fc_layers[2].out_features == output_dim, "Wrong output dimension"
    elif model_class == DropoutNeuralNetClassifier:
        assert model.module.fc_layers[0].in_features == input_dim, "Wrong input dimension"
        assert model.module.fc_layers[3].out_features == output_dim, "Wrong output dimension"
    elif model_class == TorchModel:
        assert model.module.input_dim == input_dim, "Wrong input dimension"
        assert model.module.output_dim == output_dim, "Wrong output dimension"
        assert model.module.channels == channels, "Wrong channels dimension"
    elif model_class == reshapedVGG:
        assert model.module.vgg.classifier[-1].out_features == output_dim, "Wrong output dimension"
