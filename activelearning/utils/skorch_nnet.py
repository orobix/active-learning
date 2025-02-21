from typing import Any

import torch
from skorch import NeuralNetClassifier
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


class SimpleNeuralNetClassifier(nn.Module):
    """Class for simple neural net classifier."""

    def __init__(self, input_dim: int = 512, output_dim: int = 10):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, output_dim),  # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """Forward pass through the network.

        Args:
            x: tensor

        Returns:
            out: output tensor
        """
        x = x.float()
        out = self.fc_layers(x)
        return out


class DropoutNeuralNetClassifier(nn.Module):
    """Class for neural net classifier with dropout layers."""

    def __init__(self, input_dim=512, output_dim=10, dropout_rate=0.5):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Dropout(dropout_rate),  # dropout layer
            nn.Linear(128, output_dim),  # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """Forward pass through the network.

        Args:
            x: tensor

        Returns:
            out: output tensor
        """
        x = x.float()
        out = self.fc_layers(x)
        return out


class TorchModel(nn.Module):
    """Class for generic torch model."""

    def __init__(self, channels=3, input_dim=28, output_dim=2):
        super().__init__()

        self.channels = channels
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.convs = nn.Sequential(
            nn.Conv2d(self.channels, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fcs = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """Forward pass through the network.

        Args:
            x: tensor

        Returns:
            out: output tensor
        """
        batch_size = x.size(0)
        out = x.view(batch_size, self.channels, self.input_dim, self.input_dim)
        out = self.convs(out)
        out = out.view(batch_size, 12 * 12 * 64)
        out = self.fcs(out)
        return out


class reshapedVGG(nn.Module):
    """Class for VGG16 network wich takes flat input."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)

        for param in self.vgg.parameters():
            param.requires_grad = False

        # new classifier, not frozen
        in_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),  # Output layer with dynamic number of classes
        )
        for param in self.vgg.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """Forward pass through the network.

        Args:
            x: tensor

        Returns:
            out: output tensor
        """
        out = x.view(-1, 3, 224, 224)  # reshape to image form
        out = self.vgg(out)  # pass through vgg
        return out


def createNNClassifier(
    input_dim: int = 512,
    output_dim: int = 10,
    channels: int = 3,
    classifier: nn.Module = SimpleNeuralNetClassifier,
    max_epochs: int = 50,
) -> Any:
    """Creates a neural net classifier using skorch, a sklearn wrapper for torch.
    This way the model can be passed as classifier for modAL ActiveLearner.

    Args:
        input_dim: input size fro classifier
        output_dim: output size (number of classes)
        channels: number of channels of the images
        max_epochs: maximum number of training epochs for the neural net
        classifier: a class for a neural net classifier

    Returns:
        model: neural net classifier
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if classifier == SimpleNeuralNetClassifier:
        model = NeuralNetClassifier(
            SimpleNeuralNetClassifier(input_dim=input_dim, output_dim=output_dim),
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            verbose=0,
            device=device,
        )
    elif classifier == DropoutNeuralNetClassifier:
        model = NeuralNetClassifier(
            DropoutNeuralNetClassifier(input_dim=input_dim, output_dim=output_dim),
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            verbose=0,
            device=device,
        )
    elif classifier == TorchModel:
        model = NeuralNetClassifier(
            TorchModel(input_dim=input_dim, output_dim=output_dim, channels=channels),
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            verbose=1,
            device=device,
        )
    elif classifier == reshapedVGG:
        model = NeuralNetClassifier(
            reshapedVGG(num_classes=output_dim),
            max_epochs=max_epochs,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            train_split=None,
            verbose=1,
            device=device,
        )
    return model
