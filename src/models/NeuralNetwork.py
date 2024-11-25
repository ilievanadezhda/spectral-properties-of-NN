import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Sigmoid):
        """
        Initialize neural network with configurable layers
        Args:
            layer_sizes: List of integers representing the size of each layer,
                        including input and output layers
            activation: Activation function to use between layers (default: nn.Sigmoid)
        """
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # don't add activation after last layer
                layers.append(activation())

        self.linear_activation_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_activation_stack(x)
        return logits
