import torch
import torch.nn as nn

class BasicNN(nn.Module):
    """
    A basic neural network implementation with customizable input, hidden, and output dimensions.

    Attributes:
        hidden_dims (tuple[int, ...]): A tuple specifying the number of neurons in each hidden layer.
        output_dim (int): The number of output features.
        bias (bool): Whether to include a bias term in the final layer.
        activation (nn.Module): The activation function applied between layers.
        layers (nn.ModuleList): The list of layers in the network.
    """

    def __init__(self,
                 hidden_dims: tuple[int, ...] | None,
                 output_dim: int | None,
                 bias: bool):
        """
        Initializes the BasicNN class.

        Args:
            input_dim (int): The number of input features.
            hidden_dims (tuple[int, ...]): A tuple specifying the number of neurons in each hidden layer.
            output_dim (int): The number of output features.
            bias (bool): Whether to include a bias term in the final layer.
        """
        super().__init__()
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self.bias = bias
        self.activation = nn.ELU(alpha=1)
        self.layers = None

    @property
    def input_dim(self) -> int | None:
        if self.layers is not None and len(self.layers) > 0:
            return self.layers[0].in_features
        return None

    @property
    def output_dim(self) -> int | None:
        if self.layers is not None and len(self.layers) > 0:
            return self.layers[-1].out_features
        return self._output_dim

    @property
    def hidden_dims(self) -> tuple[int, ...] | None:
        if self.layers is not None and len(self.layers) > 1:
            return tuple(layer.out_features for layer in self.layers[:-1])
        elif self.layers is not None and len(self.layers) == 1:
            return None
        return self._hidden_dims

    def _build_layers(self, input_dim: int) -> None:
        """
        Constructs the layers of the neural network based on the input, hidden, and output dimensions.

        If no hidden layers are specified, the network will be a simple linear model.
        Otherwise, it will include the specified hidden layers and an output layer.

        Uses `NormalizedLinearWrapper` for normalization of hidden and output layers.

        Args:
            input_dim (int): The number of input features.
        """
        from utils import NormalizedLinearWrapper

        hidden_dims = self.hidden_dims
        output_dim = self.output_dim
        bias = self.bias

        layers = nn.ModuleList()

        if not hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, hidden_dims[0], bias=True))
            for i in range(1, len(hidden_dims)):
                layer = NormalizedLinearWrapper(nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=True))
                layers.append(layer)
            layers.append(NormalizedLinearWrapper(nn.Linear(hidden_dims[-1], output_dim, bias=bias)))

        self.layers = layers

    def _init_layers(self) -> None:
        """
        Initializes the weights and biases of all layers in the network.

        Uses Xavier uniform initialization for weights and sets biases to zero.
        """
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x.squeeze()
