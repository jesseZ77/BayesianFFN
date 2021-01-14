import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, i):
        return self.linear(i)


class FFN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1,
                 dense_activation=torch.sigmoid,
                 output_activation=torch.sigmoid):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dense_activation = dense_activation
        self.output_activation = output_activation

    def forward(self, i):
        x = self.dense_activation(self.fc1(i))
        o = self.output_activation(self.fc2(x))
        return o


class MatrixLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.U = nn.Parameter(torch.empty([output_dim[0], input_dim[0]]))
        self.V = nn.Parameter(torch.empty([input_dim[1], output_dim[1]]))
        self.B = nn.Parameter(torch.empty(output_dim))

        torch.nn.init.kaiming_normal_(self.U)
        torch.nn.init.kaiming_normal_(self.V)
        torch.nn.init.kaiming_normal_(self.B)

    def forward(self, i):
        x = torch.matmul(torch.matmul(self.U, i), self.V) + self.B
        if self.activation is None:
            return x
        return self.activation(x)


class MatrixNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.hidden_layers = nn.ModuleList([MatrixLayer(self.input_dim, self.hidden_dim, torch.sigmoid)])
        for i in range(self.n_hidden_layers - 1):
            self.hidden_layers.append(MatrixLayer(self.hidden_dim, self.hidden_dim, torch.sigmoid))
        self.output_layer = MatrixLayer(self.hidden_dim, output_dim)

    def forward(self, i):
        x = self.hidden_layers[0](i)
        for i in range(self.n_hidden_layers - 1):
            x = self.hidden_layers[i + 1](x)
        o = self.output_layer(x)
        return o