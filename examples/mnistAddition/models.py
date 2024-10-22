# TODO: remove unused libraries: random, numpy, torch.autograd, os
import random
import numpy as np
import torch
from torch.autograd import Variable
import os


class MNIST_Net(torch.nn.Module):
    """
        A neural network model for processing MNIST digits.

        The model consists of an encoder network using convolutional layers followed by a classifier.
        It takes an input image and returns the probability distribution over digits as well its encoded feature representation.

        Attributes
        ----------
        encoder : torch.nn.Sequential
            The encoder network with convolutional layers to process the input image.
        classifier_mid : torch.nn.Sequential
            A fully connected layer block that takes in encoded features.
        classifier : torch.nn.Sequential
            A fully connected layer followed by a Softmax layer that outputs a probability distribution over digits.

        Parameters
        ----------
        N : int, optional
            Number of output classes, default is 10 (for MNIST).
        channels : int, optional
            Number of input image channels, default is 1 (for grayscale MNIST).
    """
    def __init__(self, N=10, channels=1):
        super(MNIST_Net, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 12, 5),
            torch.nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            torch.nn.ReLU(True),
            torch.nn.Conv2d(12, 16, 5),  # 6 12 12 -> 16 8 8
            torch.nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            torch.nn.ReLU(True)
        )
        self.classifier_mid = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(84, N),
            torch.nn.Softmax(1)
        )
        self.channels = channels

    def weights_init(self, m):
        """
            Initialize weights for the model using Xavier uniform initialization for both Conv2D and Linear layers.

            Parameters
            ----------
            m : torch.nn.Module
                A layer of the neural network to apply the weight initialization.
        """
        if isinstance(m, torch.nn.Conv2d):
            print('init conv2, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, torch.nn.Linear):
            print('init Linear, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Forward pass through the network.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape [batch_size, channels, height, width].

            Returns
            -------
            x1 : torch.Tensor
                Final output class predictions after passing through the classifier.
            x : torch.Tensor
                Encoded feature representation from the middle layers.
        """
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier_mid(x)
        x1 = self.classifier(x)
        return x1, x


class MNISTSumModel(torch.nn.Module):
    """
        Model designed to solve a summation task for MNIST digits using greedy/ε-greedy policy.

        This model uses two neural networks to process two digit inputs, greedy/ε-greedy policy for both
        the digit outputs and rule selection, and makes predictions based on the combination of these outputs.

        Attributes:
        ----------
        nn : torch.nn.Module
            The primary neural network used to process one MNIST digit.
        nn2 : torch.nn.Module
            The secondary neural network used to process the second MNIST digit (can be the same as nn).
        weights : torch.nn.Parameter
            A matrix of trainable weights that represent the rules for combining the digit outputs.
        epsilon_digits : float
            The epsilon value controlling greedy/ε-greedy policy for digit selection.
        epsilon_rules : float
            The epsilon value controlling greedy/ε-greedy policy for rule selection.
        device : str
            Device on which the model is running (e.g., 'cpu' or 'cuda').

        Parameters:
        ----------
        nn : torch.nn.Module
            Primary neural network for digit feature extraction.
        epsilon_digits : float
            Epsilon value for ε-greedy policy on digits.
        epsilon_rules : float
            Epsilon value for ε-greedy policy on rule selection.
        nn2 : torch.nn.Module, optional
            Secondary neural network for the second digit (defaults to the same as `nn`).
        output_dim : int, optional
            The number of output dimensions (defaults to 19 for summation).
        device : str, optional
            Device to run the model on (defaults to 'cpu').
        n_digits : int, optional
            Number of digit categories (defaults to 10 for MNIST).
    """
    def __init__(self, nn, epsilon_digits, epsilon_rules, nn2=None, output_dim=19, device='cpu', n_digits=10):
        super(MNISTSumModel, self).__init__()
        self.nn = nn
        self.device = device
        if nn2 is not None:
            self.nn2 = nn2
        else:
            self.nn2 = nn
        self.weights = torch.nn.Parameter(torch.randn([n_digits, n_digits, output_dim]).to(self.device))
        self.weights.requires_grad = True
        self.epsilon_digits = epsilon_digits
        self.epsilon_rules = epsilon_rules

    def epsilon_greedy(self, t, eval, dim=1):
        """
            Apply greedy/ε-greedy policies for symbols digit selection.
            With probability ε, it selects a random symbol (i.e., apply ε-greedy policy), otherwise it selects the max symbol
            (i.e, apply greedy policy).

            Parameters:
            ----------
            t : torch.Tensor
                The output tensor from the neural network.
            eval : bool
                If True, use the greedy policy. Otherwise, apply ε-greedy.
            dim : int, optional
                The dimension to perform selection along (default is 1).

            Returns:
            ----------
            truth_values : torch.Tensor
                The truth values corresponding to the chosen symbols.
            chosen_symbols : torch.Tensor
                The indices of the chosen symbols.
        """
        if eval:
            # During evaluation, get the truth values and (indices) of the symbols with the maximum value
            truth_values, chosen_symbols = torch.max(t, dim=dim)
        else:
            # During training, apply ε-greedy policy

            # `random_selection` is a boolean vector where True indicates selecting a random symbol and False indicates
            # selecting the symbol with the maximum value.
            random_selection = torch.rand((t.shape[0],)) < self.epsilon_digits
            random_selection = random_selection.to(self.device)

            # `symbol_index_random` contains the random indices for symbol selection.
            symbol_index_random = torch.randint(t.shape[1], (t.shape[0],))
            symbol_index_random = symbol_index_random.to(self.device)
            # `symbol_index_max` contains the indices of the symbols with the maximum values.
            _, symbol_index_max = torch.max(t, dim=dim)

            # Use the `random_selection` mask to decide between selecting a random symbol or the symbol with the maximum value.
            # If `random_selection` is True, use the random index; otherwise, use the index of the maximum value.
            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)
            # Gather the corresponding truth values (actual values from the tensor `t`) based on the indices in `chosen_symbols`.
            truth_values = torch.gather(t, dim, chosen_symbols.view(-1, 1))

        return truth_values, chosen_symbols

    def get_rules_matrix(self, eval):
        """
            Apply greedy/ε-greedy policy for rule (i.e., symbol) selection.

            Parameters
            ----------
            eval : bool
                If True, use the greedy policy. Otherwise, use  ε-greedy policy.

            Returns
            -------
            truth_values : torch.Tensor
                The selected rules' truth values based on greedy/ε-greedy policy.
            chosen_symbols : torch.Tensor
                The indices of the selected rules.
        """
        if eval:
            # During evaluation, get the truth values and (indices) of the rules with the maximum value
            return torch.max(torch.nn.functional.softmax(self.weights, dim=2), dim=2, keepdim=True)
        else:

            # number of digit classes (i.e., 10 for MNIST)
            n_digits = self.weights.shape[0]
            # number of possible output symbols (i.e., summation rules), which is 19 in this case
            # (because the sum of two MNIST digits ranges from 0 to 18)
            n_output_symbols = self.weights.shape[2]

            # `random_selection` is a boolean matrix of size (n_digits, n_digits) where each element determines
            # if a random (True) or a maximum value (False) rule should be selected.
            random_selection = torch.rand((n_digits, n_digits)) < self.epsilon_rules
            random_selection = random_selection.to(self.device)

            # `symbol_index_random` contains randomly selected indices for rules.
            # For each digit pair (i, j), it selects a random index from `n_output_symbols` possible summation rules.
            symbol_index_random = torch.randint(n_output_symbols, (n_digits, n_digits))
            symbol_index_random = symbol_index_random.to(self.device)

            # `symbol_index_max` contains the indices of the maximum values rules in the rule weights tensor.
            _, symbol_index_max = torch.max(self.weights, dim=2)

            # Use the `random_selection` mask to decide between selecting a random symbol or the symbol with the maximum value.
            # If `random_selection` is True for a given pair of digits, use the random index;
            # otherwise, use the index of the maximum value rule
            chosen_symbols = torch.where(random_selection, symbol_index_random, symbol_index_max)

            # `truth_values` contains the truth values of the selected rules.
            # First, the softmax is applied along the third dimension of `self.weights`, which converts rule weights to probabilities.
            # Then, `torch.gather` is used to retrieve the selected values based on the indices in `chosen_symbols`.
            truth_values = torch.gather(torch.nn.functional.softmax(self.weights, dim=2),
                                        2, chosen_symbols.view(n_digits, n_digits, 1)).view(n_digits, n_digits)

            return truth_values, chosen_symbols

    def forward(self, x, y, eval=False):
        # Get the predictions (digit class probabilities) from the two networks
        x, _ = self.nn(x)
        y, _ = self.nn2(y)

        # Apply  policy(ies) to get truths values and symbols for both digits and rules
        truth_values_x, chosen_symbols_x = self.epsilon_greedy(x, eval)
        truth_values_y, chosen_symbols_y = self.epsilon_greedy(y, eval)
        rules_weights, g_matrix = self.get_rules_matrix(eval)

        # Concatenate the chosen rules and the selected digits truth values
        symbols_truth_values = torch.concat(
            [rules_weights[chosen_symbols_x, chosen_symbols_y].view(-1, 1),
             truth_values_x.view(-1, 1),
             truth_values_y.view(-1, 1)], dim=1)

        # Compute the final prediction as the minimum of the three truth values
        predictions_truth_values, _ = torch.min(symbols_truth_values, 1)

        return predictions_truth_values, g_matrix[chosen_symbols_x, chosen_symbols_y]
