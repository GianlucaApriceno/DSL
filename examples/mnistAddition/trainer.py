import time
import torch
from utils import *
# TODO: remove the following unused libraries: random, numpy, os
import random
import numpy as np
import os


def train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, run=0, device='cpu',
          verbose=20, verbose_conf=10, nn2=None):
    """
        Train the model and evaluate accuracy at defined intervals.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        optimizer : torch.optim.Optimizer
            Optimizer used for updating model weights.
        loss : torch.nn.Module
            The loss function used for training (e.g., Binary Cross Entropy or custom loss).
        train_loader : DataLoader
            DataLoader for the training dataset, providing batches of training samples.
        test_loader : DataLoader
            # TODO: remove test_loader?
            DataLoader for the test dataset (not directly used in this function).
        nn : torch.nn.Module
            An auxiliary neural network used for testing MNIST performance.
        mnist_test_data : Dataset
            The MNIST test dataset for evaluation purposes.
        e : int
            Current epoch number.
        run : int, optional
            A number representing the current training run, useful for logging (default is 0).
        device : str, optional
            The device to run training on, either 'cpu' or 'cuda' (default is 'cpu').
        verbose : int, optional
            Print training progress after every `verbose` epochs (default is 20).
        verbose_conf : int, optional
            Print confusion matrix and MNIST test results after every `verbose_conf` epochs (default is 10).
        nn2 : torch.nn.Module, optional
            A second auxiliary neural network for testing (default is None).

        Returns
        -------
        accuracy_train : float
            The training accuracy on the sum task at the end of the epoch.
     """

    epoch_start = time.time()
    for i, (x, y, l) in enumerate(train_loader):

        # Move data to the specified device
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)

        # Extract the target labels
        _, labels = torch.max(torch.squeeze(l), 1)
        optimizer.zero_grad()

        truth_values, prediction = model(x, y)

        # Compute model accuracy for the batch
        model_labels = torch.where(torch.eq(labels, prediction), 1.0, 0.0).view(-1)

        # Compute the loss function
        s_loss = loss(torch.logit(truth_values, 0.0001), model_labels)

        s_loss.backward()
        optimizer.step()
    accuracy_train = test_sum(model, train_loader, device=device)

    if e % verbose == 0 and e > 0:
        print(f'End of epoch {e}')
        print('Epoch time: ', time.time() - epoch_start)

        print(f'Accuracy in sum task. Train: {accuracy_train}')

    # Evaluate the MNIST confusion matrix at intervals defined by verbose_conf
    if (e % verbose_conf == 0 and e > 0):
        test_MNIST(nn, mnist_test_data, e, 0, run, n_digits=10, device=device)
        if nn2 is not None:
            test_MNIST(nn2, mnist_test_data, e, 0, run, n_digits=10, device=device)
    return accuracy_train

