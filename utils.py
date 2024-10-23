import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from torch.autograd import Variable


#### MNIST ADDITION #####

def swap_conf(confusion):
    """
        Swaps the columns of a confusion matrix based on the predicted labels.

        This function takes a confusion matrix as input, determines the predicted labels for each true class,
        and rearranges the columns of the confusion matrix according to these predictions. The resulting
        matrix reflects how the predicted labels align with the actual labels.

        Parameters:
        -----------
        confusion : numpy.ndarray
            A confusion matrix of shape (n_classes, n_classes), where the entry at position (i, j) represents
            the number of instances of class i that were predicted as class j.

        Returns:
        --------
        swapped_confusion : numpy.ndarray
            A rearranged confusion matrix where the columns have been swapped based on the predicted classes.

        p : torch.Tensor
            A tensor containing the predicted classes for each true class, representing the column indices
            of the maximum values in each row of the original confusion matrix.
    """
    _, p = torch.max(torch.tensor(confusion.astype(np.float32)), 1)
    return torch.tensor(confusion.astype(np.int32))[:, p].cpu().numpy(), p


def swap_rules(rules, p):
    """
        Rearranges the rows and columns of a 2D tensor based on provided indices.

        This function takes as input a tensor representing a set of rules and a set of indices. It swaps the rows and
        columns of the tensor according to the provided indices to create a new tensor where the specified order is reflected.

        Parameters:
        -----------
        rules : torch.Tensor
            A 2D tensor representing a set of rules. This could be any matrix where rearranging the order of rows and
            columns is meaningful.

        p : torch.Tensor
            A 1D tensor containing the indices to reorder the rows and columns of the `rules` tensor. The same
            indices are used for both rows and columns to maintain the structure of the data.

        Returns:
        --------
        swapped_rules : torch.Tensor
            A 2D tensor where the rows and columns of the original `rules` tensor have been rearranged according
            to the specified indices in `p`.
    """
    return torch.tensor(rules)[p, :][:, p]


def test_MNIST(model, dataset, epoch, folder, num_exp, n_digits, device='cpu'):
    # TODO: should we remove epoch, folder and num_exp?
    """
        Generates a confusion matrix to analyze the model's performance.

        Parameters:
        -----------
        model : torch.nn.Module
            The neural network model to evaluate.

        dataset : DataLoader
            The dataset to evaluate, containing pairs of (input_data, label).

        n_digits : int
            The number of unique classes (or digits) in the dataset, used to define the size of the confusion matrix.

        device : str, optional
            The device on which to run the evaluation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
        --------
        confusion : numpy.ndarray
            A confusion matrix of shape (n_digits, n_digits), where the rows correspond to the true labels and the
            columns correspond to the predicted labels.

    """
    confusion = np.zeros((n_digits, n_digits), dtype=np.uint32)  # First index actual, second index predicted
    N = 0
    for d, l in dataset:
        if l < n_digits:
            N += 1
            d = Variable(d.unsqueeze(0))
            d = d.to(device)
            with torch.no_grad():
                outputs, _ = model(d)
                _, out = torch.max(outputs.data, 1)
                out = out.to(device)
                c = int(out.squeeze())
            confusion[l, c] += 1
    print()
    print(confusion)
    return confusion #, p


def test_MNIST_visual(model, dataset, n_digits, device='cpu'):
    # TODO: duplicated code?
    """
        Generates a confusion matrix to analyze the model's performance.

        Parameters:
        -----------
        model : torch.nn.Module
            The neural network model to evaluate.

        dataset : DataLoader
            The dataset to evaluate, containing pairs of (input_data, label).

        n_digits : int
            The number of unique classes (or digits) in the dataset, used to define the size of the confusion matrix.

        device : str, optional
            The device on which to run the evaluation (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
        --------
        confusion : numpy.ndarray
            A confusion matrix of shape (n_digits, n_digits), where the rows correspond to the true labels and the
            columns correspond to the predicted labels.

    """
    confusion = np.zeros((n_digits, n_digits), dtype=np.uint32)  # First index actual, second index predicted
    N = 0
    for d, l in dataset:
        if l < n_digits:
            N += 1
            d = Variable(d.unsqueeze(0))
            d = d.to(device)
            with torch.no_grad():
                outputs, _ = model(d)
                _, out = torch.max(outputs.data, 1)
                out = out.to(device)
                c = int(out.squeeze())
            confusion[l, c] += 1
    print()
    print(confusion)
    return confusion #, p


def test_sum(model, dataloader, device='cpu'):
    """
        Evaluates the accuracy of a trained model on a dataset for a task where the model predicts the sum of two input images' labels.

        Parameters:
        -----------
        model : torch.nn.Module
            The (trained) model returning the predicted sum of the two digits.
        dataloader : DataLoader
            The DataLoader object containing the dataset. It return batches of tuples, where each tuple contains two input images (`x`, `y`) and a label representing the sum of
            their corresponding class labels (`l`).
        device : str, optional
            The device on which to run the model and data processing (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
        --------
        accuracy : float
            The accuracy of the model on the given dataset.
    """
    x, y, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, eval=True)
        prediction.to(device)
        _, label = torch.max(torch.squeeze(l), 1)

    return torch.sum(label.to(device) == torch.squeeze(prediction)).float() / label.shape[0]


def visualize_confusion(confusion, name, indices="0123456789"):
    """
       Visualizes a confusion matrix as a heatmap and saves it as a PNG image.

       This function takes a confusion matrix and generates a visual representation using a heatmap.
       It also saves the heatmap as a PNG file with a specified name.

       Parameters:
       -----------
       confusion : torch.Tensor
           A confusion matrix of shape (n_classes, n_classes), where each entry (i, j) indicates
           the number of instances of class i that were predicted as class j.

       name : str
           The name used to save the generated heatmap image. The image will be saved in the
           './visualizations/' directory with the format '<name>_confusion.png'.

       indices : str, optional
           A string representing the class labels to be used as both the row and column indices in the heatmap.
           The default is "0123456789", which corresponds to the digits in the MNIST dataset.

       Returns:
       --------
       None
    """
    df_cm = pd.DataFrame(confusion, index=[i for i in indices],
                         columns=[i for i in indices])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=False, annot_kws={'size':15})
    ax.set_xticks([i for i in range(10)])
    ax.set_yticks([i for i in range(10)])
    ax.set_yticklabels([i for i in range(10)], rotation=90)
    ax.set_xticklabels([i for i in range(10)], rotation=0)

    ax.tick_params(axis='both', which='both', length=3, labelsize=15)
    plt.savefig('./visualizations/{}_confusion.png'.format(name))
    plt.close()

    return


def visualize_rules(rules, name):
    """
       Visualizes a rules matrix as a heatmap and saves it as a PNG image.

       This function takes a confusion matrix and generates a visual representation using a heatmap.
       It also saves the heatmap as a PNG file with a specified name.

       Parameters:
       -----------
       rules : torch.Tensor
           A rules matrix of shape (n_classes, n_classes), where each entry rules[i,j] (i, j = 0,...,n_classes-1)
           represents the sum of symbols i and j.

       name : str
           The name used to save the generated heatmap image. The image will be saved in the
           './visualizations/' directory with the format '<name>_rules.png'.

       Returns:
       --------
       None
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    rules=rules.cpu().numpy().squeeze()
    cmap = colors.ListedColormap(['#7b68ee'])

    ax.set_xticks([i for i in range(10)])
    ax.set_yticks([i for i in range(10)])
    ax.set_yticklabels([i for i in range(10)], rotation=90)
    ax.invert_xaxis()
    ax.xaxis.tick_top()

    ax.tick_params(axis='both', which='both', length=0, labelsize=15)
    ax.imshow(rules, vmin=0, vmax=0, cmap=cmap)
    for i in range(10):
        for j in range(10):
            ax.text(i, j, str(rules[i][j]), ha="center", va="center", color="white", fontsize=15)

    plt.savefig('./visualizations/{}_rules.png'.format(name))
    plt.close()

###############################
def F1_compute(N, max_digit, confusion):
    print(confusion)
    F1 = 0
    for nr in range(max_digit):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    return


def accuracy_rules(weights, p):
    rules_matrix_final = np.zeros([10, 10])

    for i in range(10):
        for j in range(10):
            x = p[i]
            y = p[j]
            #print(f'{i} + {j} = {r[x, y]}')
            rules_matrix_final[i, j] = torch.argmax(weights[x, y])
    return


def test_EMNIST(model, emnist_test_data, epoch, folder, num_exp, max_digit=4, device='cpu'):
    confusion = np.zeros((max_digit, max_digit), dtype=np.uint32)  # First index actual, second index predicted
    N = 0
    for d, l in emnist_test_data:
        if l < max_digit:
            N += 1
            d = Variable(d.unsqueeze(0))
            d = d.to(device)
            with torch.no_grad():
                outputs, _ = model(d)
                _, out = torch.max(outputs.data, 1)
                c = int(out.squeeze())
            confusion[l, c] += 1
    print(confusion)
    return confusion,


def test_sum_multi(model, dataloader, squeeze=True, device='cpu'):
    x, y, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, eval=True)
        prediction.to(device)

    if squeeze:
        return torch.sum(torch.all(l.to(device) == torch.squeeze(prediction), dim=1)).float() / l.shape[0]
    else:
        return torch.sum(torch.all(l.to(device) == prediction, dim=1)).float() / l.shape[0]

def test_sum_multi_single(model, dataloader, device='cpu'):
    x, y, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, eval=True)
        prediction.to(device)

    return torch.sum(torch.sum(l.to(device) == torch.squeeze(prediction), dim=1)/l.shape[1]).float() / l.shape[0]



def test_parity(model, dataloader, device='cpu'):
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    _, predictions = model(x, eval=True)

    return torch.sum(predictions.view(-1, 1) == y) / x.shape[0]


def test_visual_parity(model, dataloader, device='cpu'):
    x, y = next(iter(dataloader))
    x = [z.to(device) for z in x]
    y = y.to(device)

    _, predictions = model(x, eval=True)

    return torch.sum(predictions.view(-1) == y.to(device)) / y.shape[0]


def test_multiop(model, dataloader, device='cpu'):
    x, y, a, l = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    a = a.to(device)
    l = l.to(device)

    with torch.no_grad():
        _, prediction = model(x, y, a, eval=True)
        prediction.to(device)
        _, label = torch.max(torch.squeeze(l), 1)
    return torch.sum(label == prediction.view(-1)) / label.shape[0]


def visualize_rules(rules, name):
    fig, ax = plt.subplots(figsize=(7, 7))
    rules=rules.cpu().numpy().squeeze()
    cmap = colors.ListedColormap(['#7b68ee'])

    ax.set_xticks([i for i in range(10)])
    ax.set_yticks([i for i in range(10)])
    ax.set_yticklabels([i for i in range(10)], rotation=90)
    ax.invert_xaxis()
    ax.xaxis.tick_top()

    ax.tick_params(axis='both', which='both', length=0, labelsize=15)
    ax.imshow(rules, vmin=0, vmax=0, cmap=cmap)
    for i in range(10):
        for j in range(10):
            ax.text(i, j, str(rules[i][j]), ha="center", va="center", color="white", fontsize=15)

    plt.savefig('./visualizations/{}_rules.png'.format(name))
    plt.close()