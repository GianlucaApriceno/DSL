import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def next_example(dataset, i, list_values):
    """
        Generate a pair of samples and a corresponding label based on class sums.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing image-label pairs.
        i : iterator
            An iterator over shuffled indices of the dataset.
        list_values : list
            A list tracking the occurrence of each class label in the dataset.

        Returns
        -------
        x : torch.Tensor
            The first image sample from the dataset.
        y : torch.Tensor
            The second image sample from the dataset.
        label : torch.Tensor
            A one-hot encoded tensor indicating the sum of the class labels of the two samples.
    """
    x, y = next(i), next(i)
    (x, c1), (y, c2) = dataset[x], dataset[y]
    list_values[c1] += 1
    list_values[c2] += 1
    s__ = c1 + c2
    label = [0.0] * 19
    label[s__] = 1.0
    label = torch.tensor([label])

    return x, y, label


def gather_examples(dataset, ):
    """
        Collect paired examples from the dataset along with their corresponding labels.

        Parameters
        ----------
        dataset : Dataset
            The dataset from which samples are gathered.

        Returns
        -------
        examples : list
            A list of tuples where each tuple contains two samples and their corresponding label.
    """
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    list_values = [0 for _ in range(10)]
    i = iter(i)
    while(True):
        try:
            examples.append(next_example(dataset, i, list_values))
        except StopIteration:
            break
    print(list_values)
    return examples


class MNISTDataset(Dataset):
    """
        A custom dataset wrapper for the MNIST dataset.

        Parameters
        ----------
        dataset : list
            A list containing image-label pairs for the MNIST dataset.

        Methods
        -------
        __len__()
            Returns the total number of samples in the dataset.
        __getitem__(idx)
            Retrieves a specific sample at the given index.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        """
            Return the number of samples in the dataset.

            Returns
            -------
            int
                The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
           Get a specific sample from the dataset by index.

           Parameters
           ----------
           idx : int
               Index of the sample to retrieve.

           Returns
           -------
           tuple
               A tuple containing the image and label for the sample.
        """
        return self.dataset[idx]


def dataloader(dataset, batch_size=32):
    """
        Create a DataLoader for batching and shuffling the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to load.
        batch_size : int, optional
            Number of samples per batch (default is 32).

        Returns
        -------
        DataLoader
            A DataLoader object for iterating over the dataset in batches.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def get_dataset(batch_size, batch_size_val):
    """
        Download, process, and return DataLoaders for MNIST Addition task.

        Parameters
        ----------
        batch_size : int
            Batch size for the training data loader.
        batch_size_val : int
            Batch size for the validation/testing data loader.

        Returns
        -------
        train_loader : DataLoader
            DataLoader for the processed training dataset.
        test_loader : DataLoader
            DataLoader for the processed testing dataset.
        mnist_test_data_tsne : DataLoader
            DataLoader for the raw testing dataset, useful for t-SNE visualization.
        mnist_test_data : Dataset
            The raw testing dataset.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    # Download and transform the MNIST training and test data
    mnist_train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    mnist_test_data = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

    # Generate paired examples for training and test datasets
    train_data = gather_examples(mnist_train_data)
    test_data = gather_examples(mnist_test_data)

    # Create DataLoader for raw test data
    mnist_test_data_tsne = dataloader(mnist_test_data, batch_size_val)

    # Create DataLoader for processed training and testing data
    train_loader = dataloader(MNISTDataset(train_data), batch_size)
    test_loader = dataloader(MNISTDataset(test_data), batch_size_val)

    return train_loader, test_loader, mnist_test_data_tsne, mnist_test_data
