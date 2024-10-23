import argparse
import os
import pickle
import random
from tqdm import tqdm
from utils import *

import numpy as np
import optuna
import torch

import madgrad
from examples.mnistAddition.dataset_generation import *
from examples.mnistAddition.models import *
from examples.mnistAddition.trainer import *


def visual_addition(args):
    """
        Visualize the MNIST addition task by generating confusion matrices and rule matrices.

        Parameters
        ----------
        args : Command-line arguments passed to the function. Includes parameters like:
            - eps_sym: Epsilon value for symbols.
            - eps_rul: Epsilon value for rules.
            - lr: Learning rate.
            - ckpt: Path to the model checkpoint.

        Returns
        -------
        None
    """

    # TODO: remove unused variables: EXPERIMENT, NUM_EXPERIMENT, LR
    # Setting up experiment parameters
    EXPERIMENT = 'mnistAddition'
    NUM_EXPERIMENT = 1
    DEVICE = 'cpu'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr
    LOAD_CKPT = args.ckpt

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Load model and data
    nn = MNIST_Net().to(DEVICE)
    model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE)
    model.load_state_dict(torch.load(LOAD_CKPT))
    model.eval()

    # Extract rules and generate datasets
    rules = model.get_rules_matrix(eval=True)[1].squeeze()
    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)

    # Compute confusion matrices and visualize
    confusion = test_MNIST_visual(nn, mnist_test_data, n_digits=10, device=DEVICE)
    confusion_swapped, p = swap_conf(confusion)
    rules_swapped = swap_rules(rules.squeeze(), p)

    # Visualization
    visualize_confusion(confusion, 'Original Confusion Matrix')
    visualize_confusion(confusion_swapped, 'Swapped Confusion Matrix')
    visualize_rules(rules, 'Original Rules Matrix')
    visualize_rules(rules_swapped, 'Swapped Rules Matrix')

    return


def experiment_eval(args=None):
    """
        Run MNIST addition experiment with specified training configurations.

        Parameters
        ----------
        args : argparse.Namespace, optional
            Command-line arguments passed to the function. Includes parameters like:
            - eps_sym: Epsilon value for ε-greedy policy for symbols selection
            - eps_rul: Epsilon value for ε-greedy policy for rules selection
            - lr: Learning rate.

        Returns
        -------
        np.mean(acc_test_np) : float
            The mean accuracy of the model on the test set after multiple experiment runs.
    """

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Setting up experiment parameters
    EXPERIMENT = 'mnistAddition'
    NUM_EXPERIMENT = 5
    DEVICE = 'cpu' #TODO GPU is much faster, but scatter_add_cuda_kernel wasn't implemented in a deterministic way. Thus on GPU exp is not exactly reproducible (issue here: https://discuss.pytorch.org/t/runtimeerror-scatter-add-cuda-kernel-does-not-have-a-deterministic-implementation/132290)
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 50
    EPOCHS = 100
    EPSILON_SYMBOLS = args.eps_sym
    EPSILON_RULES = args.eps_rul
    LR = args.lr

    # Load dataset
    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)

    accuracy_test_results = []

    # Training loop over multiple runs
    for num_exp in range(NUM_EXPERIMENT):
        print('Starting training number ', num_exp)

        # Initialize model, optimizer and loss
        nn = MNIST_Net().to(DEVICE)
        model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE).to(DEVICE)
        optimizer = madgrad.MADGRAD(
            [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)

        loss = torch.nn.BCEWithLogitsLoss()

        # Print initial accuracy before training
        # TODO: test on test loader not training
        print('Accuracy in sum task (before training):' + str(test_sum(model, train_loader, DEVICE)))

        # Training loop for each epoch
        for e in tqdm(range(EPOCHS)):
            train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, run=num_exp, device=DEVICE)

            # Save model checkpoint
            if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
                if not os.path.exists('./experiments/'):
                    os.mkdir('./experiments/')
                if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                    os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
                torch.save(model.state_dict(),
                           './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))

        # Test the trained model
        accuracy_test = test_sum(model, test_loader, device=DEVICE)
        accuracy_test_results.append(accuracy_test.cpu().numpy())

    acc_test_np = np.array(accuracy_test_results)

    # Print mean and std of accuracy over the test set
    print('Experiment is over. After {} runs on test set we obtained: \n Mean: {}\n Std: {}'.format(
        NUM_EXPERIMENT, np.mean(acc_test_np), np.std(acc_test_np)))

    return np.mean(acc_test_np)


def experiment_optuna(trial=None):
    """
        Run hyperparameter optimization for the MNIST addition task using Optuna.

        Parameters
        ----------
        trial : optuna.Trial, optional
            An Optuna trial object that provides the current set of hyperparameters to test.

        Returns
        -------
        accuracy_train : float
            The final training accuracy after running the experiment with the suggested hyperparameters.
    """

    # Set experiment parameters
    EXPERIMENT = 'mnistAddition'
    # TODO: add check if gpu is available, otherwise run on cpu
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    BATCH_SIZE_VAL = 1000
    CKPT_SAVE = 50
    EPOCHS = 200

    # Suggest hyperparameters range
    EPSILON_SYMBOLS = trial.suggest_float('EPSILON_SYMBOLS', 0.0, 0.8)
    EPSILON_RULES = trial.suggest_float('EPSILON_RULES', 0.0, 0.8)
    LR = trial.suggest_float('LR', 5e-4, 5e-1, log=True)

    # Load dataset
    train_loader, test_loader, mnist_test_data_tsne, mnist_test_data = get_dataset(BATCH_SIZE, BATCH_SIZE_VAL)

    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Initialize model, optimizer, and loss function
    nn = MNIST_Net().to(DEVICE)
    model = MNISTSumModel(nn, EPSILON_SYMBOLS, EPSILON_RULES, device=DEVICE).to(DEVICE)

    optimizer = madgrad.MADGRAD(
        [{'params': list(model.parameters())[:1]}, {'params': list(model.parameters())[1:], 'lr': 1e-3}], lr=LR)
    loss = torch.nn.BCEWithLogitsLoss()

    # Print initial accuracy
    print('Accuracy in sum task (before training):' + str(test_sum(model, train_loader, DEVICE)))

    # Training loop with hyperparameter tuning
    for e in tqdm(range(EPOCHS)):
        accuracy = train(model, optimizer, loss, train_loader, test_loader, nn, mnist_test_data, e, run=0, device=DEVICE)

        # Save checkpoint
        if CKPT_SAVE > 0 and e > 0 and e % CKPT_SAVE == 0:
            if not os.path.exists('./experiments/'):
                os.mkdir('./experiments/')
            if not os.path.exists('./experiments/ckpt_{}'.format(EXPERIMENT)):
                os.mkdir('./experiments/ckpt_{}'.format(EXPERIMENT))
            torch.save(model.state_dict(),
                       './experiments/ckpt_{}/ckpt.{}.pth'.format(EXPERIMENT, e))
        trial.report(accuracy, e)

        # Handle trial pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Compute final training accuracy
    accuracy_train = test_sum(model, train_loader, device=DEVICE)

    print('Experiment is over. Accuracy: {}'.format(accuracy_train))

    return accuracy_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Addition Experiment')
    parser.add_argument('-o', '--optuna', action='store_true', default=False,
                        help='Enable hyperparameter optimization using Optuna.')
    parser.add_argument('-e', '--eval', action='store_true', default=False,
                        help='Enable evaluation mode for visualizing the addition task.')
    parser.add_argument('-c', '--ckpt', default='/home/tc94/Desktop/SQ-Learning/experiments/ckpt_mnistAddition/ckpt_final.pth',
                        help='Specify the path to the model checkpoint (default: specified path).')
    parser.add_argument('-lr', '--lr', default=0.11639833786002995,
                        help='Set the learning rate for training (default: 0.1164).')
    parser.add_argument('-es', '--eps_sym', default=0.2807344052335263,
                        help='Epsilon value for ε-greedy policy for symbols selection (default: 0.2807).')
    parser.add_argument('-er', '--eps_rul', default=0.1077119516324264,
                        help='Epsilon value for ε-greedy policy for rules selection (default: 0.1077).')
    args = parser.parse_args()

    if args.optuna:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(experiment_optuna, n_trials=30)
        pickle.dump(study, open('./studies/mnist_addition_optuna.pkl', 'wb'))
    elif not args.optuna and not args.eval:
        experiment_eval(args=args)
    elif args.eval:
        visual_addition(args=args)
    else:
        print('There is an error in your configuration!. ')
