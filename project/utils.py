import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from sklearn import metrics
from torch.utils.data.sampler import WeightedRandomSampler


def calc_metrics(labels, predictions, verbose=False):
    """Calculates evaluation metrics.

    Parameters
    ----------
    labels : list
        ground truth labels in form [0, 1, 1]
    predictions : list
        model predictions in form [0, 1, 2]
    verbose : bool, optional
        whether to print metrics, by default False

    Returns
    -------
    dict
        evaluation metrics
    """

    confusion_matrix = metrics.confusion_matrix(labels, predictions)
    accuracy = metrics.accuracy_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions, average="weighted")
    class_recall = metrics.recall_score(labels, predictions, average=None)
    precision = metrics.precision_score(labels, predictions, average="weighted")
    class_precision = metrics.precision_score(labels, predictions, average=None)

    eval_metrics = {
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "recall": recall,
        "class_recall": class_recall,
        "precision": precision,
        "class_precision": class_precision,
    }

    if verbose:
        print("--- EVAL METRICS ---")
        print(eval_metrics)

    return eval_metrics


def plot_dataset(dataset, MEAN, STD, n=6):
    # retrieve random images from dataset
    choice = np.random.randint(len(dataset), size=n)
    subset = Subset(dataset, choice)
    images = [x[0] for x in subset]
    labels = [x[1] for x in subset]

    # denormalize for visualization
    denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
    images = [denormalization(i) for i in images]

    # make grid and plot
    grid = utils.make_grid(images)
    label_string = []
    for label in labels:
        if "0":
            label_string.append("normal")
        elif "1":
            label_string.append("pneumonia")
        elif "2":
            label_string.append("COVID-19")

    title = "Labels are: " + str(label_string)
    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.show()


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def get_train_sampler(dataset, indices):

    # get labels in subset for rebalancing
    labels = [dataset.targets[i] for i in indices]

    # configure sampler to rebalance training set
    weights = 1 / torch.Tensor([labels.count(0), labels.count(1), labels.count(2),])
    sample_weights = weights[labels]
    num_samples = len(indices)
    sampler = WeightedRandomSampler(sample_weights, num_samples)

    return sampler, weights
