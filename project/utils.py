from sklearn import metrics

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

    eval_metrics = {"confusion_matrix": confusion_matrix, 
                    "accuracy": accuracy,
                    "recall": recall, 
                    "class_recall": class_recall, 
                    "precision": precision,
                    "class_precision": class_precision}

    if verbose:
        print("--- EVAL METRICS ---")
        print(eval_metrics)

    return eval_metrics