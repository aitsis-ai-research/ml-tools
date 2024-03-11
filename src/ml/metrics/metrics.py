import numpy as np
from typing import Union, List

def accuracy(y_pred: list, y_actual: list):
    intersection_count = len(set(y_pred) & set(y_actual))
    return intersection_count / len(y_pred)


def find_true_positive(y_pred: list, y_actual: list):
    return len(set(y_pred) & set(y_actual))

def find_false_positive(y_pred: list, y_actual: list):
    return len(y_pred) - find_true_positive(y_pred, y_actual)


def find_false_negative(y_pred: list, y_actual: list):
    return len(y_actual) - find_true_positive(y_actual, y_pred)


def precision(y_pred: list, y_actual: list):
    tp = find_true_positive(y_pred, y_actual)
    fp = find_false_positive(y_pred, y_actual)
    return tp / (tp + fp)

def recall(y_pred: list, y_actual: list):
    tp = find_true_positive(y_pred, y_actual)
    fn = find_false_negative(y_pred, y_actual)
    return tp / (tp + fn)


def f1_score(y_pred: list, y_actual: list):
    precision_score = precision(y_pred, y_actual)
    recall_score = recall(y_pred, y_actual)
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)


def cohens_d(np_class1: np.ndarray, np_class2: np.ndarray):
    """
    Calculate the Cohen's d between two classes.

    Args:
        class1 (numpy.ndarray): Class 1.
        class2 (numpy.ndarray): Class 2.

    Returns:
        float: Cohen's d.
    """
    if np_class1.shape[0] != np_class2.shape[0]:
        raise ValueError("Class dimensions must be equal")
    
    class1_mean = np.mean(np_class1, axis=0)
    class2_mean = np.mean(np_class2, axis=0)
    
    class1_std = np.std(np_class1, axis=0)
    class2_std = np.std(np_class2, axis=0)
    
    spool = np.sqrt(((np_class1.shape[0] - 1) *np.square(class1_std) + (np_class2.shape[0] -1 ) * np.square(class2_std)) / (np_class1.shape[0] + np_class2.shape[0] - 2))
    
    d = np.abs(class1_mean - class2_mean) / spool    
    return d


__all__ = ["accuracy", "find_true_positive", "find_false_positive", "find_false_negative", "precision", "recall", "f1_score", "cohens_d"]