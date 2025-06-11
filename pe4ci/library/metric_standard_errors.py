from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import sys


def bootstrap_accuracy(y_true, y_pred, n_bootstraps=1000, random_state=12):
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        scores.append(accuracy_score(y_true_boot, y_pred_boot))

    scores = np.array(scores)
    return scores.mean(), scores.std(ddof=1)


def bootstrap_precision(y_true, y_pred, n_bootstraps=1000, random_state=12):
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        scores.append(precision_score(y_true_boot, y_pred_boot))

    scores = np.array(scores)
    return scores.mean(), scores.std(ddof=1)


def bootstrap_recall(y_true, y_pred, n_bootstraps=1000, random_state=12):
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        scores.append(recall_score(y_true_boot, y_pred_boot))

    scores = np.array(scores)
    return scores.mean(), scores.std(ddof=1)


def bootstrap_f1(y_true, y_pred, n_bootstraps=1000, random_state=12):
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        scores.append(f1_score(y_true_boot, y_pred_boot))

    scores = np.array(scores)
    return scores.mean(), scores.std(ddof=1)


def bootstrap_f1_micro(y_true, y_pred, n_bootstraps=1000, random_state=12):
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        scores.append(f1_score(y_true_boot, y_pred_boot, average="micro"))

    scores = np.array(scores)
    return scores.mean(), scores.std(ddof=1)


def bootstrap_f1_macro(y_true, y_pred, n_bootstraps=1000, random_state=12):
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        scores.append(f1_score(y_true_boot, y_pred_boot, average="macro"))

    scores = np.array(scores)
    return scores.mean(), scores.std(ddof=1)


#
