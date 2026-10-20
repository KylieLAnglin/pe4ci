from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def hierarchical_bootstrap(y_true, y_pred, participant_ids, metric_fn, n_bootstraps, random_state):
    random_generator = np.random.default_rng(seed=random_state)

    # pseudo participant with one text in each cell of the confusion matrix
    pseudo_y_true = [1, 0, 0, 1]
    pseudo_y_pred = [1, 0, 1, 0]
    #               [TP, TN, FP, FN]
    pseudo_participant_ids = ["pseudo_participant"] * 4

    y_true_arr = np.concatenate([np.array(y_true), pseudo_y_true])
    y_pred_arr = np.concatenate([np.array(y_pred), pseudo_y_pred])
    participant_ids_arr = np.concatenate([np.array(participant_ids, dtype=str), pseudo_participant_ids])
    unique_participants = np.unique(participant_ids_arr)
    scores = []

    for _ in range(n_bootstraps):
        sampled_participants = random_generator.choice(unique_participants, size=len(unique_participants), replace=True)
        row_indices = []
        for participant_id in sampled_participants:
            rows_for_participant = np.where(participant_ids_arr == participant_id)[0]
            row_indices.extend(random_generator.choice(rows_for_participant, size=len(rows_for_participant), replace=True).tolist())
        row_indices = np.array(row_indices) # bootstrap indices to be scored
        scores.append(metric_fn(y_true_arr[row_indices], y_pred_arr[row_indices])) # send bootstrap for scoring, get score back

    return np.array(scores)


def bootstrap_accuracy(y_true, y_pred, participant_ids, n_bootstraps=1000, random_state=12):
    scores = hierarchical_bootstrap(y_true, y_pred, participant_ids, accuracy_score, n_bootstraps, random_state)
    return scores.std(ddof=1)


def bootstrap_precision(y_true, y_pred, participant_ids, n_bootstraps=1000, random_state=12):
    metric_fn = lambda y_true_sample, y_pred_sample: precision_score(y_true_sample, y_pred_sample, zero_division=0)
    scores = hierarchical_bootstrap(y_true, y_pred, participant_ids, metric_fn, n_bootstraps, random_state)
    return scores.std(ddof=1)


def bootstrap_recall(y_true, y_pred, participant_ids, n_bootstraps=1000, random_state=12):
    metric_fn = lambda y_true_sample, y_pred_sample: recall_score(y_true_sample, y_pred_sample, zero_division=0)
    scores = hierarchical_bootstrap(y_true, y_pred, participant_ids, metric_fn, n_bootstraps, random_state)
    return scores.std(ddof=1)


def bootstrap_f1(y_true, y_pred, participant_ids, n_bootstraps=1000, random_state=12):
    metric_fn = lambda y_true_sample, y_pred_sample: f1_score(y_true_sample, y_pred_sample, zero_division=0)
    scores = hierarchical_bootstrap(y_true, y_pred, participant_ids, metric_fn, n_bootstraps, random_state)
    return scores.std(ddof=1)


def bootstrap_metric_ci(y_true, y_pred, participant_ids, metric_fn, n_bootstraps=1000, random_state=12):
    scores = hierarchical_bootstrap(y_true, y_pred, participant_ids, metric_fn, n_bootstraps, random_state)
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)
