from typing import Union, List, Optional, Callable, Tuple, Dict
import numpy as np
from sklearn.metrics import f1_score, adjusted_rand_score
import sklearn.metrics
from typing import Union
Metrics = Dict[str,float]


def compute_ari(est_cps, true_cps, T):
    """
    Compute ARI score from two arrays of change-points
    :param labels: np.ndarray
    :param true_labels: np.ndarray
    :param T: int
    :return:
    """

    labels, true_labels = np.zeros(T, dtype=int), np.zeros(T, dtype=int)
    for cp in est_cps:
        labels[cp:] = labels[cp:] + 1
    for cp in true_cps:
        true_labels[cp:] = true_labels[cp:] + 1

    return adjusted_rand_score(true_labels, labels)



def find_best_threshold(
        score,
        target,
        metric = 'adjusted_f1'):

    smin = np.min(score)
    smax = np.max(score)

    threshold_values = np.linspace(smin, smax, 100)
    metric_values = np.zeros_like(threshold_values)

    for i,t in enumerate(threshold_values):

        if metric == 'adjusted_f1':
            metric_values[i] = binary_metrics_adj(
                score = score,
                target = target,
                threshold = t,
                adjust_predicts_fun = adjust_predicts_donut,
                only_f1 = True,
            )
        else:
            raise ValueError("Not yet implemented")

    i_best = np.argmax(metric_values)
    threshold_best = threshold_values[i_best]

    threshold_best_same_f1 = threshold_values[metric_values == metric_values[i_best]]
    if len(threshold_best_same_f1 ) >1:
        idx = np.argmin( np.abs( threshold_best_same_f1 -np.median(threshold_best_same_f1)) )
        threshold_best = threshold_best_same_f1[idx]

    if metric == 'adjusted_f1':
        metrics_best = binary_metrics_adj(
            score = score,
            target = target,
            threshold = threshold_best,
            adjust_predicts_fun = adjust_predicts_donut,
            only_f1 = True,
        )

        return threshold_best, metrics_best

    else:
        raise ValueError("Not yet implemented")





def adjust_predicts_donut(
        pred_label: Union[list, np.array],
        target: Union[list, np.array],
        *args, **kwargs) -> np.array:
    """Calculate adjusted predicted labels.

    Label adjustment mechanism used in [Xu et al, 2018].

    Args:
        pred_label : 1d vector with the predicted binary labels.
        target : 1d vector with true binary labels.

    Returns:
        pred_label_adj: 1d vector with the adjusted binary labels.

    Reference:
        Xu et al. 2018, Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications.
    """

    if isinstance(pred_label, list):
        pred_label = np.array(pred_label)
    if isinstance(target, list):
        target = np.array(target)

    assert pred_label.ndim == 1
    assert target.ndim == 1

    n_pred = len(pred_label)
    assert len(target) == n_pred, "score and target must have the same length"

    pred_label_adj = pred_label.copy()

    anomaly_state = 0
    anomaly_count = 0
    for i in range(n_pred):
        if (target[i] == 1) and (pred_label_adj[i] == 1) and not (anomaly_state == 1):
            anomaly_state = 1
            anomaly_count += 1
            for j in range(i, 0, -1):
                if target[j] == 1:
                    pred_label_adj[j] = 1
                else:
                    break
        elif not (target[i] == 1):
            anomaly_state = 0

        if anomaly_state == 1:
            pred_label_adj[i] = 1

    return pred_label_adj


def adjust_predicts_multiple_ts(
        pred_label: List[np.array],
        target: List[np.array],
        adjust_predicts_fun: Optional[Callable] = [None, adjust_predicts_donut, adjust_predicts_donut][1],
        *args, **kwargs) -> List[np.array]:
    """Calculate adjusted predicted labels on a set of time series.

    Args:
        pred_label : Predicted binary labels. A list with 1D vectors (possibly of different length).
        target : True binary labels. Same structure as pred_label.
        adjust_predicts_fun : Function for label adjustment on univariate time series.
            By default (None) the search is performed with no label adjustment.
            If specified, adjust_predicts_donut is currently supported, but other methos can be defined.
    Returns:
        pred_label_adj: Adjusted binary labels. Same structure as pred_label.
    """

    assert isinstance(pred_label, list)
    assert type(pred_label) == type(target)
    assert len(pred_label) == len(target)

    if adjust_predicts_fun is None:
        return pred_label

    N = len(pred_label)

    pred_label_adj = []
    for j in range(N):
        pred_label_adj.append(
            adjust_predicts_fun(pred_label[j], target[j])
        )
    return pred_label_adj


def binary_metrics(
        pred_label,
        target,
        only_f1: bool = False,
) -> Union[Metrics, float]:
    """Compute a number of relevant metrics for binary classification.

    Args:
        pred_label : 1d array-like, vector with the predicted binary labels.
        target : 1d array-like, vector with true binary labels.
        only_f1 : If True, the function returns an scalar with the adjusted f1 score.

    Output:
        metrics : Dictionary with collection of metrics for binary classification.
    """

    metrics = {}

    metrics['f1'] = sklearn.metrics.f1_score(y_true=target, y_pred=pred_label)
    if only_f1:
        return metrics['f1']
    metrics['f2'] = sklearn.metrics.fbeta_score(y_true=target, y_pred=pred_label, beta=2)
    metrics['f0.5'] = sklearn.metrics.fbeta_score(y_true=target, y_pred=pred_label, beta=1 / 2)
    metrics['precision'] = sklearn.metrics.precision_score(y_true=target, y_pred=pred_label)

    metrics['recall'] = sklearn.metrics.recall_score(y_true=target, y_pred=pred_label)

    conf_mat = sklearn.metrics.confusion_matrix(y_true=target, y_pred=pred_label)
    if conf_mat.shape != (2, 2):
        TP = np.sum(pred_label * target)
        TN = np.sum((1 - pred_label) * (1 - target))
        FP = np.sum(pred_label * (1 - target))
        FN = np.sum((1 - pred_label) * target)
        conf_mat = np.array([[TN, FP], [FN, TP]])
    metrics['TN'] = conf_mat[0, 0]
    metrics['FN'] = conf_mat[1, 0]
    metrics['TP'] = conf_mat[1, 1]
    metrics['FP'] = conf_mat[0, 1]

    return metrics


def binary_metrics_adj(
        score: Union[np.array, List[np.array]],
        target: Union[np.array, List[np.array]],
        threshold: float = 0.5,
        adjust_predicts_fun: Optional[Callable] = [None, adjust_predicts_donut, adjust_predicts_donut][1],
        only_f1: bool = False,
) -> Union[Dict, float]:
    """Compute a number of relevant metrics for binary classification.

    Args:
        score : Predicted (continuous) scores.
            If 1D array, a vector of scores for a single time series.
            If 2D array, a matrix of scores for multiple time series, of shape (time, batch)
            If List, must contain 1D vectors as elements (possibly of different length)
        target : True binary labels. Must share the same data structure and shape of score.
        threshold : values in which score>threshold are considered as predicting target=1
        adjust_predicts_fun : Function for label adjustment on univariate time series.
            By default (None) the search is performed with no label adjustment.
            If specified, adjust_predicts_donut is currently supported.
        only_f1 : If True, the function returns an scalar with the adjusted f1 score.

    Output:
        metrics : Dictionary with collection of metrics for binary classification,
            calculated over all pairs (score, target).
    """
    assert type(score) == type(target)

    # transform array to list of time series
    if isinstance(score, np.ndarray):
        assert score.shape == target.shape

        if score.ndim == 1:
            # If score and target are a 1-D array, expand it to a matrix with shape (time, batch=1)
            score = np.expand_dims(score, axis=1)
            target = np.expand_dims(target, axis=1)

        # Create a list of 1-D scores, one per element in the batch dimension (column)
        score = [score[:, i] for i in range(score.shape[1])]
        target = [target[:, i] for i in range(target.shape[1])]

    assert isinstance(score, List)
    assert all(score_i.ndim == 1 for score_i in score)
    assert all(target_i.ndim == 1 for target_i in target)

    N = len(score)
    pred_label = [1 * (score[i] > threshold) for i in range(N)]

    pred_label = adjust_predicts_multiple_ts(
        pred_label=pred_label,
        target=target,
        threshold=threshold,
        adjust_predicts_fun=adjust_predicts_fun,
    )

    out = binary_metrics(
        pred_label=np.concatenate(pred_label, axis=0),
        target=np.concatenate(target, axis=0),
        only_f1=only_f1,
    )

    return out


def best_f1_search_grid(
        score: Union[np.array, List[np.array]],
        target: Union[np.array, List[np.array]],
        adjust_predicts_fun: Optional[Callable] = [None, adjust_predicts_donut, adjust_predicts_donut][1],
        threshold_values: np.array = np.array(0.5),
        verbose = False,
        display_freq = 1,
) -> Tuple[Metrics ,float]:
    """Find the best-f1 metric along threshold_values.

    Args:
        score : Predicted (continuous) scores.
            If 1D array, a vector of scores for a single time series.
            If 2D array, a matrix of scores for multiple time series, of shape (time, batch)
            If List, must contain elements of 1D vectors (possibly of different length)
        target : True binary labels. Must share the same data structure and shape of score.
        adjust_predicts_fun : Function for label adjustment on univariate time series.
            By default (None) the search is performed with no label adjustment.
            If specified, adjust_predicts_donut is currently supported.
        threshold_values : threshold values used to evaluate the f1 metric.
    Returns:
        metrics : Dictionary with the binary metrics for the best threshold.
        threshold : the `threshold` with the highest f1 (within the range of search).
    """

    metrics_best = dict(f1=-1., precision=-1., recall=-1.)
    threshold_best = None

    threshold_values = np.array(threshold_values).copy()
    threshold_values.sort()
    f1_values = np.zeros_like(threshold_values)

    for i, threshold_i in enumerate(threshold_values):
        f1_values[i] = binary_metrics_adj(
            score = score,
            target = target,
            threshold = threshold_i,
            adjust_predicts_fun = adjust_predicts_fun,
            only_f1 = True,
        )

    i_best = np.argmax(f1_values)
    threshold_best = threshold_values[i_best]

    threshold_best_same_f1 = threshold_values[f1_values == f1_values[i_best]]
    if len(threshold_best_same_f1 ) >1:
        idx = np.argmin( np.abs( threshold_best_same_f1 -np.median(threshold_best_same_f1)) )
        threshold_best = threshold_best_same_f1[idx]

    metrics_best = binary_metrics_adj(
        score = score,
        target = target,
        threshold = threshold_best,
        adjust_predicts_fun = adjust_predicts_fun,
        only_f1 = False,
    )

    return metrics_best, threshold_best
