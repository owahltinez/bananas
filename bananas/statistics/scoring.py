"""
These are constants and functions used for scoring a set of predictions against ground truth. ML
frameworks are encouraged to override these functions with their own if they can be implemented in a
more efficient way.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Tuple
from statistics import variance
from .basic import almost_zero, mean
from ..utils.arrays import argmax, equal_nested, check_equal_shape

ScoringFunctionImpl = Callable[[List, List], float]


@dataclass
class TFPN(object):
    """ Helper data class used to pass results around """

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


class ScoringFunction(IntEnum):
    """ Enum declaring different types of scoring function """

    R2 = 0
    ACCURACY = 1
    PRECISION = 2
    RECALL = 3
    F1 = 4
    AREA_UNDER_ROC = 5

    @staticmethod
    def create(kind: "ScoringFunction") -> ScoringFunctionImpl:
        """
        Get an instance of the requested loss function, framework dependent implementation.

        Parameters
        ----------
        kind : ScoringFunction
            The specific type of scoring function that an instance is being created for
        """
        if kind == ScoringFunction.R2:
            return score_r2
        if kind == ScoringFunction.ACCURACY:
            return score_accuracy
        if kind == ScoringFunction.PRECISION:
            return score_precision
        if kind == ScoringFunction.RECALL:
            return score_recall
        if kind == ScoringFunction.F1:
            return score_f1
        if kind == ScoringFunction.AREA_UNDER_ROC:
            return score_auroc

        raise NotImplementedError("This method should be overridden")


def score_r2(y_true: List[float], y_pred: List[float]) -> float:
    """
    Coefficient of determination score, should range between 0 and 1 in most cases but can also
    output negative values.

    Parameters
    ----------
    y_true : List
        TODO
    y_pred : List
        TODO
    """
    check_equal_shape(y_true, y_pred)

    # Convert both arrays to floating type to prevent overflows, rounding errors, etc.
    y_true = list(map(float, y_true))
    y_pred = list(map(float, y_pred))

    sse = sum([(y1 - y2) ** 2 for y1, y2 in zip(y_true, y_pred)])
    tse = (len(y_true) - 1) * variance(y_true)

    # Numerical stability for very small denominator
    if almost_zero(tse):
        return 0.0 if almost_zero(sse) else 1.0

    return 1 - (sse / tse)


def score_accuracy(y_true: List[int], y_prob: List[List[float]]) -> float:
    """
    Computes accuracy of labels in predicted set compared to the ground truth.

    Parameters
    ----------
    y_true : List
        TODO
    y_prob : List
        TODO
    """
    y_pred = argmax(y_prob)
    check_equal_shape(y_true, y_pred)
    return sum(equal_nested(y_true, y_pred)) / len(y_true)


def _compute_tfpn_counts(y_true: List[float], y_prob: List[float], threshold: float = 0.5) -> TFPN:
    """
    Computes the counts of true and false positive and negative predictions.
    """

    # Initialize the counters for true/false positives/negatives
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Loop through the input and compute counts for the given threshold
    for y, p in zip(y_true, y_prob):

        # Is the sample positive or negative?
        # We allow for samples to be of float value, positive is label>=threshold
        negative_sample = y < threshold

        # Is the guess positive or negative?
        negative_guess = p < threshold

        # If threshold says it's negative
        if negative_guess:
            if negative_sample:
                # Guess may be correct, making it a true negative
                true_negatives += 1
            else:
                # Guess may be wrong, making it a false negative
                false_negatives += 1

        # If threshold says it's positive
        else:
            if negative_sample:
                # Guess may be wrong, making it a false positive
                false_positives += 1
            else:
                # Guess may be correct, making it a true positive
                true_positives += 1

    return TFPN(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def score_precision_binary(
    y_true: List[float], y_prob: List[float], threshold: float = 0.5, counts: TFPN = None
) -> float:
    """
    Computes the precision of a set of predictions compared to the ground truth.
    """
    counts = counts or _compute_tfpn_counts(y_true, y_prob, threshold=threshold)
    return counts.true_positives / max(1, counts.true_positives + counts.false_positives)


def score_recall_binary(
    y_true: List[int], y_prob: List[float], threshold: float = 0.5, counts: TFPN = None
) -> float:
    """
    Computes the recall of a set of predictions compared to the ground truth.
    """
    counts = counts or _compute_tfpn_counts(y_true, y_prob, threshold=threshold)
    return counts.true_positives / max(1, counts.true_positives + counts.false_negatives)


def score_f1_binary(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> float:
    """
    Computes the F1 score of a set of predictions compared to the ground truth.
    """
    counts = _compute_tfpn_counts(y_true, y_prob, threshold=threshold)
    precision = score_precision_binary(y_true, y_prob, threshold=threshold, counts=counts)
    recall = score_recall_binary(y_true, y_prob, threshold=threshold, counts=counts)
    return 2 * precision * recall / max(1, precision + recall)


def roc_curve_binary(
    y_true: List[float], y_prob: List[float], bin_count: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Computes the Receiver Operating Characteristic curve.

    Parameters
    ----------
    y_true : List
        TODO
    y_prob : List
        TODO
    """
    # Initialize the true and false positive rate curves
    tpr = [0.0] * bin_count
    fpr = [0.0] * bin_count

    # Iterate over all the bins
    for idx_bin in range(bin_count):

        # Iterate over thresholds starting at 1
        threshold = 1.0 - idx_bin / float(bin_count)

        # Compute true/false positive/negative counts
        counts = _compute_tfpn_counts(y_true, y_prob, threshold=threshold)

        # True Positive Rate = True Positives / (True Positives + False Negatives)
        tpr[idx_bin] = counts.true_positives / max(
            1, counts.true_positives + counts.false_negatives
        )

        # False Positive Rate = False Positives / (True Negatives + False Positives)
        fpr[idx_bin] = counts.false_positives / max(
            1, counts.true_negatives + counts.false_positives
        )

    # When threshold is zero, all guesses are true and false positives simultaneously
    tpr += [1.0]
    fpr += [1.0]

    # Return a tuple of <True Positive Rate, False Positive Rate>
    return tpr, fpr


def _one_vs_all(y_true: List[int], y_prob: List[List[float]]):

    # Get the number of classes
    num_samples = len(y_true)
    assert num_samples == len(y_prob)
    num_labels = len(y_prob[0])

    # Binarize the probabilities by applying one-vs-all with respect to each of the labels
    for idx_label in range(num_labels):
        y_prob_binary = [y_prob[idx_sample][idx_label] for idx_sample in range(num_samples)]
        y_true_binary = [int(y_true[idx_sample] == idx_label) for idx_sample in range(num_samples)]

        # Yield the binarized outputs
        yield y_true_binary, y_prob_binary


def score_precision(y_true: List[int], y_prob: List[List[float]], threshold: float = 0.5) -> float:
    """
    Computes the precision of a set of predictions compared to the ground truth.
    """
    return mean(
        [
            score_precision_binary(y_true_bin, y_prob_bin, threshold=threshold)
            for y_true_bin, y_prob_bin in _one_vs_all(y_true, y_prob)
        ]
    )


def score_recall(y_true: List[int], y_prob: List[List[float]], threshold: float = 0.5) -> float:
    """
    Computes the recall of a set of predictions compared to the ground truth.
    """
    return mean(
        [
            score_recall_binary(y_true_bin, y_prob_bin, threshold=threshold)
            for y_true_bin, y_prob_bin in _one_vs_all(y_true, y_prob)
        ]
    )


def score_f1(y_true: List[int], y_prob: List[List[float]], threshold: float = 0.5) -> float:
    """
    Computes the F1 score of a set of predictions compared to the ground truth.
    """
    return mean(
        [
            score_f1_binary(y_true_bin, y_prob_bin, threshold=threshold)
            for y_true_bin, y_prob_bin in _one_vs_all(y_true, y_prob)
        ]
    )


def roc_curve_multiclass(y_true: List[int], y_prob: List[List[float]], bin_count: int = 100):

    # Binarize the probabilities by applying one-vs-all with respect to each of the labels
    tprs, fprs = [], []
    for y_true_binary, y_prob_binary in _one_vs_all(y_true, y_prob):

        # Compute the binary ROC curve for this label
        tpr, fpr = roc_curve_binary(y_true_binary, y_prob_binary, bin_count=bin_count)
        tprs.append(tpr)
        fprs.append(fpr)

    # Output the mean ROC curve across all labels
    return [mean(samples) for samples in zip(*tprs)], [mean(samples) for samples in zip(*fprs)]


def score_auroc(y_true: List[int], y_prob: List[List[float]]) -> float:
    """
    Computes the Area Under ROC curve (AUROC).

    Parameters
    ----------
    y_true : List
        TODO
    y_prob : List
        TODO

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
    .. [2] `Analyzing a portion of the ROC curve. McClish, 1989
            <https://www.ncbi.nlm.nih.gov/pubmed/2668680>`_
    """
    assert len(y_true) == len(y_prob)
    tpr, fpr = roc_curve_multiclass(y_true, y_prob)
    fpr_diffs = [fpr[i] - fpr[i - 1] for i in range(1, len(fpr))]
    tpr_means = [(tpr[i] + tpr[i - 1]) / 2.0 for i in range(1, len(tpr))]
    return sum([tpr_i * fpr_i for tpr_i, fpr_i in zip(tpr_means, fpr_diffs)])
