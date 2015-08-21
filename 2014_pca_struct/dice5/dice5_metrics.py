# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np

from sklearn.metrics import precision_recall_fscore_support


def geometric_metrics(mask, result):
    """
    Compute the recall, precision and f-score of the support recovery.

    Examples
    --------
    >>> from parsimony.datasets.regression import dice5
    >>> shape = (100, 100, 1)
    >>> all_objects = dice5.dice_five_with_union_of_pairs(shape)
    >>> _, _, d3, _, _, union12, union45, _ = all_objects
    >>> objects = [union12, union45, d3]
    >>> masks = [o.get_mask() for o in objects]
    >>> for mask in masks: \
        print geometric_metrics(mask, mask)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, 1.0)
    >>> empty_masks = [np.zeros(mask.shape, dtype=bool) for mask in masks]
    >>> for mask, empty in zip(masks, empty_masks): \
        print geometric_metrics(mask, empty)
    (0.0, 0.0, 0.0)
    (0.0, 0.0, 0.0)
    (0.0, 0.0, 0.0)
    >>> full_masks = [np.ones(mask.shape, dtype=bool) for mask in masks]
    >>> for mask, full in zip(masks, full_masks): \
        print geometric_metrics(mask, full)
    (0.0298, 1.0, 0.057875315595261212)
    (0.0298, 1.0, 0.057875315595261212)
    (0.0149, 1.0, 0.029362498768351564)
    """
    bin_result = result.ravel() != 0
    lin_mask = mask.ravel()
    lin_result = bin_result.ravel()
    # Precision and recall rates
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(lin_mask, lin_result,
                                        pos_label=1,
                                        average='binary')

    return (precision, recall, fscore)


def dice(binarized_component, mask):
    """
    Compute the dice coefficient between a binary component and the mask.
    """
    assert(binarized_component.dtype == bool)
    assert(mask.dtype == bool)
    n_component = np.count_nonzero(binarized_component)
    n_mask = np.count_nonzero(mask)
    intersection = binarized_component * mask
    n_intersection = np.count_nonzero(intersection)
    denom = float(n_component + n_mask)
    num = float(2 * n_intersection)
    if denom == 0:
        return 0.0
    else:
        return num/denom

if __name__ == '__main__':
    import doctest
    doctest.testmod()
