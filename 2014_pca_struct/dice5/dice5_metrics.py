# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from  parsimony import datasets
from parsimony.datasets.utils import Dot, ObjImage

def dice_five_geometric_metrics(mask, result):
    """
    Compute the recall, precision and f-score of the support recovery.

    Examples
    --------
    >>> shape = (100, 100, 1)
    >>> objects = dice_five_with_union_of_pairs(shape)
    >>> masks = [o.get_mask() for o in objects]
    >>> for mask in masks: \
        print dice_five_geometric_metrics(mask, mask)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, 1.0)
    >>> empty_masks = [np.zeros(mask.shape, dtype=bool) for mask in masks]
    >>> for mask, empty in zip(masks, empty_masks): \
        print dice_five_geometric_metrics(mask, empty)
    (0.0, 0.0, 0.0)
    (0.0, 0.0, 0.0)
    (0.0, 0.0, 0.0)
    >>> full_masks = [np.ones(mask.shape, dtype=bool) for mask in masks]
    >>> for mask, full in zip(masks, full_masks): \
        print dice_five_geometric_metrics(mask, full)
    (0.0298, 1.0, 0.057875315595261212)
    (0.0149, 1.0, 0.029362498768351564)
    (0.0298, 1.0, 0.057875315595261212)
    """
    bin_result = result.ravel() != 0
    lin_mask = mask.ravel()
    lin_result = bin_result.ravel()
    # Precision and recall rates
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(lin_mask, lin_result,
                                        pos_label=1,
                                        average='micro')

    return (precision, recall, fscore)
