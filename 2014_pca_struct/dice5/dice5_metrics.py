# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np

from sklearn.metrics import precision_recall_fscore_support


def geometric_metrics(mask, binarized_component):
    """
    Compute the recall, precision and f-score of the support recovery.

    Examples
    --------
    >>> masks, mask_props, empty_masks, full_masks = _generate_test_data()
    >>> # Test with the exact mask (should get 1.0 for all metrics)
    >>> for mask in masks: print geometric_metrics(mask, mask)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, 1.0)
    (1.0, 1.0, 1.0)
    >>> # Test with empty masks (should get 0.0 for all metrics)
    >>> for mask, empty in zip(masks, empty_masks): \
        print geometric_metrics(mask, empty)
    (0.0, 0.0, 0.0)
    (0.0, 0.0, 0.0)
    (0.0, 0.0, 0.0)
    >>> # Test with full masks (recall is the propostion of True, precision
    >>> # is 1.0, f-score is the average)
    >>> for i, (mask, full) in enumerate(zip(masks, full_masks)): \
        feat = geometric_metrics(mask, full); \
        prop = mask_props[i]; \
        assert(feat == (prop, 1.0, 2*prop / (1 + prop)))
    """
    assert(binarized_component.dtype == bool)
    assert(mask.dtype == bool)
    lin_mask = mask.ravel()
    lin_component = binarized_component.ravel()
    # Precision and recall rates
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(lin_mask, lin_component,
                                        pos_label=1,
                                        average='binary')

    return (precision, recall, fscore)


def dice(mask, binarized_component):
    """
    Compute the dice coefficient between a binary component and the mask.

    Examples
    --------
    >>> masks, mask_props, empty_masks, full_masks = _generate_test_data()
    >>> # Test with the exact mask (should get 1.0)
    >>> for mask in masks: print dice(mask, mask)
    1.0
    1.0
    1.0
    >>> # Test with empty masks (should get 0.0)
    >>> for mask, empty in zip(masks, empty_masks): print dice(mask, empty)
    0.0
    0.0
    0.0
    >>> # Test with full masks (should get the f-score but due to approx we
    >>> # don't get exact equality)
    >>> from numpy.testing import assert_allclose
    >>> for i, (mask, full) in enumerate(zip(masks, full_masks)): \
        feat = dice(mask, full); \
        prop = mask_props[i]; \
        assert_allclose(feat, 2*prop / (1 + prop))
    """
    assert(binarized_component.dtype == bool)
    assert(mask.dtype == bool)
    n_component = np.count_nonzero(binarized_component)
    n_mask = np.count_nonzero(mask)
    intersection = binarized_component & mask
    n_intersection = np.count_nonzero(intersection)
    denom = float(n_component + n_mask)
    num = float(2 * n_intersection)
    if denom == 0:
        return 0.0
    else:
        return num/denom


def _generate_test_data():
    from parsimony.datasets.regression import dice5
    shape = (100, 100, 1)
    all_objects = dice5.dice_five_with_union_of_pairs(shape)
    _, _, d3, _, _, union12, union45, _ = all_objects
    objects = [union12, union45, d3]
    masks = [o.get_mask() for o in objects]
    # Proportion of True in each mask
    mask_props = [float(mask.sum())/float(np.prod(mask.shape))
                  for mask in masks]
    empty_masks = [np.zeros(mask.shape, dtype=bool) for mask in masks]
    full_masks = [np.ones(mask.shape, dtype=bool) for mask in masks]
    return masks, mask_props, empty_masks, full_masks

if __name__ == '__main__':
    import doctest
    doctest.testmod()
