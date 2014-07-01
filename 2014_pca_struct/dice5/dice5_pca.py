# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np
from  parsimony import datasets
from parsimony.datasets.utils import Dot, ObjImage

import scipy.linalg

from sklearn.metrics import precision_recall_fscore_support

def adjusted_explained_variance(X, V_k):
    Y = np.dot(X, V_k)
    Q, R = scipy.linalg.qr(Y)
    ev = np.trace(R**2)
    v = np.sum(X**2)
    return ev/v

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
    precision, recall, fscore, _ = precision_recall_fscore_support(lin_mask, lin_result,
                                                                   pos_label=1,
                                                                   average='micro')

    return (precision, recall, fscore)


def abs_correlation(x, y):
    # Correlation between the absolute values of the 2 arrays
    corr = np.corrcoef(np.abs(x.ravel()), np.abs(y.ravel()))
    return corr[1, 0]


def dice_five_with_union_of_pairs(shape, std=[1., 1., .5]):
    """Three objects, union1 (=dots 1 + 2), dot 3 and union2 (= dots 4 + 5)
    std st-dev of [5 dots,  union1, union2]

    Examples
    --------
    from parsimony.datasets.utils import *
    shape = (100, 100, 1)
    std_lattent = np.zeros(shape)
    objects = dice_five_with_union_of_pairs(shape)
    for o in objects:
       std_lattent[o.get_mask()] += o.std
    import matplotlib.pyplot as plt
    cax = plt.matshow(std_lattent.squeeze())
    plt.colorbar(cax)
    plt.title("Std-Dev of latents")
    plt.show()
    """
    nx, ny, nz = shape
    if nx < 5 or ny < 5:
        raise ValueError("Shape too small minimun is (5, 5, 0)")
    s_obj = np.max([1, np.floor(np.max(shape) / 7)])
    k = 1
    c1 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d1 = Dot(center=c1, size=s_obj, shape=shape)
    c2 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d2 = Dot(center=c2, size=s_obj, shape=shape)
    union1 = ObjImage(mask=d1.get_mask() + d2.get_mask(), std=std[0])
    k = 3
    c4 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d4 = Dot(center=c4, size=s_obj, shape=shape)
    c5 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d5 = Dot(center=c5, size=s_obj, shape=shape)
    union2 = ObjImage(mask=d4.get_mask() + d5.get_mask(),std=std[2])
    ## dot in the middle
    c3 = np.floor((nx / 2., ny / 2., nz / 2.))
    d3 = Dot(center=c3, size=s_obj, shape=shape,  std=std[1])
    return [union1, d3, union2]

if __name__ == "__main__":
    shape = (100, 100, 1)
    objects = dice_five_with_union_of_pairs(shape)
    std_lattent = np.zeros(shape)
    for o in objects:
       std_lattent[o.get_mask()] += o.std
    import matplotlib.pyplot as plt
    cax = plt.matshow(std_lattent.squeeze())
    plt.colorbar(cax)
    plt.title("Std-Dev of latents")
    plt.show()
    X3d, _, _ = datasets.regression.dice5.load(n_samples=50, shape=shape, objects=objects, random_seed=1)
    # run /home/ed203246/git/scripts/2014_pca_struct/tv_dice5/dice5_pca.py
    import doctest
    doctest.testmod()
