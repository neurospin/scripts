# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np
from  parsimony import datasets
from parsimony.datasets.utils import Dot, ObjImage


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
