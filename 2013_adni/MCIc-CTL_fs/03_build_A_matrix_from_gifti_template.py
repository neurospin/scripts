# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 19:09:38 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

fs to gifti
===========

cd /neurospin/brainomics/2013_adni/freesurfer_template
mris_convert /i2bm/local/freesurfer/subjects/fsaverage/surf/lh.pial ./lh.pial.gii
mris_convert /i2bm/local/freesurfer/subjects/fsaverage/surf/rh.pial ./rh.pial.gii



"""
import os
import numpy as np
import scipy.sparse as sparse

BASE_PATH = "/neurospin/brainomics/2013_adni/"
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")
OUTPUT = os.path.join(BASE_PATH, "MCIc-CTL_fs")

import numpy as np
import brainomics.mesh_processing as mesh_utils
mesh_coord, mesh_triangles = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))

# params

mask = np.load(os.path.join(OUTPUT, "mask.npy"))

import parsimony.functions.nesterov.tv as tv_helper
A, _ = tv_helper.nesterov_linear_operator_from_mesh(mesh_coord, mesh_triangles, mask=mask)


"""
# count neighbors (arrity) for each node
n_neighbors = np.array([len(n) for n in nodes_with_edges])
print np.sum(n_neighbors)
print np.sum(n_neighbors) / float(len(nodes_with_edges))
print [[n, np.sum(n_neighbors == n)] for n in np.unique(n_neighbors)]
# 983040
#2.99996337935
#[[0, 264], [1, 992], [2, 22115], [3, 281155], [4, 21724], [5, 1147], [6, 287]]

# count nb time a node is in a vertex
count = np.zeros(len(nodes_with_edges))
for n in nodes_with_edges:
    for v in n:
        for e in v:
            count[e] += 1

print count.min(), count.max(), count.mean()
# (5.0, 6.0, 5.9999267587065583)
"""
