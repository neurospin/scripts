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

Concat l & r mesh
=================

import brainomics.mesh_processing as mesh_utils
BASE_PATH = "/neurospin/brainomics/2013_adni/"
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")
cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "rh.pial.gii"))
cor = np.vstack([cor_l, cor_r])
tri_r += cor_l.shape[0]
tri = np.vstack([tri_l, tri_r])
mesh_utils.mesh_from_arrays(cor, tri, path=os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))

"""
import os
import numpy as np

BASE_PATH = "/neurospin/brainomics/2013_adni/"
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")
OUTPUT = os.path.join(BASE_PATH, "MCIc-CTL_fs")

import numpy as np

import brainomics.mesh_processing as mesh_utils
cor, tri = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))

nodes_with_edges = [[] for i in xrange(cor.shape[0])]
mask = np.load(os.path.join(OUTPUT, "mask.npy"))
assert mask.shape[0] ==  cor.shape[0]

def connect_edge_to_node(node_idx1, node_idx2, nodes_with_edges):
        if np.sum(cor[node_idx1] - cor[node_idx2]) >= 0: # attach edge to first node
            edge = [node_idx1, node_idx2]
            if not edge in nodes_with_edges[node_idx1]:
                nodes_with_edges[node_idx1].append(edge)
        else:  # attach edge to second node
            edge = [node_idx2, node_idx1]
            if not edge in nodes_with_edges[node_idx2]:
                nodes_with_edges[node_idx2].append(edge)

#tri = tris[0, :]
for i in xrange(tri.shape[0]):
    t = tri[i, :]
    connect_edge_to_node(t[0], t[1], nodes_with_edges)
    connect_edge_to_node(t[0], t[2], nodes_with_edges)
    connect_edge_to_node(t[1], t[2], nodes_with_edges)

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
