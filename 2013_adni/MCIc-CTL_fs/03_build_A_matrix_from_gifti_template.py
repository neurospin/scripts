# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 19:09:38 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

cd /neurospin/brainomics/2013_adni/freesurfer_template
mris_convert /i2bm/local/freesurfer/subjects/fsaverage/surf/lh.pial ./lh.pial.gii
mris_convert /i2bm/local/freesurfer/subjects/fsaverage/surf/rh.pial ./rh.pial.gii

"""
import os
import numpy as np

BASE_PATH = "/neurospin/brainomics/2013_adni/"
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")
OUTPUT = os.path.join(BASE_PATH, "MCIc-CTL_fs")

import numpy as np
from nibabel.gifti import giftiio as gio

mask = np.load(os.path.join(OUTPUT, "mask.npy"))

# cd /home/ed203246/Dropbox/python/pylearn-parsimony/datasets/surf
g  = gio.read(os.path.join(TEMPLATE_PATH, "lh.pial.gii"))

# Build a nodes graph with edges attached to nodes
# points
pts = g.darrays[0].data
nodes_with_edges = [[] for i in xrange(pts.shape[0])]

# TRIANGLE
tris = g.darrays[1].data

assert mask.shape[0] ==  pts.shape[0] * 2

def connect_edge_to_node(node_idx1, node_idx2, nodes_with_edges):
        if np.sum(pts[node_idx1] - pts[node_idx2]) >= 0: # attach edge to first node
            edge = [node_idx1, node_idx2]
            if not edge in nodes_with_edges[node_idx1]:
                nodes_with_edges[node_idx1].append(edge)
        else:  # attach edge to second node
            edge = [node_idx2, node_idx1]
            if not edge in nodes_with_edges[node_idx2]:
                nodes_with_edges[node_idx2].append(edge)

#tri = tris[0, :]
for i in xrange(tris.shape[0]):
    tri = tris[i, :]
    connect_edge_to_node(tri[0], tri[1], nodes_with_edges)
    connect_edge_to_node(tri[0], tri[2], nodes_with_edges)
    connect_edge_to_node(tri[1], tri[2], nodes_with_edges)


n_neighbors = np.array([len(n) for n in nodes_with_edges])
print np.sum(n_neighbors)
print np.sum(n_neighbors) / float(len(nodes_with_edges))
print [[n, np.sum(n_neighbors == n)] for n in np.unique(n_neighbors)]
