# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:03:05 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"

INPUT_IMAGES = os.path.join(BASE_PATH, "incident_lacunes_images")
#INPUT_MOMENTS_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments_area_from_mesh.csv"
INPUT_MOMENTS_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
INPUT_PC_MNTS_INV_CSV = os.path.join(BASE_PATH, "results_moments_invariant", "mnts-inv_pca.csv")
INPUT_PC_TSR_INV_CSV = os.path.join(BASE_PATH, "results_tensor_invariant", "tnsr-inv_pca.csv")
OUTPUT = os.path.join(BASE_PATH, "results_summary")

moments = pd.read_csv(INPUT_MOMENTS_CSV)#, index_col=0)
mnts_inv_pc = pd.read_csv(INPUT_PC_MNTS_INV_CSV)#, index_col=0)
tsr_inv_pc = pd.read_csv(INPUT_PC_TSR_INV_CSV)#, index_col=0)


moments["angle_lacune_plane__perfo"] = np.pi / 2 - moments.perfo_angle_inertia_min
moments["angle_lacune_plane__perfo_deg"] = 180 * moments["angle_lacune_plane__perfo"] / np.pi

#############################################################################
## Angle between perfo an lacune
s = moments["perfo_lacune"]
s.hist(normed=True, alpha=0.2)
s.plot(kind='kde')
plt.xlabel("Angle (deg) perforator and plan ortho. to orientation with min inertia")
plt.savefig(os.path.join(OUTPUT, "angle_deg_perforator_plan_orthogonal_to_orientation_with_min_inertia.png"))
plt.savefig(os.path.join(OUTPUT, "angle_deg_perforator_plan_orthogonal_to_orientation_with_min_inertia.svg"))
plt.show()

#############################################################################
## scatter plot
col_to_plot = [col for col in moments.columns if col.count('tensor_invariant')]
col_to_plot.remove('tensor_invariant_mode')
col_to_plot += ["angle_lacune_plane__perfo_deg"]
short = moments[col_to_plot]
short.columns = ["FA", "LA", "PA", "SA", "Angle"]

from pandas.tools.plotting import scatter_matrix
axs = scatter_matrix(short, alpha=1., diagonal='kde')
plt.savefig(os.path.join(OUTPUT, "scatter_matrix_tensors_inv_plus_angle.png"))
plt.savefig(os.path.join(OUTPUT, "scatter_matrix_tensors_inv_plus_angle.svg"))

plt.show()
