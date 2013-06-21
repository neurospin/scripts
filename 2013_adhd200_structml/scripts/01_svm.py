# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:43:24 2013

@author: ed203246

get_csv("mw")
les chemins et diagnostics pour gris modulé, training set (attention, diagnostic 0, 1, 2, ou 3, control=0 et les autres sont les sous-types d'adhd)

get_csv("mw",test=True) 
pour la même chose pour le test set

get_mask_path()
path du masque ou carrément get_mask() pour les données du masque (binaire numpy)

get_data("mw",test=False ou True)
renvoie dans (X,Y) le tableau des données (nb sujets,nb voxels dans le masque) et des diagnostics (nb sujets) [ 0 ou 1 cette fois-ci]
[ pour info les sujets sont ordonnées dans le même ordre que le csv, si tu veux récupérer le NIP d'un sujet précis ]

tu peux remplacer "mw" par "w" pour le gris non modulé, "jd" pour le déterminant du Jacobien, cf le listing du 
répertoire data -- je crois que vous n'en êtes pas là mais pour les données multivariées ça ne rentre pas en 
mémoire, si tu en as besoin dis moi je te dirais comment j'avais fait

"""
import os.path


ADHD200_SCRIPTS_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADHD200_DATA_BASE_PATH = "/neurospin/adhd200"

execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "data_api.py"))


import numpy as np
get_csv("mw")

X_tr , y_tr = get_data("mw",test=False)
np.sum(np.isnan(X_tr))

X_test , y_test = get_data("mw",test=True)


import nibabel as nib
mask_im = nib.load(get_mask_path())
mask_arr = mask_im.get_data()
mask_bool = mask_arr == 1

#
tmp_im = np.zeros(mask_arr.shape, dtype=np.int16)
tmp_im2 = tmp_im[mask_bool]

tmp_im[mask_bool][np.isnan(X_tr).sum(0) == 1] = 1
np.sum(tmp_im2 != 0)
#tmp_im[mask_bool] = 1 # OK


img = nib.Nifti1Image(tmp_im, affine=mask_im.get_affine())
img.to_filename(os.path.join('/tmp','nan.nii.gz'))


#anatomist /neurospin/adhd200/python_analysis/data/mask_t0.1_sum0.8_closing.nii /tmp/nan.nii.gz

from sklearn.svm import SVC
svm = SVC(kernel="linear")

svm.fit(X_tr, y_tr)