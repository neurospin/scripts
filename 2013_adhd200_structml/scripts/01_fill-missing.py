# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:43:24 2013

@author: ed203246

Read adhd file, fill missing data and save results.

Infomations:
-----------

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
import os
import os.path
import numpy as np
import tables

# Scripts PATH
try:
    ADHD200_SCRIPTS_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
  ADHD200_SCRIPTS_BASE_PATH = os.path.join(os.environ["HOME"] , "git", "scripts", "2013_adhd200_structml")
ADHD200_DATA_BASE_PATH = "/neurospin/adhd200"

# DATA PATH
INPUT_PATH = os.path.join(ADHD200_DATA_BASE_PATH, "python_analysis", "data")
OUTPUT_PATH = os.path.join("/volatile/adhd200/data", "python_analysis", "data")


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "data_api.py"))
execfile(os.path.join(ADHD200_SCRIPTS_BASE_PATH, "lib", "missing_values.py"))


## mGM =======================================================================
#get_csv("mw")
feature = "mw"
X_train , y_train = get_data(INPUT_PATH, feature, test=False)
## Some QC
# Out[7]: (array([327]),)
print "Train samples with missing data:", len(np.where(np.isnan(X_train).sum(1) != 0)[0])
X_train_nonans, nans = missing_fill_with_mean(X_train)

diff = X_train != X_train_nonans
print "Check that differances are only where NaNs have been found (Ok=True):",
print np.all(np.where(diff.sum(0)) == np.array(zip(*nans)[0]))
print 'Check no more NaNs (Ok=0):',
print np.isnan(X_train_nonans).sum()
# Now X_train is X_train_nonans
X_train = X_train_nonans

X_test , y_test = get_data(INPUT_PATH, feature, test=True)
print "Test samples with missing (Ok=0):", len(np.where(np.isnan(X_test).sum(1) != 0)[0])


print "Write to:", OUTPUT_PATH
write_data(X_train , y_train, OUTPUT_PATH, feature, test=False)
write_data(X_test , y_test, OUTPUT_PATH, feature, test=True)

