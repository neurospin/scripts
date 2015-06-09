# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:24:37 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

http://brainvisa.info/doc/pyaims-4.4/sphinx/index.html

"""
import os
import numpy as np
import json
import glob
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH = "/neurospin/brainomics/2013_adni"
WD = os.path.join(BASE_PATH, "MCIc-CTL-FS")
OUTPUT = "weight_map"

os.chdir(WD)
if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)


TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT, "rh.pial.gii"))


config  = json.load(open("config_5cv.json"))

#from soma import aims
#os.path.join(OUTPUT, "lh.pial.gii")
#mesh = aims.read(os.path.join(OUTPUT, "lh.pial.gii"))
#mesh.header()

cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "rh.pial.gii"))
assert cor_l.shape[0] == cor_r.shape[0] == 163842

cor_both, tri_both = mesh_utils.mesh_arrays(config["structure"]["mesh"])
mask__mesh = np.load(config["structure"]["mask"])
assert mask__mesh.shape[0] == cor_both.shape[0] == cor_l.shape[0] * 2 ==  cor_l.shape[0] + cor_r.shape[0]
assert mask__mesh.shape[0], mask__mesh.sum() == (327684, 317089)

# Find the mapping from beta in masked mesh to left_mesh and right_mesh
# concat was initialy: cor = np.vstack([cor_l, cor_r])
mask_left__mesh = np.arange(mask__mesh.shape[0])  < mask__mesh.shape[0] / 2
mask_left__mesh[np.logical_not(mask__mesh)] = False
mask_right__mesh = np.arange(mask__mesh.shape[0]) >= mask__mesh.shape[0] / 2
mask_right__mesh[np.logical_not(mask__mesh)] = False
assert mask__mesh.sum() ==  (mask_left__mesh.sum() + mask_right__mesh.sum())

# the mask of the left/right emisphere within the left/right mesh
mask_left__left_mesh = mask_left__mesh[:cor_l.shape[0]]
mask_right__right_mesh = mask_right__mesh[cor_l.shape[0]:]

# compute mask from beta (in masked mesh) to left/right
a = np.zeros(mask__mesh.shape, int)
a[mask_left__mesh] = 1
a[mask_right__mesh] = 2
mask_left__beta = a[mask__mesh] == 1  # project mesh to mesh masked
mask_right__beta = a[mask__mesh] == 2
assert (mask_left__beta.sum() + mask_right__beta.sum()) == mask_left__beta.shape[0] == mask_right__beta.shape[0] == mask__mesh.sum() 
assert mask_left__mesh.sum() == mask_left__beta.sum()
assert mask_right__mesh.sum() == mask_right__beta.sum()

# Check mapping from beta left part to left_mesh
assert mask_left__beta.sum() == mask_left__left_mesh.sum()
assert mask_right__beta.sum() == mask_right__right_mesh.sum()

todo = dict(
l2	  = dict(param=(0.01, 0.0, 1.0, 0.0, -1.0)),
l2tv	  = dict(param=(0.01, 0.0, 0.5, 0.5, -1.0)),
l1	  = dict(param=(0.01, 1.0, 0.0, 0.0, -1.0)),
l1tv	  = dict(param=(0.01, 0.5, 0.0, 0.5, -1.0)),
tv	  = dict(param=(0.01, 0.0, 0.0, 1.0, -1.0)),
l1l2	  = dict(param=(0.01, 0.5, 0.5, 0.0, -1.0)),
l1l2tv  = dict(param=(0.01, 0.35, 0.35, 0.3, -1.0)))

for k in todo:
    todo[k]["path"] = os.path.join("5cv", "*", "_".join([str(p) for p in todo[k]["param"]]))

#############################################################################
## CV 0
for k in todo:
    #k = "l1l2"
    Betas = np.vstack([array_utils.arr_threshold_from_norm2_ratio(
    np.load(os.path.join(filename, "beta.npz"))['arr_0'][config["penalty_start"]:, :].ravel(), .99)[0]
        for filename in glob.glob(todo[k]["path"])])
    # left
    tex = np.zeros(mask_left__left_mesh.shape)
    tex[mask_left__left_mesh] = Betas[0, mask_left__beta]
    print k, "left", np.sum(tex != 0), tex.max(), tex.min()
    mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_%s_left.gii" % k), data=tex)#, intent='NIFTI_INTENT_TTEST')
    tex[mask_left__left_mesh] = np.sum(Betas[1:, mask_left__beta] != 0, axis=0)
    mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_%s_left_cvcountnonnull.gii" % k), data=tex)#, intent='NIFTI_INTENT_TTEST')
    # right
    tex = np.zeros(mask_right__right_mesh.shape)
    tex[mask_right__right_mesh] = Betas[0, mask_right__beta]
    print k, "right", np.sum(tex != 0), tex.max(), tex.min()
    mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_%s_right.gii" % k), data=tex)#, intent='NIFTI_INTENT_TTEST')
    tex[mask_right__right_mesh] = np.sum(Betas[1:, mask_right__beta] != 0, axis=0)
    mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_%s_right_cvcountnonnull.gii" % k), data=tex)#, intent='NIFTI_INTENT_TTEST')

"""
l2tv left 76582 0.00314405576456 -0.00234570367821
l2tv left 97835 0.0256844622603 -0.0057810300272
l1tv left 2514 0.00345520687922 -0.00792580028994
l1tv left 2420 0.0253011691567 -0.00683345243997
l2 left 106357 0.00155634266392 -0.00191985498207
l2 left 104866 0.00144490807638 -0.00166196124374
l1 left 40 0.0569327459761 -0.493701359497
l1 left 28 0.0875157293883 -0.363745375713
tv left 63218 0.0026027757624 -0.00204264114237
tv left 94118 0.0324677399297 -0.00666839699601
l1l2tv left 2584 0.00517515547967 -0.0091712622082
l1l2tv left 2276 0.027860926291 -0.00749398089718
l1l2 left 150 0.0884005502511 -0.165126127696
l1l2 left 106 0.0811859481669 -0.134776560751

cd /neurospin/brainomics/2013_adni/MCIc-CTL-FS/oldies/weight_map

l2       signed_value             -0.001  +0.001
l2tv     signed_value_whitecenter -0.002  +0.002
l1l2     signed_value_whitecenter -0.05  +0.05
l1l2tv   signed_value_whitecenter -0.005  +0.005

/neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/bv_env /neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/anatomist tex_

/neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/bv_env /neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/anatomist *.gii

Pour lh/lg.pial charger les référentiels, les afficher dans lh.pial/rh.pial
Color / Rendering / Polygines face is clockwize


/neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/bv_env /neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/anatomist *.gii
/neurospin/brainvisa/build/Ubuntu-14.04-x86_64/bug_fix/bin/bv_env /neurospin/brainvisa/build/Ubuntu-14.04-x86_64/bug_fix/bin/anatomist *.gii

ls *.png|while read input; do
convert  $input -trim /tmp/toto.png;
#convert  /tmp/toto.png -transparent black $input;
cp  /tmp/toto.png $input;
done

"""