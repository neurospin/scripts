#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:51:15 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import os, sys, argparse
import math

import warnings
# Qt
try:
    from PyQt4 import QtGui, QtCore
except:
    warnings.warn('Qt not installed: the mdodule may not work properly, \
                   please investigate')

from soma import aims

try:
    import anatomist.api as anatomist
    # needed here in oder to be compliant with AIMS
    app = QtGui.QApplication(sys.argv)
except:
    warnings.warn('Anatomist no installed: the mdodule may not work properly, \
                   please investigate')

import shutil

def do_mesh_cluster_rendering(title,
                              clust_mesh_file,
                              clust_texture_file,
                              brain_mesh_file,
                              anat_file,
                              palette_file="palette_signed_values_blackcenter",
                              a = None,
                              check=False,
                              verbose=True):
    """ Vizualization of signed stattistic map or weigths.

    Parameters
    ----------
    clust_mesh_file : str (mandatory)
        a mesh file of interest.
    clust_texture_file : str (mandatory)
        an image from which to extract texture (for the mesh file).
    brain_mesh_file : str (mandatory)
        a mesh file of the underlying neuroanatomy.
    anat_file : str (mandatory)
        an image of the underlying neuroanatomy.
    check : bool (default False)
        manual check of the input parameters.
    verbose : bool (default True)
        print to stdout the function prototype.

    Returns
    -------
    None
    """
    #FunctionSummary(check, verbose)

    # instance of anatomist
    if a is None:
        a = anatomist.Anatomist()
    # ------------
    # load objects
    # ------------
    clust_mesh = a.loadObject(clust_mesh_file)
    brain_mesh = a.loadObject(brain_mesh_file)
    clust_texture = a.loadObject(clust_texture_file)
    a_anat = a.loadObject(anat_file)

    # mesh option
    material = a.Material(diffuse=[0.8, 0.8, 0.8, 0.7])
    brain_mesh.setMaterial(material)

    # change palette
    #palette_file = get_sample_data("brainvisa_palette").edouard
#    bv_rgb_dir = os.path.join(os.environ["HOME"], ".anatomist", "rgb")
#    if not os.path.isdir(bv_rgb_dir):
#        os.makedirs(bv_rgb_dir)
#    bv_rgb_file = os.path.join(bv_rgb_dir,
#                               os.path.basename(palette_file))
#    if not os.path.isfile(bv_rgb_file):
#        shutil.copyfile(palette_file, bv_rgb_file)
    palette = a.getPalette(palette_file)
    clust_texture.setPalette(palette, minVal=-10, maxVal=10,
                         absoluteMode=True)

    # view object
    block = a.createWindowsBlock(2)
    windows = list()
    objects = list()
    # --------------------------------
    # 3D view = fusion 3D + brain_mesh
    # --------------------------------
    win = a.createWindow("3D", block=block)
    fusion3d = a.fusionObjects([clust_mesh, clust_texture], "Fusion3DMethod")
    win.addObjects([fusion3d, brain_mesh])
    a.execute("Fusion3DParams", object=fusion3d, step=0.1, depth=5.,
              sumbethod="max", method="line_internal")
    windows.append(win)
    objects.append(fusion3d)
    # -------------------------------------
    # Slices view = three views offusion 2D
    # -------------------------------------
    for i in xrange(3):
        win = a.createWindow("3D", block=block)
#    win_coronal = a.createWindow("3D", block=block)
#    win_sagital = a.createWindow("3D", block=block)
    # fusion 2D
        fusion2d = a.fusionObjects([a_anat, clust_texture], "Fusion2DMethod")
        a.execute("Fusion2DMethod", object=fusion2d)

        # change 2D fusion settings
        a.execute("TexturingParams", texture_index=1, objects=[fusion2d, ],
                  mode="linear_A_if_B_black", rate=0.1)

        # fusion cut mesh
        fusionmesh = a.fusionObjects([brain_mesh, fusion2d], "FusionCutMeshMethod")
        a.execute("FusionCutMeshMethod", object=fusionmesh)
        fusionmesh.addInWindows(win)
        windows.append(win)
        objects.append([fusion2d, fusionmesh])

    #fusionmesh.addInWindows(win_coronal)
    #fusionmesh.addInWindows(win_sagital)

    # Global windows info
    try:
        block.widgetProxy().widget.setWindowTitle(str(title))
    except:
        print "could not set name"
    # rotation
#    rot_quat_coronal = aims.Quaternion()
#    rot_quat_coronal.fromAxis([0, 0, 1], math.pi/2)  # rotate x 90°
#    print rot_quat_coronal, math.pi/2
#    win_axial.camera(slice_quaternion=rot_quat_coronal.vector())

    #app.exec_()
    #print 1
    #sys.exit(app.exec_())
    return a, [clust_mesh, brain_mesh, clust_texture, a_anat, material, palette, block,
               windows, objects]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs',  nargs='+',
        help='Input directory(ies): output(s) of "image_cluster_analysis" command', type=str)
    options = parser.parse_args()
    if options.inputs is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)
    a = None
    gui_objs = list()
    for input_dirname in options.inputs:
        print input_dirname
        title = os.path.basename(input_dirname)
        clust_mesh_file = os.path.join(input_dirname, "clust.gii")
        clust_texture_file = os.path.join(input_dirname, "clust_values.nii.gz")
        brain_mesh_file = "/neurospin/brainomics/neuroimaging_ressources/mesh/MNI152_T1_1mm_Bothhemi.gii"
        anat_file = os.path.join(input_dirname, "MNI152_T1_1mm_brain.nii.gz")
        print 'clust_mesh_file = "%s"' % clust_mesh_file
        print 'clust_texture_file = "%s"' % clust_texture_file
        print 'brain_mesh_file = "%s"' %  brain_mesh_file
        print 'anat_file = "%s"' % anat_file
        a, objs = do_mesh_cluster_rendering(title=title,
            clust_mesh_file=clust_mesh_file,
            clust_texture_file=clust_texture_file,
            brain_mesh_file=brain_mesh_file,
            anat_file=anat_file,
            a=a)
        gui_objs.append(objs)
    sys.exit(app.exec_())
