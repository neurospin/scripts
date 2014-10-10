#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:51:15 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import os, sys, argparse
import warnings
# Qt
try:
    from PyQt4 import QtGui, QtCore
except:
    warnings.warn('Qt not installed: the mdodule may not work properly, \
                   please investigate')

try:
    import anatomist.api as anatomist
    # needed here in oder to be compliant with AIMS
    app = QtGui.QApplication(sys.argv)
except:
    warnings.warn('Anatomist no installed: the mdodule may not work properly, \
                   please investigate')

import shutil

def do_mesh_cluster_rendering(mesh_file,
                              texture_file,
                              white_mesh_file,
                              anat_file,
                              palette_file="palette_signed_values_blackcenter",
                              a = None,
                              check=False,
                              verbose=True):
    """ Vizualization of signed stattistic map or weigths.

    Parameters
    ----------
    mesh_file : str (mandatory)
        a mesh file of interest.
    texture_file : str (mandatory)
        an image from which to extract texture (for the mesh file).
    white_mesh_file : str (mandatory)
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

    # load objects
    a_mesh = a.loadObject(mesh_file)
    a_wm_mesh = a.loadObject(white_mesh_file)
    a_texture = a.loadObject(texture_file)
    a_anat = a.loadObject(anat_file)

    # mesh option
    material = a.Material(diffuse=[0.8, 0.8, 0.8, 0.7])
    a_wm_mesh.setMaterial(material)

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
    a_texture.setPalette(palette, minVal=-10, maxVal=10,
                         absoluteMode=True)

    # view object
    block = a.createWindowsBlock(1)
    w1 = a.createWindow("3D", block=block)
    w2 = a.createWindow("3D", block=block)

    # fusion 3D
    fusion3d = a.fusionObjects([a_mesh, a_texture], "Fusion3DMethod")
    w1.addObjects([fusion3d, a_wm_mesh])
    a.execute("Fusion3DParams", object=fusion3d, step=0.1, depth=5.,
              sumbethod="max", method="line_internal")

    # fusion 2D
    fusion2d = a.fusionObjects([a_anat, a_texture], "Fusion2DMethod")
    a.execute("Fusion2DMethod", object=fusion2d)

    # change 2D fusion settings
    a.execute("TexturingParams", texture_index=1, objects=[fusion2d, ],
              mode="linear_A_if_B_black", rate=0.1)

    # fusion cut mesh
    fusionmesh = a.fusionObjects([a_wm_mesh, fusion2d], "FusionCutMeshMethod")
    fusionmesh.addInWindows(w2)
    a.execute("FusionCutMeshMethod", object=fusionmesh)

    # start loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
        help='Input directory: output of "image_cluster_analysis" command', type=str)
    options = parser.parse_args()
    #print __doc__
    if options.input is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)

    mesh_file = os.path.join(options.input, "clust.gii")
    texture_file = os.path.join(options.input, "clust_values.nii.gz")
    white_mesh_file = white_mesh_file = "/neurospin/brainomics/neuroimaging_ressources/mesh/MNI152_T1_1mm_Bothhemi.gii"
    anat_file = os.path.join(options.input, "MNI152_T1_1mm_brain.nii.gz")
    
    # Set default values to parameters
    print 'mesh_file = "%s"' % mesh_file
    print 'texture_file = "%s"' % texture_file
    print 'white_mesh_file = "%s"' %  white_mesh_file
    print 'anat_file = "%s"' % anat_file
    
    do_mesh_cluster_rendering(mesh_file=mesh_file,
                              texture_file=texture_file,
                              white_mesh_file=white_mesh_file,
                              anat_file=anat_file)
