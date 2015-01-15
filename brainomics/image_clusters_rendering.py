#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:51:15 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

####################################################
## DO SNAPSHOTS

http://brainvisa.info/forum/viewtopic.php?f=6&t=1610
Explain how:
    1. Scene - Tools - uncheck 'Display Cursor' (I don't want any cursor visible)
    3. Window - Save, to export this standardized view as a jpg or png or whatever...

Still need how to add a menu item
"""

import os, sys, argparse
import math
import numpy as np

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

anatomist_instance = None
views = dict()
image = None

class SnapshotAction(anatomist.cpp.Action):
    def name( self ):
        return 'Snapshot'

    def left_click(self, x, y, globx, globy):
        global anatomist_instance
        a = anatomist_instance
        print 'coucou', x,y

    def linked_cursor_disable(self):
        global anatomist_instance
        global views
        print "linked_cursor_disable"
        for name, view_type in views:
            #print name, view_type, views[(name, view_type)]
            anatomist_instance.execute('WindowConfig', windows=[views[(name, view_type)][0]], cursor_visibility=0)

    def linked_cursor_enable(self):
        global anatomist_instance
        global views
        print "linked_cursor_enable"
        for name, view_type in views:
            #print name, view_type, views[(name, view_type)]
            anatomist_instance.execute('WindowConfig', windows=[views[(name, view_type)][0]], cursor_visibility=1)

    def snapshot_single2(self):
        global anatomist_instance
        print "snapshot_single"
        win = self.view().window()
        #print "==="
        #print win
        #print self.view().aWindow()
        #print "==="
        #print self.view().getInfos()
        #win_info = win.getInfos()
        #print win_info["view_name"]
        # FIXME !!!
        fi = open("/tmp/snapshot/current")
        prefix = fi.readline().strip()
        fi.close()
        filename = '%s.png' % prefix
        print "save in to", filename
        anatomist_instance.execute('WindowConfig', windows=[win], snapshot=filename)
    #print 'takePolygon', x, y

    def snapshot(self):
        global anatomist_instance
        global anatomist_instance
        global views
        global image
        image_dim = image.header()['volume_dimension'].arraydata()[:3]
        for name, view_type in views:
            #, views[(name, view_type)]
            win, mesh = views[(name, view_type)]
            print name, view_type, win, mesh
            if name == "axial":
                sclice_axial(fusionmesh_axial, -(image_dim / 2)[2])

class SnapshotControl(anatomist.cpp.Control):
    def __init__(self, prio = 25 ):
        anatomist.cpp.Control.__init__( self, prio, 'SnapshotControl')
    def eventAutoSubscription(self, pool):
        key = QtCore.Qt
        NoModifier = key.NoModifier
        self.mousePressButtonEventSubscribe(key.LeftButton, NoModifier,
                                            pool.action('SnapshotAction').left_click)
        self.keyPressEventSubscribe(key.Key_D, NoModifier,
                                 pool.action('SnapshotAction').linked_cursor_disable)
        self.keyPressEventSubscribe(key.Key_E, NoModifier,
                                 pool.action('SnapshotAction').linked_cursor_enable)
        self.keyPressEventSubscribe(key.Key_S, NoModifier,
                                 pool.action('SnapshotAction').snapshot)

# composition of rotation in quaternion
def product_quaternion(Q1, Q2):
    a1, a2 = Q1[0], Q2[0]
    v1, v2 = Q1[1:], Q2[1:]
    v1, v2 = np.asarray(v1), np.asarray(v2)
    a = a1 * a2 - np.inner(v1, v2)
    v = a1 * v2 + a2 * v1 + np.cross(v1, v2)
    Q = np.hstack([a, v])
    return Q.tolist()

def sclice_axial(fusionmesh, sclice=-90):
    global anatomist_instance
    cut_plane_axial = [0, 0, 1, sclice]
    anatomist_instance.execute("SliceParams", objects=[fusionmesh],
          plane=cut_plane_axial)

def sclice_coronal(fusionmesh, sclice=-90):
    global anatomist_instance
    cut_plane_coronal = [0, 1, 0, sclice]
    anatomist_instance.execute("SliceParams", objects=[fusionmesh],
          plane=cut_plane_coronal)

def sclice_sagital(fusionmesh, sclice=-90):
    global anatomist_instance
    cut_plane_sagital = [1, 0, 0, sclice]
    anatomist_instance.execute("SliceParams", objects=[fusionmesh],
          plane=cut_plane_sagital)
"""
def xyz_to_mm(trm, vox_size):
    trmcp = aims.Motion(trm)
    trmcp.scale(aims.Point3d(vox_size), aims.Point3d(1, 1, 1))
    return trmcp

def ima_get_trm_xyz_to_mm(ima, refNb=0):
    trm_ima_mm2ref_mm = aims.Motion(ima.header()['transformations'][refNb])
    ima_voxel_size = ima.header()['voxel_size'][:3]
    return(xyz_to_mm(trm_ima_mm2ref_mm, ima_voxel_size))
"""
def do_mesh_cluster_rendering(title,
                              clust_mesh_file,
                              clust_texture_file,
                              brain_mesh_file,
                              anat_file,
                              palette_file="palette_signed_values_blackcenter",
                              #a = None,
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
    global anatomist_instance
    global views
    global image
    # instance of anatomist
    if anatomist_instance is None:
        anatomist_instance = anatomist.Anatomist()
        # Add new SnapshotControl button
        pix = QtGui.QPixmap( 'control.xpm' )
        anatomist.cpp.IconDictionary.instance().addIcon('Snap', pix)
        ad = anatomist.cpp.ActionDictionary.instance()
        ad.addAction( 'SnapshotAction', lambda: SnapshotAction() )
        cd = anatomist.cpp.ControlDictionary.instance()
        cd.addControl( 'Snap', lambda: SnapshotControl(), 25 )
        cm = anatomist.cpp.ControlManager.instance()
        cm.addControl( 'QAGLWidget3D', '', 'Snap' )
    a = anatomist_instance
    # ------------
    # load objects
    # ------------
    clust_mesh = a.loadObject(clust_mesh_file)
    brain_mesh = a.loadObject(brain_mesh_file)
    clust_texture = a.loadObject(clust_texture_file)
    image = aims.Reader().read(clust_texture_file)
    image_dim = image.header()['volume_dimension'].arraydata()[:3]
    """
    cd /home/ed203246/mega/data/mescog/wmh_patterns/summary/cluster_mesh/tvl1l20001
    from soma import aims
    clust_texture_file = "clust_values.nii.gz"


    trm_ima_mm2ref_mm = aims.Motion(ima.header()['transformations'][0])

    trm_xyz_to_mm = ima_get_trm_xyz_to_mm(ima)
    min_mm = trm_ima_mm2ref_mm.transform([0, 0, 0])
    #a=ima.header()['volume_dimension']
    ima_dim_xyz = ima.header()['volume_dimension'].arraydata()[:3]
    max_mm = trm_ima_mm2ref_mm.transform(ima_dim_xyz)

    trm_ima_mm2ref_mm = aims.Motion(ima.header()['transformations'][0])
    ima_voxel_size = ima.header()['voxel_size'][:3]
    xyz_to_mm(trm_ima_mm2ref_mm, ima_voxel_size)
    """
    #print clust_texture.header()#['volume_dimension']
    a_anat = a.loadObject(anat_file)

    # mesh option
    material = a.Material(diffuse=[0.8, 0.8, 0.8, 0.6])
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
    win3d = a.createWindow("3D", block=block)
    # orientation front left
    Q_rot = [np.cos(np.pi / 8.), np.sin(np.pi / 8.), 0, 0]
    fusion3d = a.fusionObjects([clust_mesh, clust_texture], "Fusion3DMethod")
    win3d.addObjects([fusion3d, brain_mesh])
    a.execute("Fusion3DParams", object=fusion3d, step=0.1, depth=5.,
              sumbethod="max", method="line_internal")
    #define point of view
    Q1 = win3d.getInfos()["view_quaternion"]  # current point of view
    Q = product_quaternion(Q1, Q_rot)
    a.execute("Camera", windows=[win3d], view_quaternion=Q)
    windows.append(win3d)
    objects.append(fusion3d)
    views[(title, "3d")] = [win3d, fusion3d]
    # -------------------------------------
    # Slices view = three views offusion 2D
    # -------------------------------------
    # fusion 2D
    fusion2d = a.fusionObjects([a_anat, clust_texture], "Fusion2DMethod")
    a.execute("Fusion2DMethod", object=fusion2d)
    # change 2D fusion settings
    a.execute("TexturingParams", texture_index=1, objects=[fusion2d, ],
                  mode="linear_A_if_B_black", rate=0.1)

    ##############
    # Coronal view
    win_coronal = a.createWindow("3D", block=block)
    print "win_coronal", win_coronal
    # Fusion cut mesh
    fusionmesh_coronal = a.fusionObjects([brain_mesh, fusion2d],
                                         "FusionCutMeshMethod")
    a.execute("FusionCutMeshMethod", object=fusionmesh_coronal)
    fusionmesh_coronal.addInWindows(win_coronal)
    # Coronal view
    # Slice
    #sclice_coronal(fusionmesh_coronal, -90)
    sclice_coronal(fusionmesh_coronal, -(image_dim / 2)[1])
    # Store
    windows.append(win_coronal)
    #a.execute
    objects.append([fusion2d, fusionmesh_coronal])
    views[(title, "coronal")] = [win_coronal, fusionmesh_coronal]

    ##############
    # Axial view
    win_axial = a.createWindow("3D", block=block)
    print "win_axial", win_axial, win_axial.getInfos()
    # Fusion cut mesh
    fusionmesh_axial = a.fusionObjects([brain_mesh, fusion2d],
                                       "FusionCutMeshMethod")
    a.execute("FusionCutMeshMethod", object=fusionmesh_axial)
    fusionmesh_axial.addInWindows(win_axial)
    # Axial view, orientation front top left
    Q_rot = [np.cos(np.pi / 8.), np.sin(np.pi / 8.), 0, np.sin(np.pi / 8.)]
    Q1 = win_axial.getInfos()["view_quaternion"]  # current point of view
    Q = product_quaternion(Q1, Q_rot)
    a.execute("Camera", windows=[win_axial], view_quaternion=Q, zoom=0.8)
    # Slice
    sclice_axial(fusionmesh_axial, -(image_dim / 2)[2])
    # Store
    windows.append(win_axial)
    #a.execute
    objects.append([fusion2d, fusionmesh_axial])
    views[(title, "axial")] = [win_axial, fusionmesh_axial]

    ##############
    #Sagital view
    win_sagital = a.createWindow("3D", block=block)
    print "win_sagital", win_sagital
    # Fusion cut mesh
    fusionmesh_sagital = a.fusionObjects([brain_mesh, fusion2d],
                                         "FusionCutMeshMethod")
    a.execute("FusionCutMeshMethod", object=fusionmesh_sagital)
    fusionmesh_sagital.addInWindows(win_sagital)
    # Sagital view: rotation -Pi/4 --> orientation right
    Q_rot = [np.sqrt(2) / 2., -np.sqrt(2) / 2., 0, 0]
    Q1 = win_sagital.getInfos()["view_quaternion"]  # current point of view
    Q = product_quaternion(Q1, Q_rot)
    a.execute("Camera", windows=[win_sagital], view_quaternion=Q, zoom=0.9)
    # Slice
    sclice_sagital(fusionmesh_sagital, -(image_dim / 2)[0])
    # Store
    windows.append(win_sagital)
    #a.execute
    objects.append([fusion2d, fusionmesh_sagital])
    views[(title, "sagital")] = [win_sagital, fusionmesh_sagital]

    # Global windows info
    try:
        block.widgetProxy().widget.setWindowTitle(str(title))
    except:
        print "could not set name"

    return [clust_mesh, brain_mesh, clust_texture, a_anat, material, palette,
            block, windows, objects]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs',  nargs='+',
        help='Input directory(ies): output(s) of "image_cluster_analysis" command', type=str)
    options = parser.parse_args()
    if options.inputs is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)
    #a = None
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
        objs = do_mesh_cluster_rendering(title=title,
            clust_mesh_file=clust_mesh_file,
            clust_texture_file=clust_texture_file,
            brain_mesh_file=brain_mesh_file,
            anat_file=anat_file)#,
            #a=a)
        gui_objs.append(objs)
    sys.exit(app.exec_())
