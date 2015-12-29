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

SCLICE_THIKNESS = 5
ANATOMIST = None
VIEWS = dict()
IMAGE = None
OUTPUT = None

class SnapshotAction(anatomist.cpp.Action):
    def name( self ):
        return 'Snapshot'

    def left_click(self, x, y, globx, globy):
        global ANATOMIST
        a = ANATOMIST
        print 'coucou', x,y

    def linked_cursor_disable(self):
        global ANATOMIST
        global VIEWS
        print "linked_cursor_disable"
        for name, view_type in VIEWS:
            #print name, view_type, VIEWS[(name, view_type)]
            ANATOMIST.execute('WindowConfig', windows=[VIEWS[(name, view_type)][0]], cursor_visibility=0)

    def linked_cursor_enable(self):
        global ANATOMIST
        global VIEWS
        print "linked_cursor_enable"
        for name, view_type in VIEWS:
            #print name, view_type, VIEWS[(name, view_type)]
            ANATOMIST.execute('WindowConfig', windows=[VIEWS[(name, view_type)][0]], cursor_visibility=1)

    def snapshot_single2(self):
        global ANATOMIST
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
        #fi = open("/tmp/snapshot/current")
        #prefix = fi.readline().strip()
        #fi.close()
        #filename = '%s.png' % prefix
        #print "save in to", filename
        #ANATOMIST.execute('WindowConfig', windows=[win], snapshot=filename)
    #print 'takePolygon', x, y

    def snapshot(self):
        global ANATOMIST
        global VIEWS
        global IMAGE
        image_dim = IMAGE.header()['volume_dimension'].arraydata()[:3]
        voxel_size = IMAGE.header()['voxel_size'].arraydata()[:3]
        print image_dim, voxel_size, image_dim * voxel_size
        image_dim *= voxel_size
        for name, view_type in VIEWS:
            #, VIEWS[(name, view_type)]
            win, mesh = VIEWS[(name, view_type)]
            print name, view_type, win, mesh, view_type == "axial"
            if view_type == "axial":
                print "snapshot axial"
                for slice_ in range(SCLICE_THIKNESS, image_dim[2], SCLICE_THIKNESS):
                    sclice_axial(mesh, -slice_)
                    filename = '%s/%s_%s_%03d.png' % (OUTPUT, name, view_type, slice_)
                    print filename
                    ANATOMIST.execute('WindowConfig', windows=[win],
                                       snapshot=filename)
            if view_type == "coronal":
                print "snapshot coronal"
                for slice_ in range(SCLICE_THIKNESS, image_dim[1], SCLICE_THIKNESS):
                    sclice_coronal(mesh, -slice_)
                    filename = '%s/%s_%s_%03d.png' % (OUTPUT, name, view_type, slice_)
                    print filename
                    ANATOMIST.execute('WindowConfig', windows=[win],
                                       snapshot=filename)
            if view_type == "sagital":
                print "snapshot sagital"
                for slice_ in range(SCLICE_THIKNESS, image_dim[0], SCLICE_THIKNESS):
                    sclice_sagital(mesh, -slice_)
                    filename = '%s/%s_%s_%03d.png' % (OUTPUT, name, view_type, slice_)
                    print filename
                    ANATOMIST.execute('WindowConfig', windows=[win],
                                       snapshot=filename)


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

def sclice_axial(fusionmesh, sclice_=-90):
    global ANATOMIST
    cut_plane_axial = aims.Point4df([0, 0, 1, sclice_])
    ANATOMIST.execute("SliceParams", objects=[fusionmesh],
          plane=cut_plane_axial)

def sclice_coronal(fusionmesh, sclice_=-90):
    global ANATOMIST
    cut_plane_coronal = aims.Point4df([0, 1, 0, sclice_])
    ANATOMIST.execute("SliceParams", objects=[fusionmesh],
          plane=cut_plane_coronal)

def sclice_sagital(fusionmesh, sclice_=-90):
    global ANATOMIST
    cut_plane_sagital = aims.Point4df([1, 0, 0, sclice_])
    ANATOMIST.execute("SliceParams", objects=[fusionmesh],
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
        an IMAGE from which to extract texture (for the mesh file).
    brain_mesh_file : str (mandatory)
        a mesh file of the underlying neuroanatomy.
    anat_file : str (mandatory)
        an IMAGE of the underlying neuroanatomy.
    check : bool (default False)
        manual check of the input parameters.
    verbose : bool (default True)
        print to stdout the function prototype.

    Returns
    -------
    None
    """
    #FunctionSummary(check, verbose)
    global ANATOMIST
    global VIEWS
    global IMAGE
    # instance of anatomist
    if ANATOMIST is None:
        ANATOMIST = anatomist.Anatomist()
        # Add new SnapshotControl button
        pix = QtGui.QPixmap( 'control.xpm' )
        anatomist.cpp.IconDictionary.instance().addIcon('Snap', pix)
        ad = anatomist.cpp.ActionDictionary.instance()
        ad.addAction( 'SnapshotAction', lambda: SnapshotAction() )
        cd = anatomist.cpp.ControlDictionary.instance()
        cd.addControl( 'Snap', lambda: SnapshotControl(), 25 )
        cm = anatomist.cpp.ControlManager.instance()
        cm.addControl( 'QAGLWidget3D', '', 'Snap' )
    a = ANATOMIST
    # ------------
    # load objects
    # ------------
    clust_mesh = a.loadObject(clust_mesh_file)
    brain_mesh = a.loadObject(brain_mesh_file)
    clust_texture = a.loadObject(clust_texture_file)
    IMAGE = aims.Reader().read(clust_texture_file)
    image_dim = IMAGE.header()['volume_dimension'].arraydata()[:3]
    """
    cd /home/ed203246/mega/data/mescog/wmh_patterns/summary/cluster_mesh/tvl1l20001
    from soma import aims
    clust_texture_file = "clust_values.nii.gz"
    IMAGE = aims.Reader().read(clust_texture_file)
    image_dim = IMAGE.header()['volume_dimension'].arraydata()[:3]


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
    VIEWS[(title, "3d")] = [win3d, fusion3d]
    # -------------------------------------
    # Slices view = three VIEWS offusion 2D
    # -------------------------------------
    # fusion 2D
    fusion2d = a.fusionObjects([a_anat, clust_texture], "Fusion2DMethod")
    a.execute("Fusion2DMethod", object=fusion2d)
    # change 2D fusion settings
    a.execute("TexturingParams", texture_index=1, objects=[fusion2d, ],
                  mode="Geometric")#, rate=0.1)

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
    sclice_coronal(fusionmesh_coronal, int(-(image_dim / 2)[1]))
    # Store
    windows.append(win_coronal)
    #a.execute
    objects.append([fusion2d, fusionmesh_coronal])
    VIEWS[(title, "coronal")] = [win_coronal, fusionmesh_coronal]

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
    sclice_axial(fusionmesh_axial, int(-(image_dim / 2)[2]))
    # Store
    windows.append(win_axial)
    #a.execute
    objects.append([fusion2d, fusionmesh_axial])
    VIEWS[(title, "axial")] = [win_axial, fusionmesh_axial]

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
    sclice_sagital(fusionmesh_sagital, int(-(image_dim / 2)[0]))
    # Store
    windows.append(win_sagital)
    #a.execute
    objects.append([fusion2d, fusionmesh_sagital])
    VIEWS[(title, "sagital")] = [win_sagital, fusionmesh_sagital]

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
    parser.add_argument('--output', '-o', help='Output directory', type=str, default="snapshosts")
    options = parser.parse_args()
    if options.inputs is None:
        print "Error: Input is missing."
        parser.print_help()
        exit(-1)
    #a = None
    gui_objs = list()
    print options
    if not os.path.exists(options.output):
        os.mkdir(options.output)
    OUTPUT = options.output
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
