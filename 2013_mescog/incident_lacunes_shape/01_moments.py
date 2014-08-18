# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:33:56 2014

@author: edouard.duchesnay@cea.fr
"""

import os, os.path
import numpy as np
import scipy, scipy.ndimage
import glob
import pandas as pd
import nibabel

BASE_PATH = "/home/ed203246/data/mescog/incident_lacunes_shape"
INPUT_IMAGES = os.path.join(BASE_PATH, "incident_lacunes_images")
OUPUT_CSV = os.path.join(BASE_PATH, "incident_lacunes_moments.csv")
filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL.nii*"))
filename = filenames[0]
columns = None
columns_assert = None
data = list()

for filename in filenames:
    lacune_id = os.path.basename(os.path.dirname(filename))
    print lacune_id
    columns = ["lacune_id"]
    moments = [lacune_id]
    # 1) QC check only one CC is found
    #import scipy.ndimage as ndimage
    arr = nibabel.load(filename).get_data()
    if not np.all(np.unique(arr) == [0, 1]):
        print "More than one label", filename
        continue
    scipy.ndimage.find_objects(arr)
    label, num_features = scipy.ndimage.label(arr)
    if not num_features == 1:
        print "More than one CC", filename
        continue
    # 2) AimsMoment
    out = os.popen('AimsMoment -i %s' % filename)
    #lacune = None
    Moment_invariant = False
    #lacunes = list()
    found_cc = 0
    for line in out:
        #print line
        item = [v.strip() for v in line.split(":")]
        if item[0] == 'Results for label':
            #print "->",lacune_id
            lacune_keys_values = list()
            # Check only one object was found
            found_cc += 1
            if found_cc > 1:
                print "More than one object found by AimsMoment", filename
                continue
            #lacune_id = str(item[1])
            Moment_invariant = False
    #    if len(item) == 1 and item[0] == '':
    #        lacune_cols = ["lacune_id"] + lacune_cols
    #        lacune_values = [lacune_id] + lacune_values
    #        lacunes.append(lacune_values)
    #        lacune_keys_values = None
        if item[0] == 'Results for label' or \
           item[0] == 'Volume' or \
           item[0] == 'Image geometry' or\
           item[0] == 'Orientation':
               continue
        if item[0] == 'Number of points':
            lacune_keys_values.append(['Number_of_points', int(item[1])])
        elif item[0] == 'Volume (mm3)':
            lacune_keys_values.append(['Vol(mm3)', float(item[1])])
        elif item[0] == 'Order 1 moments':
            values = [float(v) for v in item[1].split(" ")]
            lacune_keys_values += \
                [['Order_1_Monent_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'Order 2 moments':
            values = [float(v) for v in item[1].split(" ")]
            lacune_keys_values += \
                [['Order_2_Monent_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'Order 3 moments':
            values = [float(v) for v in item[1].split(" ")]
            lacune_keys_values += \
                [['Order_3_Monent_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'Center of Mass':
            values = [float(v) for v in item[1].strip("()").split(",")]
            lacune_keys_values += \
                [['Center_of_Mass_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'Inerty':
            values = [float(v) for v in item[1].split(";")]
            lacune_keys_values += \
                [['Orientation_Inerty_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'V1':
            values = [float(v) for v in item[1].strip("()").split(",")]
            lacune_keys_values += \
                [['Orientation_V1_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'V2':
            values = [float(v) for v in item[1].strip("()").split(",")]
            lacune_keys_values += \
                [['Orientation_V2_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == 'V3':
            values = [float(v) for v in item[1].strip("()").split(",")]
            lacune_keys_values += \
                [['Orientation_V3_%i' % i, values[i]] for i in xrange(len(values))]
        elif item[0] == "Moment invariant":
            Moment_invariant = True
        elif Moment_invariant:
            values = [float(v) for v in item[0].split(" ")]
            lacune_keys_values += \
                [['Moment_Invariant_%i' % i, values[i]] for i in xrange(len(values))]
            Moment_invariant = False
    cols, values = zip(*lacune_keys_values)
    #if columns is None:
    #    columns = cols
    #assert cols == columns
    columns += list(cols)
    moments += list(values)
    # 3) Mesh file
    out = os.popen('AimsMesh -i %s' % filename)
    filename_mesh = filename.replace(".nii.gz", "_1_0.nii.gii")
    # 4) Compute surface & volume from mesh
    out = os.popen('AimsMeshArea -i %s' % filename_mesh)
    columns += ["area_mesh", "vol_mesh"]
    moments += [float(l.strip().split(" ")[1]) for l in out.readlines()]
   # 5) Get perforator orientation
    from soma import aims
    filename_perforator = os.path.join(os.path.dirname(filename), "%s-Perf.nii.gz" % lacune_id)
    vol_aims = aims.Reader().read(filename_perforator)
    # t,z,y,x
    arr = vol_aims.arraydata().squeeze()
    perfo_zyx = np.asarray(np.where(arr == 2)).squeeze() - np.asarray(np.where(arr == 1)).squeeze()
    voxel_size = vol_aims.header()['voxel_size'].arraydata()
    perfo_xyz = (perfo_zyx * voxel_size[::-1])[::-1]
    columns += ['Orientation_Perfo_%i' % i for i in xrange(len(perfo_xyz))]
    moments += perfo_xyz.tolist()
    if columns_assert is None:
        columns_assert = columns
    assert columns_assert == columns
    #
    data.append(moments)

moments = pd.DataFrame(data, columns=columns)

## Compute some additional moments
# compactness
# moments = pd.read_csv(OUPUT_CSV)
moments["compactness"] = moments["vol_mesh"] ** (2. / 3) / moments["area_mesh"]
# Inerties
I = np.array(moments[["Orientation_Inerty_0", "Orientation_Inerty_1", "Orientation_Inerty_2"]])
Im = I.mean(axis=1)
moments["fa"] = np.sqrt(3. / 2.) * \
np.sqrt(((I[:, 0] - Im) ** 2 + (I[:, 1] - Im) ** 2 + (I[:, 2] - Im) ** 2) /
np.sum(I ** 2, axis=1))
moments["inertie_max_norm"] = I[:, 0] ** 2 / np.sum(I ** 2, axis=1)
moments["inertie_min_norm"] = I[:, 2] ** 2 / np.sum(I ** 2, axis=1)
#
I = np.array(moments[["Orientation_V1_0", "Orientation_V1_1", "Orientation_V1_2"]])
I /= np.sqrt(np.sum(I ** 2, axis=1)[:, np.newaxis])
#np.sum(I ** 2, axis=1)

P = np.array(moments[["Orientation_Perfo_0", "Orientation_Perfo_1", "Orientation_Perfo_2"]])
P /= np.sqrt(np.sum(P ** 2, axis=1)[:, np.newaxis])
IP = np.sum(I * P, axis=1)
IP[IP < 0] = 1 + IP[IP < 0]
moments["angle_inertie_max_perf"] = np.arccos(IP)

moments.to_csv(OUPUT_CSV, index=False)

# descriptive statistics
out = os.popen('python ~/git/datamind/descriptive/descriptive_statistics.py -i %s -o %s' %
(OUPUT_CSV, OUPUT_CSV.replace(".csv", "_descriptive.xls")))
# run -i /home/ed203246/git/scripts/2013_mescog/incident_lacunes_shape/01_calc_lacunes_moments.py
