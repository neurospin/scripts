#file_ll_path = "/media/mma/mescog/originals/Munich/CAD_bioclinica_nifti/1001/1001-M0-LL.nii.gz"

import os, os.path
#from soma import aims
#import tempfile
#import scipy, scipy.ndimage
import glob
import pandas as pd

INPUT_IMAGES = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_images"
OUPUT_CSV = "/home/ed203246/data/mescog/incident_lacunes_shape/incident_lacunes_moments.csv"

filenames = glob.glob(os.path.join(INPUT_IMAGES, "*", "*LacBL.nii*"))
#filename = filenames[0]

columns = None
tuples = list()
for filename in filenames:
    lacune_id = os.path.basename(os.path.dirname(filename))
    print lacune_id
    out = os.popen('AimsMoment -i %s' % filename)
    #lacune = None
    Moment_invariant = False
    #lacunes = list()
    for line in out:
        #print line
        item = [v.strip() for v in line.split(":")]
        if item[0] == 'Results for label':
            lacune_keys_values = list()
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
    if columns is None:
        columns = cols
    assert cols == columns
    tuples.append([lacune_id] + list(values))

d = pd.DataFrame(tuples, columns=["lacune_id"] + list(columns))
d.to_csv(OUPUT_CSV, index=False)

        #    if lacune is not None:
#        lacune.append(item)

#print lacune_key
#print lacunes



#l = l.strip()

"""
Results for label :  14
Image geometry: (1.01562x1.01562x0.800003)
Volume :  4.12595mm3
Number of points : 5
Volume (mm3) : 4.12595
Order 1 moments : 604.255 475.191 491.815
Order 2 moments : 2.38328 1.0214 -4.50304e-12 0.510702 -3.00202e-12 -4.50304e-12
Order 3 moments : -0.622415 0.207472 -2.30555e-09 -0.380365 1.53704e-09 0.103736 -2.68981e-09 2.30555e-09 3.07407e-09 1.53704e-09
Center of Mass :  (146.452,115.171,119.2)
Orientation :
    Inerty :  2.55351 ; 0.851171 ; -4.50304e-12
    V1 :  (0.948683,0.316228,-1.67297e-12)
    V2 :  (-0.316228,0.948683,-3.9036e-12)
    V3 :  (3.52694e-13,4.23232e-12,1)
Moment invariant:
0.615663 0.387248 -0.33664 -0.31061 -0.295948 -0.301664 0.298136 -0.276809 0.291022 0.323634 -0.309514 0.308811
"""