
import os, glob, re, json, argparse
import pandas as pd
import time
import nibabel.gifti.giftiio as gio
from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np
filename  ='/neurospin/brainomics/2016_HCP/functional_analysis/HCP_MMP1.0/parcel_names.csv'
df_labels = pd.read_csv(filename)
df_labels.index = df_labels['Index']
parcels_name = [name.replace('\n', '').replace('/', ' ').replace(' ','_') for name in df_labels['Area Description']]
df_labels['Area Description'] = parcels_name

group_path = '/neurospin/brainomics/2016_HCP/functional_analysis/HCP_MMP1.0'
file_parcels = os.path.join(group_path, 'L'+'.fsaverage41k.label.gii')
array_parcels = gio.read(file_parcels).darrays[0].data
parcels_org =  np.unique(array_parcels)
parcels_org = parcels_org[1:]

working_dir = '/neurospin/brainomics/2016_HCP/functional_analysis/solar_analysis'
tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
COPE_NUMS = [[1,3], [1,3], [1,4], [1,4], [1,3], [1, 22]]

sides = ['L', 'R']

nb = ['.', '-', 'e', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
fil = 'pe1'


def parse_solar_output(pair):
    i, task = pair
    dict_h2 = {}
    dict_pval = {}
    output = '/neurospin/brainomics/2016_HCP/functional_analysis/herit_dict/pheno_mean_value/'+task+'_'+str(i)
    if not os.path.isdir(output):
        os.makedirs(output)
    for side in sides:
        dict_h2[side] = {}
        dict_pval[side] = {}
        for k, parcel in enumerate(parcels_org):
            output_solar = 'pheno_mean_value/'+task+'_'+str(i)+'/'+side+'/'+parcels_name[k]
            file_path = os.path.join(working_dir, output_solar, fil, 'polygenic.out')
            if os.path.isfile(file_path):
                for line in open(file_path, 'r'):
                    if 'H2r is' in line and '(Significant)' in line:
                        print line[4:len(line)-15]
                        h2 = line[11:len(line)-30] 
                        p = line[26:len(line)-15]                
                        for k,l in enumerate(h2):
                            if not (l  in nb):
                                break
                        h2 = float(h2[:k])
                        p = float(p)
                        print "We extracted h2: "+str(h2)+" pval: "+str(p)
                        dict_h2[side][str(parcel)] = h2
                        dict_pval[side][str(parcel)] = p
                        #dict_subj
            else:
                print "Check why "+file_path+" doesn't exist !"
         

    encoded = json.dumps(dict_h2)
    with open(output+'h2_dict.json', 'w') as f:
        json.dump(encoded, f)

    encoded = json.dumps(dict_pval)
    with open(output+'pval_dict.json', 'w') as f:
        json.dump(encoded, f)


t0 = time.time()
parameters = []
for j, task in enumerate(tasks):
    for i in range(COPE_NUMS[j][0],COPE_NUMS[j][1]+1):
        parameters.append([i, task])
number_CPU = cpu_count()
pool = Pool(processes = number_CPU-2)
pool.map(parse_solar_output, parameters)
pool.close()
pool.join()
print "Elapsed time for all parcels phenotyping: "+str(time.time()-t0)
