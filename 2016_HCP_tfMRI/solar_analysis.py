import os
import numpy as np
import pandas as pd
import nibabel.gifti.giftiio as gio
from multiprocessing import cpu_count
from multiprocessing import Pool
import time

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

path = '/neurospin/tmp/yleguen/task_fMRI_HCP_analysis_data/'
tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
COPE_NUMS = [[1,3], [1,3], [1,4], [1,4], [1,3], [1, 22]]
s_ids = os.listdir('/neurospin/population/HCP/task_fMRI_HCP_analysis_data/')

sides = ['R', 'L']
reg = "_MSMAll"
files = ['pe1', 'cope1', 'zstat1'] # cope1 and pe1 are identical
files = ['pe1']
fil = 'pe1'

working_dir = 'functional_analysis/solar_analysis'
def solar_analysis(pair):
    i, task = pair
    for side in sides:
        for k, parcel in enumerate(parcels_org):
            output_dir = os.path.join('/neurospin/brainomics/2016_HCP/functional_analysis/pheno_mean_value', task+'_'+str(i), side)
            filename = os.path.join(output_dir, parcels_name[k]+'_'+fil+'.csv')
            output_solar = 'pheno_mean_value/'+task+'_'+str(i)+'/'+side+'/'+parcels_name[k]
            os.system("solar makeped "+ working_dir+" "+ output_solar+" "+fil)
            os.system("solar pheno_analysis "+working_dir+" "+ output_solar+" "+fil+" "+filename)


t0 = time.time()
parameters = []
for j, task in enumerate(tasks):
    for i in range(COPE_NUMS[j][0],COPE_NUMS[j][1]+1):
        parameters.append([i, task])
number_CPU = cpu_count()
pool = Pool(processes = number_CPU-2)
pool.map(solar_analysis, parameters)
pool.close()
pool.join()
print "Elapsed time for all parcels phenotyping: "+str(time.time()-t0)  
                
