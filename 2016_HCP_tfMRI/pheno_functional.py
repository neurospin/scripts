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


path = '/neurospin/tmp/yleguen/task_fMRI_HCP_analysis_data/'
tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
COPE_NUMS = [[1,3], [1,3], [1,4], [1,4], [1,3], [1, 22]]
s_ids = os.listdir('/neurospin/population/HCP/task_fMRI_HCP_analysis_data/')

sides = ['L', 'R']
reg = "_MSMAll"
files = ['pe1', 'cope1', 'zstat1'] # cope1 and pe1 are identical
files = ['pe1']
fil = 'pe1'
group_path = '/neurospin/brainomics/2016_HCP/functional_analysis/HCP_MMP1.0'

file_parcels = os.path.join(group_path, 'L'+'.fsaverage41k.label.gii')
array_parcels = gio.read(file_parcels).darrays[0].data
parcels_org =  np.unique(array_parcels)
parcels_org = parcels_org[1:]

def mean_value_phenotyping(pair):
    i, task = pair
    for side in sides:
        file_parcels = os.path.join(group_path, side+'.fsaverage41k.label.gii')
        array_parcels = gio.read(file_parcels).darrays[0].data
        for k, parcel in enumerate(parcels_org):
            parameters.append([k, parcel])
            working_dir = 'functional_analysis/solar_analysis'
            output_dir = os.path.join('/neurospin/brainomics/2016_HCP/functional_analysis/pheno_mean_value', task+'_'+str(i), side)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            index = np.where(array_parcels == parcel)[0]
            array_s_ids = []
            array_mu = []
            for s_id in s_ids:
                file_path = path+s_id+'/MNINonLinear/Results/tfMRI_'+task+'/tfMRI_'+task+'_hp200_s2_level2'+reg+'.feat/GrayordinatesStats/cope'+str(i)+'.feat/'
                metric_out = os.path.join(file_path, side+"_"+fil+".41k_fsavg_"+side+".func.gii")
                if os.path.isfile(metric_out):
                    array_s_ids.append(s_id)
                    array_fil = gio.read(metric_out).darrays[0].data
                    array_mu.append(np.mean(array_fil[index]))
            df = pd.DataFrame()
            df['IID'] = array_s_ids
            df[fil] = array_mu
            output = os.path.join(output_dir, parcels_name[k]+'_'+fil)
            print "saving "+output
            df.to_csv(output+'.csv',  header=True, index=False)
"""
def max_value_phenotyping(pair):
    k, parcel = pair
    index = np.where(array_parcels == parcel)[0]
    for fil in files:
        array_s_ids = []
        array_mu = []
        for s_id in s_ids:
            file_path = path+s_id+'/MNINonLinear/Results/tfMRI_'+task+'/tfMRI_'+task+'_hp200_s2_level2'+reg+'.feat/GrayordinatesStats/cope'+str(COPE_NUM)+'.feat/'
            metric_out = os.path.join(file_path, side+"_"+fil+".41k_fsavg_"+side+".func.gii")
            if os.path.isfile(metric_out):
                array_s_ids.append(s_id)
                array_fil = gio.read(metric_out).darrays[0].data
                array_mu.append(np.amax(array_fil[index]))
        df = pd.DataFrame()
        df['IID'] = array_s_ids
        df[fil] = array_mu
        output = os.path.join(output_dir, parcels_name[k]+'_max_'+fil)
        df.to_csv(output+'.csv',  header=True, index=False)
"""




file_parcels = os.path.join(group_path, 'L'+'.fsaverage41k.label.gii')
array_parcels = gio.read(file_parcels).darrays[0].data



t0 = time.time()
parameters = []
for j, task in enumerate(tasks):
    for i in range(COPE_NUMS[j][0],COPE_NUMS[j][1]+1):
        parameters.append([i, task])
number_CPU = cpu_count()
pool = Pool(processes = number_CPU-2)
pool.map(mean_value_phenotyping, parameters)
pool.close()
pool.join()
print "Elapsed time for all parcels phenotyping: "+str(time.time()-t0)



"""
# Test heritability for each parcel found  0.35, 0.46, 0.35 for mean(pe) in sup temporal d, c, b
print "\n"
print "Test heritability for each parcel mean value"

    # Compute heritability estimate with SOLAR
    t0 = time.time()
    os.system("solar makeped "+ working_dir)
    for k, parcel in enumerate(parcels_org):
        for fil in files:
            filename = os.path.join(output_dir, parcels_name[k]+'_'+fil+'.csv')
            output_solar = 'pheno_mean_value/'+task+'_'+str(COPE_NUM)+'/'+side+'/'+parcels_name[k]
            os.system("solar pheno_analysis "+working_dir+" "+ output_solar+" "+fil+" "+filename)
    print "Elapsed time for all parcels solar: "+str(time.time()-t0)
"""
