
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import os, glob, re, json, shutil
from multiprocessing import cpu_count
from multiprocessing import Pool
import time


#s_ids 848 subjects in LANGUAGE and 869 subjects in MOTOR (apparently we have two missing MOTOR)
path_org = '/neurospin/population/HCP/task_fMRI_HCP_analysis_data/'
path_dest = '/neurospin/tmp/yleguen/task_fMRI_HCP_analysis_data/'
path_midthick = path_dest

path0 = '/neurospin/population/HCP/'
directories = ['S500-1',  'S500-2',  'S500-3',  'S500-4',  'S500-5', 'extradata_500_to_900/HCP_unzip_Structural_preproc', 'Structural_S500_plus_MEG2_release_new_subjects']
dir_resample_by_HCP_fsaverage = "/neurospin/tmp/yleguen/resample_fsaverage/"

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'RELATIONAL', 'SOCIAL', 'WM']
COPE_NUMS = [[1,3], [1,3], [1,4], [1,4], [1,3], [1, 22]]

s_ids = sorted(os.listdir('/neurospin/population/HCP/task_fMRI_HCP_analysis_data/'))

registrations = ['_MSMAll']#['', '_MSMAll']
files = ['pe1']#['pe1', 'cope1', 'zstat1']
# Select files to convert pe1 (beta), cope1 (contrast), zstat1
fil = 'pe1'
extension =  '.dtseries.nii'
scalar_extension = '.dscalar.nii'
func_extension = '.func.gii'

def convert_dtseries_to_gii(s_id):
    # Select the task
    for k, task in enumerate(tasks):
        # Select registration type MSMAll or MSMSulc = ''
        for reg in registrations:
             for i in range(COPE_NUMS[k][0],COPE_NUMS[k][1]+1):
                # Convert dtseries.nii to dscalar.nii both are ciifti formats (cf doc)
                file_path_org = path_org+s_id+'/MNINonLinear/Results/tfMRI_'+task+'/tfMRI_'+task+'_hp200_s2_level2'+reg+'.feat/GrayordinatesStats/cope'+str(i)+'.feat/'
                file_path_dest = path_dest+s_id+'/MNINonLinear/Results/tfMRI_'+task+'/tfMRI_'+task+'_hp200_s2_level2'+reg+'.feat/GrayordinatesStats/cope'+str(i)+'.feat/'
                if not os.path.isdir(file_path_dest):
                    os.makedirs(file_path_dest)
                if not os.path.isfile(file_path_dest+fil+scalar_extension):
                    cmd = 'wb_command -cifti-convert-to-scalar '+file_path_org+fil+extension+' ROW '+file_path_dest+fil+scalar_extension#+' -name-file '+file_path+'Contrasttemp.txt'
                    print cmd
                    os.system(cmd)
                
                # Convert dscalar.nii to giifti
                cifti_in = os.path.join(file_path_dest, fil+scalar_extension)
                metric_left = os.path.join(file_path_dest, "L_"+fil+func_extension)
                metric_right = os.path.join(file_path_dest, "R_"+fil+func_extension)
                cmd = "wb_command -cifti-separate "+cifti_in+" COLUMN -metric CORTEX_LEFT "+metric_left+" -metric CORTEX_RIGHT "+metric_right
                if not os.path.isfile(metric_left) or not os.path.isfile(metric_right):
                    print cmd
                    os.system(cmd)


def project_native_func_to_41k(s_id):
    # Select registration type MSMAll = "_MSMAll" or MSMSulc = ''
    reg0 = ['_MSMAll']#["", "_MSMAll"]
    for reg in reg0:
        if reg == "_MSMAll":
            directories = ['3T_Structural_preproc_S500_to_S900_extension', 'extradata_500_to_900/HCP_unzip_Structural_preproc', 'Structural_S500_plus_MEG2_release_new_subjects']
        else:
            directories = ['S500-1',  'S500-2',  'S500-3',  'S500-4',  'S500-5', 'extradata_500_to_900/HCP_unzip_Structural_preproc', 'Structural_S500_plus_MEG2_release_new_subjects']
        for directory in directories:
            if os.path.isdir(os.path.join(path0, directory,s_id)):
                for side in ['L', 'R']:
                    # Select the task
                    for k, task in enumerate(tasks):
                        for i in range(COPE_NUMS[k][0],COPE_NUMS[k][1]+1):
                            file_path_org = path_org+s_id+'/MNINonLinear/Results/tfMRI_'+task+'/tfMRI_'+task+'_hp200_s2_level2'+reg+'.feat/GrayordinatesStats/cope'+str(i)+'.feat/'
                            file_path_dest = path_dest+s_id+'/MNINonLinear/Results/tfMRI_'+task+'/tfMRI_'+task+'_hp200_s2_level2'+reg+'.feat/GrayordinatesStats/cope'+str(i)+'.feat/'
                            if not os.path.isdir(file_path_dest):
                                os.makedirs(file_path_dest)

                            metric_in = os.path.join(file_path_dest, side+"_"+fil+func_extension)
                            metric_out = os.path.join(file_path_dest, side+"_"+fil+".41k_fsavg_"+side+".func.gii")
                            midthick_reg0  = os.path.join(path0, directory,s_id, "T1w/fsaverage_LR32k", s_id+"."+side+".midthickness"+reg+".32k_fs_LR.surf.gii")
                            fs_LR_to_32k = os.path.join(dir_resample_by_HCP_fsaverage, "fs_LR-deformed_to-fsaverage."+side+".sphere.32k_fs_LR.surf.gii")
                            fsavg_std_41k = os.path.join(dir_resample_by_HCP_fsaverage,"fsaverage6_std_sphere."+side+".41k_fsavg_"+side+".surf.gii")
                            midthick_41k = os.path.join(path_midthick, s_id, s_id+'.'+side+'.midthickness.41k_fsavg_'+side+'.surf.gii')
                            cmd = "wb_command -metric-resample "+metric_in+" "+fs_LR_to_32k+" "+fsavg_std_41k+" ADAP_BARY_AREA "+metric_out+" -area-surfs "+midthick_reg0+" "+midthick_41k
                            if not os.path.isfile(metric_out):
                                print cmd
                                os.system(cmd)



def midthick_41k(s_id):
    for directory in directories:
        if os.path.isdir(os.path.join(path0, directory,s_id)):
            for side in ['L', 'R']:
                midthick_nativ = os.path.join(path0, directory, s_id,'T1w/Native', s_id+'.'+side+'.midthickness.native.surf.gii')
                sphere_nativ = os.path.join(path0, directory, s_id,'MNINonLinear/Native', s_id+'.'+side+'.sphere.native.surf.gii')
                dir_midthick_41k = os.path.join(path_midthick, s_id)
                if not os.path.isdir(dir_midthick_41k):
                    os.makedirs(dir_midthick_41k)
                midthick_41k = os.path.join(path_midthick, s_id, s_id+'.'+side+'.midthickness.41k_fsavg_'+side+'.surf.gii')
                fsavg_std_41k = os.path.join(dir_resample_by_HCP_fsaverage,"fsaverage6_std_sphere."+side+".41k_fsavg_"+side+".surf.gii")
                cmd  ='wb_command -surface-resample '+midthick_nativ+' '+sphere_nativ+' '+fsavg_std_41k+' BARYCENTRIC '+midthick_41k
                if not os.path.isfile(midthick_41k):
                    print cmd
                    os.system(cmd)



# Resample midthickness from native to fsaverage6
t0 = time.time()
number_CPU = cpu_count()
pool = Pool(processes = number_CPU-2)
pool.map(midthick_41k, s_ids)
pool.close()
pool.join()
t0 = time.time()-t0
# Convert dtseries format to giifti
t1 = time.time()
number_CPU = cpu_count()
pool = Pool(processes = number_CPU-2)
pool.map(convert_dtseries_to_gii, s_ids)
pool.close()
pool.join()
t1 = time.time()-t1
# Resample native gii to fsaverage6
t2 = time.time()
number_CPU = cpu_count()
pool = Pool(processes = number_CPU-2)
pool.map(project_native_func_to_41k, s_ids)
pool.close()
pool.join()
t2 = time.time()-t2

print "Elapsed time for midthick_41k: "+str(t0)
print "Elapsed time for dtseries_to_gii: "+str(t1)
print "Elapsed time for native func to 41k: "+str(t2)
