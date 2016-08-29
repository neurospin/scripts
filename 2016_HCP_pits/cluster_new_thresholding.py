"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json, time, argparse
import nibabel.gifti.giftiio as gio


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    parser.add_argument('-f', '--feature', type=str,
                        help="Features to measure")
    parser.add_argument('-s', '--symmetric', type=int,
                        help="Boolean, need to give 0 for False and another int for True " 
                        "Specify if symmetric template is used or not")
    parser.add_argument('-d', '--database', type=str,
                        help='Data base from which we take the cluster')
    parser.add_argument('-t', '--feature_threshold', type=str,
                        help='Specify the feature used for thresholding either sulc or DPF')
    options = parser.parse_args()
    ## INPUTS ###
    feature = options.feature
    SYMMETRIC = bool(int(options.symmetric))
    database_parcel  = options.database

    feature_threshold = options.feature_threshold
    if SYMMETRIC:
        INPUT  = '/neurospin/brainomics/2016_HCP/new_distrib2/distribution_sym_'+feature_threshold+'_'+database_parcel+'_Freesurfer_new/'
    else:
        INPUT  = '/neurospin/brainomics/2016_HCP/new_distrib2/distribution_'+feature_threshold+'_'+database_parcel+'_Freesurfer_new/'

    path0 = '/media/yl247234/SAMSUNG/hcp/Freesurfer_mesh_database/'
    #path0 = '/media/yl247234/SAMSUNG/hcp/databaseBV/'
    path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/Freesurfer_mesh_database/'
    #path_parcels = '/media/yl247234/SAMSUNG/'+database_parcel+'/databaseBV/'
    path = path0+'hcp/'
    temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
    sides = ['R', 'L']
    with open(temp_file_s_ids, 'r') as f:
        data = json.load(f)
    s_ids  = list(json.loads(data))

    ### OUTPUT ###
    if SYMMETRIC:
        OUTPUT = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_sym_'+feature+'_'+database_parcel+'_Freesurfer_new/'
    else:
        OUTPUT = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_'+feature+'_'+database_parcel+'_Freesurfer_new/'

    for side in sides:
        if SYMMETRIC:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_total_average_pits_smoothed0.7_60_sym_dist15.0_area100.0_ridge2.0.gii'
        else:
            file_parcels_on_atlas = path_parcels +'pits_density_update/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'   
        array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
        parcels_org =  np.unique(array_parcels)
        parcels_org = parcels_org[1:]
        NB_PARCELS = len(parcels_org)
        DATA_DPF = np.zeros((len(s_ids), NB_PARCELS))*np.nan
        INPUT_side  = INPUT+side+'/'
        thresholds = np.loadtxt(INPUT_side+'thresholds'+side+'.txt')
        for j,s_id in enumerate(s_ids):
            if SYMMETRIC:
                file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                  "default_analysis", "segmentation", "mesh",
                                                  "surface_analysis_sym", s_id+"_"+side+"white_pits_on_atlas.gii")
                file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                 "default_analysis", "segmentation",
                                                 #s_id+"_"+side+"white_depth_on_atlas.gii")
                                                 "mesh", "surface_analysis_sym", ""+s_id+"_"+side+"white_"+feature_threshold+"_on_atlas.gii")
                file_feature_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                 "default_analysis", "segmentation",
                                                 #s_id+"_"+side+"white_depth_on_atlas.gii")
                                                 "mesh", "surface_analysis_sym", ""+s_id+"_"+side+"white_"+feature+"_on_atlas.gii")
            else:
                file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                  "default_analysis", "segmentation", "mesh",
                                                  "surface_analysis", s_id+"_"+side+"white_pits_on_atlas.gii")
                file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                 "default_analysis", "segmentation",
                                                 #s_id+"_"+side+"white_depth_on_atlas.gii")
                                                 "mesh", "surface_analysis", ""+s_id+"_"+side+"white_"+feature_threshold+"_on_atlas.gii")
                file_feature_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                                 "default_analysis", "segmentation",
                                                 #s_id+"_"+side+"white_depth_on_atlas.gii")
                                                 "mesh", "surface_analysis", ""+s_id+"_"+side+"white_"+feature+"_on_atlas.gii")
            if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
                array_pits = gio.read(file_pits_on_atlas).darrays[0].data
                array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data
                array_feature = gio.read(file_feature_on_atlas).darrays[0].data
                for k,parcel in enumerate(parcels_org):
                    ind = np.where(parcel == array_parcels)
                    array_pits[ind] =  array_pits[ind]*(array_DPF[ind]>thresholds[k])
                index_pits = np.nonzero(array_pits)[0]
                parcels = array_parcels[index_pits]
                for k,parcel in enumerate(parcels_org):
                    ind = np.where(parcel == parcels)
                    # If the subject has pit in this parcel we consider the deepest
                    if ind[0].size:
                        index_max_DPF = np.argmax(array_DPF[index_pits[ind[0]]])
                        # MULTIPLY BY 20 MAYBE ONLY NEEDED FOR DPF
                        # SOLAR REQUIRE THIS TO ALLOW LARGE ENOUGH STD FOR ITS MODEL
                        #print array_feature[index_pits[ind[0]]]
                        #print array_DPF[index_pits[ind[0]]]
                        DATA_DPF[j,k] = 20*array_feature[index_pits[ind[0]]][index_max_DPF]
        # We will not consider subject with exactly 0 for now
        # Else use find zeros of numpy and replace them with almost 0
        DATA_DPF = np.nan_to_num(DATA_DPF)
        index_columns_kept = []
        for j in range(DATA_DPF.shape[1]):
            print np.count_nonzero(DATA_DPF[:,j])
            if np.count_nonzero(DATA_DPF[:,j]) > DATA_DPF.shape[0]*0.5:
                index_columns_kept.append(j)

        print index_columns_kept
        if not os.path.exists(OUTPUT):
            os.makedirs(OUTPUT)
        for index in index_columns_kept:
            df3 = pd.DataFrame()
            df3['IID'] = np.asarray(s_ids)[np.nonzero(DATA_DPF[:,index])].tolist()
            df3['Parcel_'+str(int(parcels_org[index]))] = DATA_DPF[:,index][np.nonzero(DATA_DPF[:,index])]
            if not os.path.exists(OUTPUT):
                os.makedirs(OUTPUT)
            output = OUTPUT+feature+'_pit'+str(int(parcels_org[index]))+"side"+side
            df3.to_csv(output+'.csv',  header=True, index=False)

        df = pd.DataFrame()
        for j in range(DATA_DPF.shape[1]):
            df['Parcel_'+str(int(parcels_org[j]))] = DATA_DPF[:,j]
        df[df != 0] = 2
        df[df == 0] = 1
        df['IID'] = np.asarray(s_ids)
        OUTPUT2 = OUTPUT + 'case_control/'
        if not os.path.exists(OUTPUT2):
            os.makedirs(OUTPUT2)
        output = OUTPUT2+'all_pits_side'+side
        df.to_csv(output+'.csv', sep= ',',  header=True, index=False)
