import glob
import os
import shutil
import re
import numpy as np
import pandas as pd

path = '/neurospin/imagen/workspace/cati/BVdatabase/'
path_source = '/neurospin/imagen/FU2/processed/nifti/'
T1_sub_dir = 'SessionA/ADNI_MPRAGE/'
path_source_BL = '/neurospin/imagen/BL/processed/nifti/'


centres = ['Berlin','Dresden', 'Dublin', 'Hamburg', 'London', 'Mannheim', 'Nottingham', 'Paris']
centres_subjects_cati ={}
filenames_BL = []
subjects_BL = []
filenames_FU2 = []
subjects_FU2 = []
for j in range(len(centres)):
    path2 = os.path.join(path, centres[j])
    subjects = [os.path.join(path2,subject) for subject in os.listdir(path2) if os.path.isdir(os.path.join(path2,subject))]
    for i in range(len(subjects)):
        centres_subjects_cati[os.path.basename(subjects[i])]= centres[j]
        path3 = os.path.join(subjects[i],'t1mri/')
        path4 = os.path.join(path3,'BL/')
        for filename in glob.glob(os.path.join(path4,'*nii.gz')):
            filenames_BL.append(filename)
            subjects_BL.append(os.path.basename(filename)[:len(os.path.basename(filename))-len('.nii.gz')])
        path5 = os.path.join(path3, 'FU2/')
        for filename in glob.glob(os.path.join(path5,'*nii.gz')):
            filenames_FU2.append(filename)
            subjects_FU2.append(os.path.basename(filename)[:len(os.path.basename(filename))-len('.nii.gz')])


path_old_sulci_files = '/neurospin/cati/Imagen/results/sulcalMorphometry/'
sulcus_list = []
for filename in glob.glob(os.path.join(path_old_sulci_files,'*.csv')):
    m = re.search('morpho_(.+?).csv', filename)
    if m:
        sulcus_list.append(m.group(1))

acquisition = 'BL'
subjects_no_csv = []
for j in range(len(subjects_BL)):
    centre = centres_subjects_cati[subjects_BL[j]]
    subject = subjects_BL[j]
    path_csv = os.path.join('/neurospin/imagen/workspace/cati/BVdatabase',
                            centre,subject,'t1mri',acquisition,
                            'default_analysis/folds/3.1/default_session_auto/')

    filename = subject + '_default_session_auto_sulcal_morphometry.csv'
    if os.path.isfile(path_csv + filename): 
        df = pd.read_csv(path_csv+filename, delimiter=';')
        df.index = df['sulcus']
    else:
        subjects_no_csv.append(subject)
        continue
    ### BUILD THE SULCUS NAME LIST ###
    if j == 0:
        #sulcus_list = np.asarray(df.index)
        # Not all sulci are always present so actually build the sulci list from old results
        # Else I would have to look at all df.index and take what is in common (computationaly expensive)
        surface = np.zeros((len(sulcus_list),len(subjects_BL)))
        depthMax = np.zeros((len(sulcus_list),len(subjects_BL)))
        depthMean = np.zeros((len(sulcus_list),len(subjects_BL)))
        length = np.zeros((len(sulcus_list),len(subjects_BL)))
        GM_thickness = np.zeros((len(sulcus_list),len(subjects_BL)))
        opening = np.zeros((len(sulcus_list),len(subjects_BL)))

    for i in range(len(sulcus_list)):
        if sulcus_list[i] in df.index:
            surface[i,j] = df.loc[sulcus_list[i]]['surface_talairach']
            depthMax[i,j] = df.loc[sulcus_list[i]]['maxdepth_talairach']
            depthMean[i,j] = df.loc[sulcus_list[i]]['meandepth_talairach']
            length[i,j] = df.loc[sulcus_list[i]]['length_talairach']
            GM_thickness[i,j] = df.loc[sulcus_list[i]]['GM_thickness']
            opening[i,j] = df.loc[sulcus_list[i]]['opening']


## At some point there was a selection of the main sulci in the past but I don't know under which criteria (Yann's comment)
# So I consider all the sulci
"""path_old_sulci_files = '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/full_sulci/'
main_sulcus_list = []
for filename in glob.glob(os.path.join(path_old_sulci_files,'*.csv')):
    m = re.search('mainmorpho_(.+?).csv', filename)
    if m:
        main_sulcus_list.append(m.group(1))"""

for i in range(len(sulcus_list)):
    if sulcus_list[i] in df.index:
        df_sulci_BL = pd.DataFrame({'subject': np.asarray(subjects_BL),
                                    'label': np.repeat(df.loc[sulcus_list[i]]['label'], len(subjects_BL)),
                                    'side': np.repeat(df.loc[sulcus_list[i]]['side'], len(subjects_BL)),
                                    'surface': np.asarray(surface[i,:]),
                                    'depthMax': np.asarray(depthMax[i,:]),
                                    'depthMean': np.asarray(depthMean[i,:]),
                                    'length': np.asarray(length[i,:]),
                                    'GM_thickness': np.asarray(GM_thickness[i,:]),
                                    'opening': np.asarray(opening[i,:]),
                                })
        df_sulci_BL.index = df_sulci_BL['subject']
        df_sulci_BL = df_sulci_BL[['subject', 'label', 'side', 'surface', 'depthMax', 'depthMean', 'length', 'GM_thickness', 'opening']]
        df_sulci_BL.to_csv('/volatile/yann/sulci_data/all_sulci/BL/morpho_'+sulcus_list[i]+'.csv', sep= ';', header=True, index= False)

       

acquisition = 'FU2'
subjects_no_csv_FU2 = []
for j in range(len(subjects_FU2)):
    centre = centres_subjects_cati[subjects_FU2[j]]
    subject = subjects_FU2[j]
    path_csv = os.path.join('/neurospin/imagen/workspace/cati/BVdatabase',
                            centre,subject,'t1mri',acquisition,
                            'default_analysis/folds/3.1/default_session_auto/')

    filename = subject + '_default_session_auto_sulcal_morphometry.csv'
    if os.path.isfile(path_csv + filename): 
        df = pd.read_csv(path_csv+filename, delimiter=';')
        df.index = df['sulcus']
    else:
        subjects_no_csv_FU2.append(subject)
        continue
    ### BUILD THE SULCUS NAME LIST ###
    if j == 0:
        #sulcus_list = np.asarray(df.index)
        # Not all sulci are always present so actually build the sulci list from old results
        # Else I would have to look at all df.index and take what is in common (computationaly expensive)
        surface = np.zeros((len(sulcus_list),len(subjects_FU2)))
        depthMax = np.zeros((len(sulcus_list),len(subjects_FU2)))
        depthMean = np.zeros((len(sulcus_list),len(subjects_FU2)))
        length = np.zeros((len(sulcus_list),len(subjects_FU2)))
        GM_thickness = np.zeros((len(sulcus_list),len(subjects_FU2)))
        opening = np.zeros((len(sulcus_list),len(subjects_FU2)))

    for i in range(len(sulcus_list)):
        if sulcus_list[i] in df.index:
            surface[i,j] = df.loc[sulcus_list[i]]['surface_talairach']
            depthMax[i,j] = df.loc[sulcus_list[i]]['maxdepth_talairach']
            depthMean[i,j] = df.loc[sulcus_list[i]]['meandepth_talairach']
            length[i,j] = df.loc[sulcus_list[i]]['length_talairach']
            GM_thickness[i,j] = df.loc[sulcus_list[i]]['GM_thickness']
            opening[i,j] = df.loc[sulcus_list[i]]['opening']

for i in range(len(sulcus_list)):
    if sulcus_list[i] in df.index:
        df_sulci_FU2 = pd.DataFrame({'subject': np.asarray(subjects_FU2),
                                     'label': np.repeat(df.loc[sulcus_list[i]]['label'], len(subjects_FU2)),
                                     'side': np.repeat(df.loc[sulcus_list[i]]['side'], len(subjects_FU2)),
                                     'surface': np.asarray(surface[i,:]),
                                     'depthMax': np.asarray(depthMax[i,:]),
                                     'depthMean': np.asarray(depthMean[i,:]),
                                     'length': np.asarray(length[i,:]),
                                     'GM_thickness': np.asarray(GM_thickness[i,:]),
                                     'opening': np.asarray(opening[i,:]),
                                 })
        df_sulci_FU2.index = df_sulci_FU2['subject']
        df_sulci_FU2 = df_sulci_FU2[['subject', 'label', 'side', 'surface', 'depthMax', 'depthMean', 'length', 'GM_thickness', 'opening']]
        df_sulci_FU2.to_csv('/volatile/yann/sulci_data/all_sulci/FU2/morpho_'+sulcus_list[i]+'.csv', sep= ';', header=True, index= False)
