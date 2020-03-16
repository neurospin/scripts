import pandas as pd
import numpy as np
import os

BASE_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data'


## Extracts subcortical volumes from 46 regions + cortical thickness from 68 regions (34 per hemisphere)

subcortical_vol_features = ['Right-Thalamus-Proper', 'CC_Anterior', 'CortexVol', 'Left-Pallidum', 'Left-Caudate',
                            'Left-VentralDC', 'CC_Posterior', 'rhCortexVol', 'SupraTentorialVol', 'Left-Hippocampus',
                            'Right-Cerebellum-White-Matter', 'CC_Central', 'Right-vessel', '3rd-Ventricle',
                            '5th-Ventricle', 'Left-choroid-plexus', 'Optic-Chiasm', 'CC_Mid_Posterior',
                            'SupraTentorialVolNotVent', 'Left-vessel', 'Brain-Stem', 'Right-Lateral-Ventricle', 'SupraTentorialVolNotVentVox',
                            'rhSurfaceHoles', 'Right-Accumbens-area', 'lhCortexVol', 'Right-VentralDC',
                            'Left-Putamen', 'Right-Cerebellum-Cortex', 'Right-Caudate', 'Right-Pallidum', 'CC_Mid_Anterior',
                            'Left-Lateral-Ventricle', 'Right-Amygdala', 'Left-Thalamus-Proper', 'Left-Cerebellum-White-Matter',
                            'CSF', 'Right-Hippocampus', 'Right-choroid-plexus', 'Left-Accumbens-area', '4th-Ventricle',
                            'Left-Amygdala', 'Right-Putamen', 'Right-Inf-Lat-Vent', 'Left-Inf-Lat-Vent',
                            'Left-Cerebellum-Cortex']

cortical_thickness_features = ['lh_bankssts', 'lh_caudalanteriorcingulate', 'lh_caudalmiddlefrontal',
                               'lh_cuneus', 'lh_entorhinal', 'lh_frontalpole', 'lh_fusiform', 'lh_inferiorparietal',
                               'lh_inferiortemporal', 'lh_insula', 'lh_isthmuscingulate', 'lh_lateraloccipital',
                               'lh_lateralorbitofrontal', 'lh_lingual', 'lh_medialorbitofrontal', 'lh_middletemporal',
                               'lh_paracentral', 'lh_parahippocampal', 'lh_parsopercularis', 'lh_parsorbitalis',
                               'lh_parstriangularis', 'lh_pericalcarine', 'lh_postcentral', 'lh_posteriorcingulate',
                               'lh_precentral', 'lh_precuneus', 'lh_rostralanteriorcingulate', 'lh_rostralmiddlefrontal',
                               'lh_superiorfrontal', 'lh_superiorparietal', 'lh_superiortemporal', 'lh_supramarginal',
                               'lh_temporalpole', 'lh_transversetemporal', 'rh_bankssts', 'rh_caudalanteriorcingulate',
                               'rh_caudalmiddlefrontal', 'rh_cuneus', 'rh_entorhinal', 'rh_frontalpole', 'rh_fusiform',
                               'rh_inferiorparietal', 'rh_inferiortemporal', 'rh_insula', 'rh_isthmuscingulate',
                               'rh_lingual', 'rh_medialorbitofrontal', 'rh_middletemporal', 'rh_paracentral',
                               'rh_parahippocampal', 'rh_parsopercularis', 'rh_parsorbitalis', 'rh_parstriangularis',
                               'rh_pericalcarine', 'rh_postcentral', 'rh_posteriorcingulate', 'rh_precentral',
                               'rh_precuneus', 'rh_rostralanteriorcingulate', 'rh_rostralmiddlefrontal', 'rh_superiorfrontal',
                               'rh_superiorparietal', 'rh_superiortemporal', 'rh_supramarginal', 'rh_temporalpole',
                               'rh_transversetemporal', 'rh_lateraloccipital', 'rh_lateralorbitofrontal']

# Concatenates the 3 .tsv from all the studies and generates the npy matrix
cortical_thickness = pd.read_csv(os.path.join(BASE_PATH,
                                 'FS_thickness_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t')
subcortical_vol = pd.read_csv(os.path.join(BASE_PATH,
                                 'FS_subcort_vol_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t')
phenotype = pd.read_csv(os.path.join(BASE_PATH, 'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'),
                        sep='\t')

assert cortical_thickness.shape == (2317, 71)
assert subcortical_vol.shape == (2317, 67)
assert phenotype.shape == (3871, 46)

concat = cortical_thickness.merge(subcortical_vol, how='inner', on=['participant_id'])
concat = concat.merge(phenotype, how='inner', on=['participant_id'])


assert concat.shape == (2300, 71+67+46-2) # merge on participant id, everything else is unique

data = np.array(concat[cortical_thickness_features + subcortical_vol_features].values, dtype=np.float32)

# Add the channel dimension
data = np.expand_dims(data, 1)

np.save(os.path.join(BASE_PATH, 'FS_thickness_subcort_vol_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.npy'), data)

concat.to_csv(os.path.join(BASE_PATH,'FS_thickness_subcort_vol_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'),
              sep='\t', index=False)










