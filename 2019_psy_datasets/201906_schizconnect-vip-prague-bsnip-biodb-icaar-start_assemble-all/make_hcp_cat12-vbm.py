import xml.etree.ElementTree as ET
import glob
import pandas as pd
import re

INPUT_CSV_hcp = "/neurospin/hcp/participants.csv"
OUTPUT_CSV_hcp = "/neurospin/psy_sbox/hc/hcp/participants.tsv"
# Retrieve the TIV for every subject in the study
filenames_hcp = glob.glob("/neurospin/psy/hcp/derivatives/cat12vbm/sub-*/report/cat_*_3T_T1w_MPR1.xml")

assert len(filenames_hcp) == 1113

phenotype = pd.read_csv(INPUT_CSV_hcp)
assert phenotype.shape == (1206,383)
assert len(set(phenotype.index)) == 1206

phenotype = phenotype.rename(columns={'Subject': 'participant_id', "Age": "age", "Gender": "sex"})
phenotype.sex = phenotype.sex.map({'M': 0.0, 'F': 1.0})

phenotype["study"] = "HCP"
phenotype["site"] = "WashU"
phenotype['diagnosis'] = 'control'


for hcp_file in filenames_hcp:
    tree = ET.parse(hcp_file)
    tiv = float(tree.find('subjectmeasures').find('vol_TIV').text)
    # Get the participant id from the file name
    participant_id = int(re.search('/neurospin/psy/hcp/derivatives/cat12vbm/sub-(\d*)[^$]', hcp_file).group(1))
    index = phenotype.loc[phenotype['participant_id'] == participant_id].index
    if len(index) > 1:
        raise ValueError("Multiple index for subject {}".format(participant_id))
    # Add the TIV
    phenotype.loc[index[0], 'tiv'] = tiv

phenotype = phenotype.drop(phenotype[phenotype['tiv'].isna()].index)
phenotype['participant_id'] = phenotype['participant_id'].astype(str)

assert phenotype.shape == (1113, 387)
assert len(set(phenotype.index)) == 1113

phenotype.to_csv(OUTPUT_CSV_hcp, sep="\t", index=False)

## Append the real age on the mwp1 phenotype
hcp_restricted = pd.read_csv('/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/HCP_restricted_data.csv')
df = pd.read_csv('/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/all_t1mri_mwp1_participants.tsv', sep='\t')

# Use the age in HCP restricted data
assert set(hcp_restricted.Subject.astype(str)) >= set(df[df.study.eq('HCP')].participant_id)
for id, age in hcp_restricted[['Subject', 'Age_in_Yrs']].values:
    df.loc[df.participant_id.eq(str(id)), 'age'] = float(age)

df.to_csv('/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/all_t1mri_mwp1_participants.tsv', sep='\t', index=False)
