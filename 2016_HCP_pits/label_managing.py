import pandas as pd
import numpy as np

file_manual  = '/neurospin/brainomics/2016_HCP/LABELS/manual_labelling_sym_template.csv'
file_Im = '/neurospin/brainomics/2016_HCP/LABELS/Labels_Im.csv'
file_autocomplete = '/neurospin/brainomics/2016_HCP/LABELS/labelling_sym_template.csv'

df_manual = pd.read_csv(file_manual)
df_im = pd.read_csv(file_Im)
df_im.index = df_im['Numeros']

array_name = []
for k, num in enumerate(df_manual['Num_Im']):
    if ~pd.isnull(num):
        array_name.append(df_im.loc[num]['Labels'])
    else:
        array_name.append(df_manual.loc[k]['Name'])
df_manual['Name'] = array_name

df = pd.DataFrame()
df = df_manual[['Parcel', 'Name']]
df.to_csv(file_autocomplete, sep=',', header=True, index=False)
