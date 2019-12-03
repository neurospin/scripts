#!/usr/bin/env python
import pandas as pd
import numpy as np
import os

OUTPUT = '/neurospin/psy_sbox/analyses/201906_psy-hc_sex-age'
SCRIPT = 'git/scripts/201906_psy-hc_sex-age/'
OUTPUT_DX = '/neurospin/psy_sbox/analyses/201906_psy-scz_dx'


INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/{dataset}_mwp1_gs_{type}.{ext}'

SCHIZCONNECT_VIP_PRAGUE = (INPUT_PATH.format(dataset="schizconnect-vip", type="participants", ext="csv"),
                           INPUT_PATH.format(dataset="schizconnect-vip", type="data-32", ext="npy"))
BIOBD = (INPUT_PATH.format(dataset="biobd", type="participants", ext="csv"),
         INPUT_PATH.format(dataset="biobd", type="data-32", ext="npy"))
BSNIP = (INPUT_PATH.format(dataset="bsnip", type="participants", ext="csv"),
         INPUT_PATH.format(dataset="bsnip", type="data-32", ext="npy"))
ICAAR = (INPUT_PATH.format(dataset="icaar-start", type="participants", ext="csv"),
         INPUT_PATH.format(dataset="icaar-start", type="data-32", ext="npy"))

def concatenate_datasets(input_paths, output_paths, cols_projections=None):
    assert len(output_paths) == 2, 'output_path should be formatted as (out_df_path, out_npy_path)'
    df_conc = []
    npy_conc = []
    # 1st: concatenates all the csv/npy arrays
    for (df, npy) in input_paths:
        npy = np.load(npy)
        df = pd.read_csv(df)
        df_conc.append(df)
        npy_conc.append(npy)
    df_conc = pd.concat(df_conc, axis=0, ignore_index=True)
    npy_conc = np.concatenate(npy_conc, axis=0)
    # 2nd: make a projection accross particular sites, diagnosis...
    p_index = get_projected_index(df_conc, cols_projections)
    df_conc = df_conc.iloc[p_index]
    npy_conc = npy_conc[p_index]
    # 3rd: saves the data
    df_conc.to_csv(output_paths[0])
    np.save(output_paths[1], npy_conc)

def get_projected_index(df, cols_vals):
    if cols_vals is None:
        return df.index
    projected_index = df.index
    for (col, val) in cols_vals.items():
        projected_index = projected_index.intersection(df[getattr(df, col).isin(val)].index)
    return projected_index

########################################################################################################################
# Age and sex

# Train = control of 'SCHIZCONNECT-VIP', 'PRAGUE', 'BIOBD'
# Val = control of BSNIP


concatenate_datasets([SCHIZCONNECT_VIP_PRAGUE, BIOBD], [os.path.join(OUTPUT, 'train.csv'), os.path.join(OUTPUT, 'train.npy')],
                     cols_projections={'diagnosis': ['control']})

concatenate_datasets([BSNIP], [os.path.join(OUTPUT, 'validation.csv'), os.path.join(OUTPUT, 'validation.npy')],
                     cols_projections={'diagnosis': ['control']})


# Intagrate IXI
# /neurospin/psy/hc/ixi/ (600 MR images)
# /neurospin/psy/hc/localizer/ (~100 )
# /neurospin/psy/abide2/ (593 controls)


########################################################################################################################
# predict DX (SCZ)

# Train + val= all of 'SCHIZCONNECT-VIP', schizophrenia vs control
concatenate_datasets([SCHIZCONNECT_VIP_PRAGUE], [os.path.join(OUTPUT_DX, 'train.csv'), os.path.join(OUTPUT_DX, 'train.npy')],
                     cols_projections={'diagnosis': ['control', 'schizophrenia'], 'study': ['SCHIZCONNECT-VIP']})

# test = 'PRAGUE' FEP vs control
concatenate_datasets([SCHIZCONNECT_VIP_PRAGUE], [os.path.join(OUTPUT_DX, 'test_prague.csv'), os.path.join(OUTPUT_DX, 'test_prague.npy')],
                     cols_projections={'diagnosis': ['FEP', 'control'], 'study': ['PRAGUE']})

# Test 2 = 'icaar-start'
concatenate_datasets([ICAAR], [os.path.join(OUTPUT_DX, 'test_icaar.csv'), os.path.join(OUTPUT_DX, 'test_icaar.npy')],
                     cols_projections={'diagnosis': ['schizophrenia', 'control']})

########################################################################################################################
