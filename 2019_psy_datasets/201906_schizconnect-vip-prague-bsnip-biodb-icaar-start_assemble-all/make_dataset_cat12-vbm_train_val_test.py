#!/usr/bin/env python
INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/*.csv'
OUTPUT_SEXAGE = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_sex-age/data/cat12vbm/{dataset}_{modality}_{tags}_{type}.{ext}'
OUTPUT_DX = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_dx/data/cat12vbm/{dataset}_{modality}_{tags}_{type}.{ext}'



########################################################################################################################
# Age and sex

# Train = control of 'SCHIZCONNECT-VIP', 'PRAGUE', 'BIOBD'
# Val = control of BSNIP

# Intagrate IXI
# /neurospin/psy/hc/ixi/ (600 MR images)
# /neurospin/psy/hc/localizer/ (~100 )
# /neurospin/psy/abide2/ (593 controls)


########################################################################################################################
# predict DX (SCZ)

# Train + val= all of 'SCHIZCONNECT-VIP', schizophrenia vs control
# test = 'PRAGUE' FEP vs control

# Test 2 = 'icaar-start'

########################################################################################################################