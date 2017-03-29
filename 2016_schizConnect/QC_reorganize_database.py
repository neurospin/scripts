import os
import numpy as np
import pandas as pd
import re
import glob


FS_BASE_PATH = "/neurospin/abide/schizConnect/processed/freesurfer"
QC_CSV = "/neurospin/abide/schizConnect/freesurfer_QC.xlsx"


qc_file = pd.read_excel(QC_CSV)

qc_file.DECISION

os.chdir(FS_BASE_PATH )
for i in range(qc_file.shape[0]):
    if qc_file.DECISION[i] == 0.0:
        cmd = "mv " + qc_file.IMAGES[i] + " passed_QC/"
        os.system(cmd)
        
    if qc_file.DECISION[i] == 1.0:
        cmd = "mv " + qc_file.IMAGES[i] + " did_not_passed_QC/"
        os.system(cmd)
        