# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:47:22 2016

@author: ad247405
"""

import re
import glob
import os
import nibabel as nibabel
import numpy as np
import os
import pandas as pd

BASE_PATH =  '/neurospin/abide/schizConnect'

in_file = '/neurospin/abide/schizConnect/completed_schizconnect_metaData_1829.csv'
out_file = '/neurospin/abide/schizConnect/selected_images_schizConnect.csv'
outf = open(out_file, "w")
outf.write("study"+","+"subjectid"+","+"age"+","+"sex"+","+"dx"+","+"field_strength"+","+"img_date"+","+"datauri"+","+"maker"+","+"model"+","+"szc_protocol_hier"+","+"notes"+","+"imaging_protocol_site\n")
outf.flush()

images = pd.read_csv(in_file,delimiter=',')
n=0
id = 0
