# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:58:35 2013

@author: edouard.duchesnay@cea.fr
"""

import pandas as pd
import numpy as np
import os.path

def read_Xy(WD):
    INPUT_clinical_filepath = os.path.join(WD, "data", "transitionPREP.csv")
    clinic = pd.read_table(INPUT_clinical_filepath, header=0, sep="\t")     
    # ne prendre que les a risque
    clinic.M0CAARMS == "AR"
    # Variables
    # ---------
    # Pre-Morbid Adjustment scale: PAS2gr
    # CAARMS
    CAARMS = ["@1.1", "@1.2", "@1.3", "@2.1", "@2.2", "@3.1", "@3.2", "@3.3", "@4.1", "@4.2", "@4.3", "@5.1", "@5.2", "@5.3", "@5.4", "@6.1", "@6.3", "@6.4", "@7.2", "@7.3", "@7.4", "@7.5", "@7.6", "@7.7", "@7.8"]
    # 7.1 has been removed : missign data for one subject
    # Canabis
    "CB_EXPO"    
    predictors = ["PAS2gr", "CB_EXPO"] + CAARMS
    Xd = clinic[clinic.M0CAARMS == "AR"][predictors]
    yd = clinic[clinic.M0CAARMS == "AR"]["TRANSITION"]
    #print X.shape
    # (27, 28)
    return Xd, yd

## MISSING DATA
#np.where(np.isnan(X))[0][0]
#predictors[np.where(np.isnan(X))[1][0]]
#'@7.1'
