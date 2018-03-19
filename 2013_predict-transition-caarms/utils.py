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
    clinic = clinic[clinic.M0CAARMS == "AR"]
    #print X.shape
    # (27, 28)
    return Xd, yd, clinic

## MISSING DATA
#np.where(np.isnan(X))[0][0]
#predictors[np.where(np.isnan(X))[1][0]]
#'@7.1'

CV10=[
([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25],[0,18,26]),
([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,25,26],[8,19,24]),
([0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,25,26],[6,21,23]),
([0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26],[3,5,22]),
([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26],[1,4,20]),
([0,1,3,4,5,6,7,8,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26],[2,9,16]),
([0,1,2,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20,21,22,23,24,25,26],[12,13,14]),
([0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26],[7,25]),
([0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,18,19,20,21,22,23,24,25,26],[11,15]),
([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26],[10,17])]

#for i, (tr, te) in enumerate(CV10): print i, tr,te