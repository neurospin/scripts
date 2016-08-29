"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re, json
import pandas as pd
import time

path = '/neurospin/brainomics/2016_HCP/sulci_analysis/'
sulci = os.listdir(path)
sulci = set(sulci)-set(['pedigree'])
features = ['GM_thickness', 'hull_junction_length_native', 'maxdepth_native', 'meandepth_native', 'opening', 'surface_native']
sulci_dict_h2 = {}
sulci_dict_pval = {}
nb = ['.', '-', 'e', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

for sulcus in sulci:
    sulci_dict_h2[sulcus] = {}
    sulci_dict_pval[sulcus] = {}
    for feature in features:
        file_path = os.path.join(path, sulcus, feature, 'polygenic.out')
        for line in open(file_path, 'r'):
            if 'H2r is' in line and '(Significant)' in line:
                print line[4:len(line)-15]
                h2 = line[11:len(line)-30] 
                p = line[26:len(line)-15]                
                for k,l in enumerate(h2):
                    if not (l  in nb):
                        break
                h2 = float(h2[:k])
                p = float(p)
                print sulcus+" "+feature
                print "We extracted h2: "+str(h2)+" pval: "+str(p)
                sulci_dict_h2[sulcus][feature] = h2
                sulci_dict_pval[sulcus][feature] = p
                
output = '/neurospin/brainomics/2016_HCP/sulci_dict_h2.json'
encoded = json.dumps(sulci_dict_h2)
with open(output, 'w') as f:
    json.dump(encoded, f)
output = '/neurospin/brainomics/2016_HCP/sulci_dict_pval.json'
encoded = json.dumps(sulci_dict_pval)
with open(output, 'w') as f:
    json.dump(encoded, f)
