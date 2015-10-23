# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:09:00 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas as pd
import getpass
import pickle
import os
import json
from genibabel import imagen_imputed_genotype_measure

# Sandbox to adjust methods
if __name__ == "__main__":
    """ Main to test the methods of imagen_get_plink.py file
    """
    # read plos list of height SNPs
    fname = ('/neurospin/brainomics/imagen_central/reproducing/height/'
             'SNPheight.csv')
    df = pd.read_csv(fname, sep=';')

    #set login, password, DB
    if 'KEYPASS' in os.environ:
        if os.path.isfile(os.environ['KEYPASS']):
            login = json.load(open(os.environ['KEYPASS']))['login']
    else:
        login = raw_input("\nImagen2 login: ")
    if 'KEYPASS' in os.environ:
        if os.path.isfile(os.environ['KEYPASS']):
            password = json.load(open(os.environ['KEYPASS']))['passwd']
    else:
        password = getpass.getpass("Imagen2 password: ")
    imagen2DB_url = "https://imagen2.cea.fr/database/"

    # get the reference imputed data
    imputed_height_plos = imagen_imputed_genotype_measure(
                                                    login=login,
                                                    password=password,
                                                    snp_ids=df['SNP'].tolist())
    #
    fout = fname.replace('.csv', '_imputed.pickle')
    fp = open(fout, 'wb')
    pickle.dump(imputed_height_plos, fp)
    fp.close()
