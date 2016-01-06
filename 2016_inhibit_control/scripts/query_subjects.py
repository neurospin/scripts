# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:15:05 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas as pd
import getpass
import pickle
import os
import json
from genibabel import imagen_imputed_genotype_measure



import os
import getpass


# Cwbrowser import
from cwbrowser.cw_connection import CWInstanceConnection

def get_credentials():
    """ get_credentials.

    Function to get password and username.

    Parameters
    ----------


    Returns
    -------
    login : String

    password : String
    """
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
        
    return login, password


# Sandbox to adjust methods
if __name__ == "__main__":
    """ Main XXXX  file
    """
#    # read plos list of height SNPs
#    fname = ('/neurospin/brainomics/imagen_central/reproducing/height/'
#             'SNPheight.csv')
#    df = pd.read_csv(fname, sep=';')

    #set login, password, DB
    url = "https://imagen2.cea.fr/database/"
    login, password = get_credentials()
    print login, password
    connect = CWInstanceConnection(url, login, password, realm="Imagen2")
    
    rql = ("Any C, G Where X is Subject, X code_in_study C, "
                          "X handedness 'ambidextrous', X gender G")

    rset = connect.execute(rql, export_type="json")
    
    for item in rset:
        print item                         
