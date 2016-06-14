# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:15:05 2015

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import pandas as pd
import os
import sys
import json
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
    login, password = None, None
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


# Query demographics
if __name__ == "__main__":
    """ query_subject
    """
    #set login, password, DB
    url = "https://imagen2.cea.fr/database/"
    login, password = get_credentials()
    if (login is None) or (password is None):
        print 'login: ',
        login = sys.stdin.readline().rstrip()
        password = getpass.getpass('password: ')
    connect = CWInstanceConnection(url, login, password, realm="Imagen")

    # rql to get metadata and data extracted from a table (not indexed)
    # see D which is returned as a dict
    rql = ("Any ID, HAND, SEX, D WHERE QR is QuestionnaireRun, "
           "QR questionnaire Q, Q name 'DAWBA-dawba-youth', "
           "QR in_assessment A, A timepoint 'BL', QR file F, "
           "F data D, QR subject S,  S code_in_study ID, "
           "S handedness HAND, S gender SEX")

    # execute
    rset = connect.execute(rql, export_type="json")

    # parse the result with specifi attention for the last field
    loaded_rset = [(sid, handednesss, sex, int(sdata[u'sstartdate']))
                   for sid, handednesss, sex, sdata in rset
                   if u'sstartdate' in sdata.keys()]
    demog = pd.DataFrame(loaded_rset,
                         columns=['IID', 'Handedness', 'Gender', 'Age'])
    print demog.head()

    # save the dataframe in a csv
    outf = '/tmp/demog.csv'
    print ('Demo output file : see file : in {0}'.format(outf))
    with open(outf, 'w') as fp:
        demog.to_csv(fp, sep='\t', index=False)
