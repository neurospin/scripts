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
import numpy as np
from optparse import OptionParser

# For measured genotypes For imputed genotypes
from genibabel import imagen_genotype_measure, imagen_imputed_genotype_measure

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


def main():
    """ Main function
    """
    # ancillary
    def get_comma_separated_args(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    #default
    fout = '/tmp/genes.csv'

    #
    parser = OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
    parser.add_option('-o', '--output',
                      help='path to outputfile file',
                      default=fout, type="string")
    parser.add_option('-g', '--genelist',
                      type='string',
                      help='[REQUIRED] comma separated list of legal gene names',
                      action='callback',
                      callback=get_comma_separated_args,
                      dest="genes")
    (options, args) = parser.parse_args()

    return options, args

if __name__ == '__main__':
    """ query_subject
    """
    options, args = main()

    if options.genes is None:
        print ('--genelist option: [REQUIRED] comma separated list of '
               'legal gene names')

    #set login, password, DB
    url = "https://imagen2.cea.fr/database/"
    login, password = get_credentials()
    if (login is None) or (password is None):
        print 'login: ',
        login = sys.stdin.readline().rstrip()
        password = getpass.getpass('password: ')

    # query DB for measured data
    print 'Requesting measured data ',
    genotypes = imagen_genotype_measure(login,
                                        password,
                                        gene_names=options.genes)

    # export data to CSV
    genotypes.csv_export(options.output)
    print '...Done'

    #query DB for imputed data
    print 'Requesting measured data ',
    genotypes = imagen_imputed_genotype_measure(login,
                                    password,
                                    gene_names=options.genes)

    # store in a pandas dataframe
    print 'Requesting imputed data ',

    imput_out = os.path.splitext(options.output)
    imput_out = imput_out[0] + '_imputed' + imput_out[1]
    genotypes.csv_export(imput_out)
    print '...Done'
