# -*- coding: utf-8 -*-
from __future__ import print_function
# system imports
import os
import datetime
from glob import glob

import csv
import xlrd

import shutil
import json
import pandas as pd

#
ROOT = '/neurospin/psy_sbox/am_hbn'
PHENO = 'phenotypic_scores'


def build_battery(psy_quest_files):
    """ Populate a dict from the list of csv files to read

    Parameters
    ----------
    psy_quest_files: list
        a list of path name to csv format questionnaires.

    Returns
    -------
    battery : dictionary
        embark in dictionary format the information read from csv.
        The key value is  < questionnary names, subdict>.
        The key of th subdicts are : ['timestamp', 'quest_fname', 'df']

    """
    battery = dict()

    # crawl the list and build main battery entry
    for quest_fname in psy_quest_files:
        tmp = os.path.basename(quest_fname).split('.')[0]
        timestamp = tmp.split('_')[-1]  # -1 mean the last in the least
        my_key = '_'.join(tmp.split('_')[1:-1])
        #
        print(my_key, ',', end='')
        battery[my_key] = dict()
        battery[my_key]['timestamp'] = timestamp
        battery[my_key]['quest_fname'] = quest_fname
    print('\n\n')

    # potential quetionnaire to remove
    pop_list = []
    for quest_name in battery:
        quest_fname = battery[quest_name]['quest_fname']

        # hard coded lines to eliminate from read_csv scop
        # line starting with ""ID", ... the second ie line = 1 starting from 0
        tmp = pd.read_csv(quest_fname, skiprows=[1])

        # inspect non fatal errors
        # first is EID within the colums if not suppr this questionnaire
        if 'EID' not in tmp.keys():
            print("Warning: No EID in {}".format(quest_name))
            pop_list.append(quest_name)
            continue

        # inspect non fatal errors
        # suppress trivial duplicates (all fields are identical)
        dupl_index_num = sum(tmp.duplicated(keep=False))
        if dupl_index_num != 0:
            print("Warning: Fully identical items are {} in {}".format(
                                                        dupl_index_num,
                                                        quest_name))
            tmp = tmp.drop_duplicates(keep='first')
        
        # inspect non fatal errors
        # suppress trivial duplicates (all fields are identical)
        rows_with_nan_eid = sum(pd.isnull(tmp.EID))
        if rows_with_nan_eid != 0:
            print("Warning: Row with NaN EID are {} in {}".format(
                                                        rows_with_nan_eid,
                                                        quest_name))
            tmp = tmp.dropna(subset=['EID'])
        
        # now assign to the dict entry
        battery[quest_name]["df"] = tmp

    # suppress the questionnaire listed in pop list
    for i in pop_list:
        battery.pop(i)

    return battery


def inspect_problematic_dupl(battery):
    """ Parse the results dir following dedicated logics

    Parameters
    ----------
    battery: dict()
        embark in dictionary format the information read from csv.
        The key value is  < questionnary names, subdict>.
        The key of th subdicts are : ['timestamp', 'quest_fname', 'df']

    Returns
    -------
    battery : dict()
        same with embarked df free of duplicate

    """

    for quest_name in battery:
        tmp = battery[quest_name]['df'].copy()
        dupl_ind = tmp.duplicated(subset='EID', keep=False)
        if sum(dupl_ind) != 0:
            print(("Error: Problematic duplcates in {}, "
                   "first of the dupl is kept").format(quest_name))
            print(tmp[dupl_ind].head(2))
            battery[quest_name]['df'] = tmp.drop_duplicates(subset='EID',
                                                            keep='first')

    return battery

"""
Script start
"""
# forge pathes
pheno_dir = os.path.join(ROOT, PHENO)

# list neuropsy
psy_quest_files = glob(os.path.join(pheno_dir, '*.csv'))

# battery will contain info from all the current questionnaires
# create a dict and populate it
battery = build_battery(psy_quest_files)

# inspect problematic duplicates: Same EID but different values for colums
battery = inspect_problematic_dupl(battery)

# create a pandas Data Frame
# using long (stacked) format 'eid', 'var_name', 'var_value'
columns = ['eid', 'var_name', 'var_val']
avail_data = pd.DataFrame(columns=columns)
for quest_name in battery:
    # for each questionnaires with name quest_name
    # create 3 lists/series corresponding
    s1 = battery[quest_name]['df']['EID']
    s2 = [quest_name] * len(s1)
    s3 = [True] * len(s1)

    # now append them (as a list of lists) to avail_data DataFrame
    avail_data = \
        avail_data.append(pd.DataFrame(dict(eid=s1, var_name=s2, var_val=s3)))

# print
print(avail_data.head())

# pivot the stacked dataframe in a wide format
avail_data = avail_data.pivot(index='eid', columns='var_name', values='var_val')

print(avail_data.sum(axis=0).to_string())
