# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:39:25 2014

@author: hl237680

This script aims at selecting the subjects for building a new template.
This template will results from as many obese, overweight and normal people,
as many girls as boys, well distributed along the imaging centre cities.

INPUT: clinical data of IMAGEN subjects, including their weight status
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"

OUTPUT: distribution of IMAGEN subjects along weight status, gender and
        imaging centre city
    "/neurospin/brainomics/2013_imagen_bmi/data/template/
    subjects_distribution.txt"

"""


import os
import pandas as pd
import csv


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
TEMPLATE_DATA_PATH = os.path.join(DATA_PATH, 'template')


# Read clinical data on IMAGEN population
clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                  'population.csv'),
                                     index_col=0)

colnames = ['Gender de Feuil2',
            'ImagingCentreCity',
            'Status']

# Keep clinical data of interest for building a template with as many obese,
# overweight and normal people, as many girls as boys, well distributed
# along the different imaging centres
clinical_df = clinical_df[colnames]


###################
# Group selection #
###################

# Work on obese people
obese_group = clinical_df[clinical_df['Status'] == 'Obese']

female_obese_group = obese_group[obese_group['Gender de Feuil2'] == 'Female']
male_obese_group = obese_group[obese_group['Gender de Feuil2'] == 'Male']

London_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'LONDON']
Nottingham_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'NOTTINGHAM']
Dublin_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'DUBLIN']
Paris_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'PARIS']
Berlin_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'BERLIN']
Hamburg_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'HAMBURG']
Dresden_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'DRESDEN']
Mannheim_female_obese_group = female_obese_group[female_obese_group['ImagingCentreCity'] == 'MANNHEIM']

London_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'LONDON']
Nottingham_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'NOTTINGHAM']
Dublin_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'DUBLIN']
Paris_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'PARIS']
Berlin_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'BERLIN']
Hamburg_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'HAMBURG']
Dresden_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'DRESDEN']
Mannheim_male_obese_group = male_obese_group[male_obese_group['ImagingCentreCity'] == 'MANNHEIM']

print "Among the ", obese_group.shape[0], "obese subjects:"
print female_obese_group.shape[0], "are girls", male_obese_group.shape[0], "are boys."

print "\n Among the ", female_obese_group.shape[0], "obese girls:"
print London_female_obese_group.shape[0], "come from London:"
print London_female_obese_group.index.tolist(), "\n"
print Nottingham_female_obese_group.shape[0], "come from Nottingham:"
print Nottingham_female_obese_group.index.tolist(), "\n"
print Dublin_female_obese_group.shape[0], "come from Dublin:"
print Dublin_female_obese_group.index.tolist(), "\n"
print Paris_female_obese_group.shape[0], "come from Paris:"
print Paris_female_obese_group.index.tolist(), "\n"
print Berlin_female_obese_group.shape[0], "come from Berlin:"
print Berlin_female_obese_group.index.tolist(), "\n"
print Hamburg_female_obese_group.shape[0], "come from Hamburg:"
print Hamburg_female_obese_group.index.tolist(), "\n"
print Dresden_female_obese_group.shape[0], "come from Dresden:"
print Dresden_female_obese_group.index.tolist(), "\n"
print Mannheim_female_obese_group.shape[0], "come from Mannheim:"
print Mannheim_female_obese_group.index.tolist(), "\n"

print "\n Among the ", male_obese_group.shape[0], "obese boys:"
print London_male_obese_group.shape[0], "come from London:"
print London_male_obese_group.index.tolist(), "\n"
print Nottingham_male_obese_group.shape[0], "come from Nottingham:"
print Nottingham_male_obese_group.index.tolist(), "\n"
print Dublin_male_obese_group.shape[0], "come from Dublin:"
print Dublin_male_obese_group.index.tolist(), "\n"
print Paris_male_obese_group.shape[0], "come from Paris:"
print Paris_male_obese_group.index.tolist(), "\n"
print Berlin_male_obese_group.shape[0], "come from Berlin:"
print Berlin_male_obese_group.index.tolist(), "\n"
print Hamburg_male_obese_group.shape[0], "come from Hamburg:"
print Hamburg_male_obese_group.index.tolist(), "\n"
print Dresden_male_obese_group.shape[0], "come from Dresden:"
print Dresden_male_obese_group.index.tolist(), "\n"
print Mannheim_male_obese_group.shape[0], "come from Mannheim:"
print Mannheim_male_obese_group.index.tolist(), "\n"


# Work on overweight people
overweight_group = clinical_df[clinical_df['Status'] == 'Overweight']

female_overweight_group = overweight_group[overweight_group['Gender de Feuil2'] == 'Female']
male_overweight_group = overweight_group[overweight_group['Gender de Feuil2'] == 'Male']

London_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'LONDON']
Nottingham_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'NOTTINGHAM']
Dublin_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'DUBLIN']
Paris_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'PARIS']
Berlin_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'BERLIN']
Hamburg_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'HAMBURG']
Dresden_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'DRESDEN']
Mannheim_female_overweight_group = female_overweight_group[female_overweight_group['ImagingCentreCity'] == 'MANNHEIM']

London_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'LONDON']
Nottingham_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'NOTTINGHAM']
Dublin_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'DUBLIN']
Paris_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'PARIS']
Berlin_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'BERLIN']
Hamburg_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'HAMBURG']
Dresden_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'DRESDEN']
Mannheim_male_overweight_group = male_overweight_group[male_overweight_group['ImagingCentreCity'] == 'MANNHEIM']

print "\n \n Among the ", overweight_group.shape[0], "overweight subjects:"
print female_overweight_group.shape[0], "are girls", male_overweight_group.shape[0], "are boys."

print "\n Among the ", female_overweight_group.shape[0], "overweight girls:"
print London_female_overweight_group.shape[0], "come from London:"
print London_female_overweight_group.index.tolist(), "\n"
print Nottingham_female_overweight_group.shape[0], "come from Nottingham:"
print Nottingham_female_overweight_group.index.tolist(), "\n"
print Dublin_female_overweight_group.shape[0], "come from Dublin:"
print Dublin_female_overweight_group.index.tolist(), "\n"
print Paris_female_overweight_group.shape[0], "come from Paris:"
print Paris_female_overweight_group.index.tolist(), "\n"
print Berlin_female_overweight_group.shape[0], "come from Berlin:"
print Berlin_female_overweight_group.index.tolist(), "\n"
print Hamburg_female_overweight_group.shape[0], "come from Hamburg:"
print Hamburg_female_overweight_group.index.tolist(), "\n"
print Dresden_female_overweight_group.shape[0], "come from Dresden:"
print Dresden_female_overweight_group.index.tolist(), "\n"
print Mannheim_female_overweight_group.shape[0], "come from Mannheim:"
print Mannheim_female_overweight_group.index.tolist(), "\n"

print "\n Among the ", male_overweight_group.shape[0], "overweight boys:"
print London_male_overweight_group.shape[0], "come from London:"
print London_male_overweight_group.index.tolist(), "\n"
print Nottingham_male_overweight_group.shape[0], "come from Nottingham:"
print Nottingham_male_overweight_group.index.tolist(), "\n"
print Dublin_male_overweight_group.shape[0], "come from Dublin:"
print Dublin_male_overweight_group.index.tolist(), "\n"
print Paris_male_overweight_group.shape[0], "come from Paris:"
print Paris_male_overweight_group.index.tolist(), "\n"
print Berlin_male_overweight_group.shape[0], "come from Berlin:"
print Berlin_male_overweight_group.index.tolist(), "\n"
print Hamburg_male_overweight_group.shape[0], "come from Hamburg:"
print Hamburg_male_overweight_group.index.tolist(), "\n"
print Dresden_male_overweight_group.shape[0], "come from Dresden:"
print Dresden_male_overweight_group.index.tolist(), "\n"
print Mannheim_male_overweight_group.shape[0], "come from Mannheim:"
print Mannheim_male_overweight_group.index.tolist(), "\n"


# Work on normal people
normal_group = clinical_df[clinical_df['Status'] == 'Normal']

female_normal_group = normal_group[normal_group['Gender de Feuil2'] == 'Female']
male_normal_group = normal_group[normal_group['Gender de Feuil2'] == 'Male']

London_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'LONDON']
Nottingham_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'NOTTINGHAM']
Dublin_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'DUBLIN']
Paris_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'PARIS']
Berlin_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'BERLIN']
Hamburg_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'HAMBURG']
Dresden_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'DRESDEN']
Mannheim_female_normal_group = female_normal_group[female_normal_group['ImagingCentreCity'] == 'MANNHEIM']

London_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'LONDON']
Nottingham_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'NOTTINGHAM']
Dublin_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'DUBLIN']
Paris_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'PARIS']
Berlin_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'BERLIN']
Hamburg_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'HAMBURG']
Dresden_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'DRESDEN']
Mannheim_male_normal_group = male_normal_group[male_normal_group['ImagingCentreCity'] == 'MANNHEIM']

print "\n \n Among the ", normal_group.shape[0], "normal subjects:"
print female_normal_group.shape[0], "are girls", male_normal_group.shape[0], "are boys."

print "\n Among the ", female_normal_group.shape[0], "normal girls:"
print London_female_normal_group.shape[0], "come from London:"
print London_female_normal_group.index.tolist(), "\n"
print Nottingham_female_normal_group.shape[0], "come from Nottingham:"
print Nottingham_female_normal_group.index.tolist(), "\n"
print Dublin_female_normal_group.shape[0], "come from Dublin:"
print Dublin_female_normal_group.index.tolist(), "\n"
print Paris_female_normal_group.shape[0], "come from Paris:"
print Paris_female_normal_group.index.tolist(), "\n"
print Berlin_female_normal_group.shape[0], "come from Berlin:"
print Berlin_female_normal_group.index.tolist(), "\n"
print Hamburg_female_normal_group.shape[0], "come from Hamburg:"
print Hamburg_female_normal_group.index.tolist(), "\n"
print Dresden_female_normal_group.shape[0], "come from Dresden:"
print Dresden_female_normal_group.index.tolist(), "\n"
print Mannheim_female_normal_group.shape[0], "come from Mannheim:"
print Mannheim_female_normal_group.index.tolist(), "\n"

print "\n Among the ", male_normal_group.shape[0], "normal boys:"
print London_male_normal_group.shape[0], "come from London:"
print London_male_normal_group.index.tolist(), "\n"
print Nottingham_male_normal_group.shape[0], "come from Nottingham:"
print Nottingham_male_normal_group.index.tolist(), "\n"
print Dublin_male_normal_group.shape[0], "come from Dublin:"
print Dublin_male_normal_group.index.tolist(), "\n"
print Paris_male_normal_group.shape[0], "come from Paris:"
print Paris_male_normal_group.index.tolist(), "\n"
print Berlin_male_normal_group.shape[0], "come from Berlin:"
print Berlin_male_normal_group.index.tolist(), "\n"
print Hamburg_male_normal_group.shape[0], "come from Hamburg:"
print Hamburg_male_normal_group.index.tolist(), "\n"
print Dresden_male_normal_group.shape[0], "come from Dresden:"
print Dresden_male_normal_group.index.tolist(), "\n"
print Mannheim_male_normal_group.shape[0], "come from Mannheim:"
print Mannheim_male_normal_group.index.tolist(), "\n"



# Write results in a csv file
subjects_for_template_file_path = os.path.join(TEMPLATE_DATA_PATH,
                                               'subjects_distribution.txt')

with open(subjects_for_template_file_path, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

    spamwriter.writerow(['Among the']
                        + [obese_group.shape[0]]
                        + ['obese subjects: \n']
                        + [female_obese_group.shape[0]]
                        + ['are girls,']
                        + [male_obese_group.shape[0]]
                        + ['are boys. \n']
                        # Obese girls
                        + ['\n Among the']
                        + [female_obese_group.shape[0]]
                        + ['obese girls: \n']
                        + [London_female_obese_group.shape[0]]
                        + ['come from London: \n']
                        + [London_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Nottingham_female_obese_group.shape[0]]
                        + ['come from Nottingham: \n']
                        + [Nottingham_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Dublin_female_obese_group.shape[0]]
                        + ['come from Dublin: \n']
                        + [Dublin_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Paris_female_obese_group.shape[0]]
                        + ['come from Paris: \n']
                        + [Paris_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Berlin_female_obese_group.shape[0]]
                        + ['come from Berlin: \n']
                        + [Berlin_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Hamburg_female_obese_group.shape[0]]
                        + ['come from Hamburg: \n']
                        + [Hamburg_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Dresden_female_obese_group.shape[0]]
                        + ['come from Dresden: \n']
                        + [Dresden_female_obese_group.index.tolist()]
                        + ['\n']
                        + [Mannheim_female_obese_group.shape[0]]
                        + ['come from Mannheim: \n']
                        + [Mannheim_female_obese_group.index.tolist()]
                        + ['\n']
                        # Obese boys
                        + ['\n Among the']
                        + [male_obese_group.shape[0]]
                        + ['obese boys:\n ']
                        + [London_male_obese_group.shape[0]]
                        + ['come from London: \n']
                        + [London_male_obese_group.index.tolist()]
                        + ['\n']
                        + [Nottingham_male_obese_group.shape[0]]
                        + ['come from Nottingham: \n']
                        + [Nottingham_male_obese_group.index.tolist()]
                        + ['\n']
                        + [Dublin_male_obese_group.shape[0]]
                        + ['come from Dublin: \n']
                        + [Dublin_male_obese_group.index.tolist()]
                        + ['\n']
                        + [Paris_male_obese_group.shape[0]]
                        + ['come from Paris: \n']
                        + [Paris_male_obese_group.index.tolist()]
                        + ['\n']
                        + [Berlin_male_obese_group.shape[0]]
                        + ['come from Berlin: \n']
                        + [Berlin_male_obese_group.index.tolist(),]
                        + ['\n']
                        + [Hamburg_male_obese_group.shape[0]]
                        + ['come from Hamburg: \n']
                        + [Hamburg_male_obese_group.index.tolist()]
                        + ['\n']
                        + [Dresden_male_obese_group.shape[0]]
                        + ['come from Dresden: \n']
                        + [Dresden_male_obese_group.index.tolist()]
                        + ['\n']
                        + [Mannheim_male_obese_group.shape[0]]
                        + ['come from Mannheim: \n']
                        + [Mannheim_male_obese_group.index.tolist()]
                        + ['\n']
                        # Overweight people
                        + ['\n \n Among the']
                        + [overweight_group.shape[0]]
                        + ['overweight subjects: \n']
                        + [female_overweight_group.shape[0]]
                        + ['are girls,']
                        + [male_overweight_group.shape[0]]
                        + ['are boys. \n']
                        # Overweight girls
                        + ['\n Among the']
                        + [female_overweight_group.shape[0]]
                        + ['overweight girls: \n']
                        + [London_female_overweight_group.shape[0]]
                        + ['come from London: \n']
                        + [London_female_overweight_group.index.tolist()]
                        + ['\n']
                        + [Nottingham_female_overweight_group.shape[0]]
                        + ['come from Nottingham: \n']
                        + [Nottingham_female_overweight_group.index.tolist()]
                        + ['\n']
                        + [Dublin_female_overweight_group.shape[0]]
                        + ['come from Dublin: \n']
                        + [Dublin_female_overweight_group.index.tolist()]
                        + ['\n']
                        + [Paris_female_overweight_group.shape[0]]
                        + ['come from Paris: \n']
                        + [Paris_female_overweight_group.index.tolist()]
                        + ['\n']
                        + [Berlin_female_overweight_group.shape[0]]
                        + ['come from Berlin: \n']
                        + [Berlin_female_overweight_group.index.tolist(),]
                        + ['\n']
                        + [Hamburg_female_overweight_group.shape[0]]
                        + ['come from Hamburg: \n']
                        + [Hamburg_female_overweight_group.index.tolist()]
                        + ['\n']
                        + [Dresden_female_overweight_group.shape[0]]
                        + ['come from Dresden: \n']
                        + [Dresden_female_overweight_group.index.tolist()]
                        + ['\n']
                        + [Mannheim_female_overweight_group.shape[0]]
                        + ['come from Mannheim: \n']
                        + [Mannheim_female_overweight_group.index.tolist()]
                        + ['\n']
                        # Overweight boys
                        + ['\n Among the']
                        + [male_overweight_group.shape[0]]
                        + ['overweight boys: \n']
                        + [London_male_overweight_group.shape[0]]
                        + ['come from London: \n']
                        + [London_male_overweight_group.index.tolist()]
                        + ['\n']
                        + [Nottingham_male_overweight_group.shape[0]]
                        + ['come from Nottingham: \n']
                        + [Nottingham_male_overweight_group.index.tolist()]
                        + ['\n']
                        + [Dublin_male_overweight_group.shape[0]]
                        + ['come from Dublin: \n']
                        + [Dublin_male_overweight_group.index.tolist()]
                        + ['\n']
                        + [Paris_male_overweight_group.shape[0]]
                        + ['come from Paris: \n']
                        + [Paris_male_overweight_group.index.tolist()]
                        + ['\n']
                        + [Berlin_male_overweight_group.shape[0]]
                        + ['come from Berlin: \n']
                        + [Berlin_male_overweight_group.index.tolist(),]
                        + ['\n']
                        + [Hamburg_male_overweight_group.shape[0]]
                        + ['come from Hamburg: \n']
                        + [Hamburg_male_overweight_group.index.tolist()]
                        + ['\n']
                        + [Dresden_male_overweight_group.shape[0]]
                        + ['come from Dresden: \n']
                        + [Dresden_male_overweight_group.index.tolist()]
                        + ['\n']
                        + [Mannheim_male_overweight_group.shape[0]]
                        + ['come from Mannheim: \n']
                        + [Mannheim_male_overweight_group.index.tolist()]
                        + ['\n']
                        # Normal people
                        + ['\n \n Among the']
                        + [normal_group.shape[0]]
                        + ['normal subjects: \n']
                        + [female_normal_group.shape[0]]
                        + ['are girls,']
                        + [male_normal_group.shape[0]]
                        + ['are boys. \n']
                        # Normal girls
                        + ['\n Among the']
                        + [female_normal_group.shape[0]]
                        + ['normal girls: \n']
                        + [London_female_normal_group.shape[0]]
                        + ['come from London: \n']
                        + [London_female_normal_group.index.tolist()]
                        + ['\n']
                        + [Nottingham_female_normal_group.shape[0]]
                        + ['come from Nottingham: \n']
                        + [Nottingham_female_normal_group.index.tolist()]
                        + ['\n']
                        + [Dublin_female_normal_group.shape[0]]
                        + ['come from Dublin: \n']
                        + [Dublin_female_normal_group.index.tolist()]
                        + ['\n']
                        + [Paris_female_normal_group.shape[0]]
                        + ['come from Paris: \n']
                        + [Paris_female_normal_group.index.tolist()]
                        + ['\n']
                        + [Berlin_female_normal_group.shape[0]]
                        + ['come from Berlin: \n']
                        + [Berlin_female_normal_group.index.tolist(),]
                        + ['\n']
                        + [Hamburg_female_normal_group.shape[0]]
                        + ['come from Hamburg: \n']
                        + [Hamburg_female_normal_group.index.tolist()]
                        + ['\n']
                        + [Dresden_female_normal_group.shape[0]]
                        + ['come from Dresden: \n']
                        + [Dresden_female_normal_group.index.tolist()]
                        + ['\n']
                        + [Mannheim_female_normal_group.shape[0]]
                        + ['come from Mannheim: \n']
                        + [Mannheim_female_normal_group.index.tolist()]
                        + ['\n \n']
                        # Normal boys
                        + ['\n Among the']
                        + [male_normal_group.shape[0]]
                        + ['normal boys: \n']
                        + [London_male_normal_group.shape[0]]
                        + ['come from London: \n']
                        + [London_male_normal_group.index.tolist()]
                        + ['\n']
                        + [Nottingham_male_normal_group.shape[0]]
                        + ['come from Nottingham: \n']
                        + [Nottingham_male_normal_group.index.tolist()]
                        + ['\n']
                        + [Dublin_male_normal_group.shape[0]]
                        + ['come from Dublin: \n']
                        + [Dublin_male_normal_group.index.tolist()]
                        + ['\n']
                        + [Paris_male_normal_group.shape[0]]
                        + ['come from Paris: \n']
                        + [Paris_male_normal_group.index.tolist()]
                        + ['\n']
                        + [Berlin_male_normal_group.shape[0]]
                        + ['come from Berlin: \n']
                        + [Berlin_male_normal_group.index.tolist(),]
                        + ['\n']
                        + [Hamburg_male_normal_group.shape[0]]
                        + ['come from Hamburg: \n']
                        + [Hamburg_male_normal_group.index.tolist()]
                        + ['\n']
                        + [Dresden_male_normal_group.shape[0]]
                        + ['come from Dresden: \n']
                        + [Dresden_male_normal_group.index.tolist()]
                        + ['\n']
                        + [Mannheim_male_normal_group.shape[0]]
                        + ['come from Mannheim: \n']
                        + [Mannheim_male_normal_group.index.tolist()])