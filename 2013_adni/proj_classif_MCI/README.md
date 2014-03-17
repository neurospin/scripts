The goal of the project is to study the performances of CONESTA with logitstic regression on the classification of MCI patients (MCIc vs MCInc).

Data
====

We use the quality-controlled segmented images from ADNI for which he grade is A or B.
The images are not smoothed.
From this we extract the MCIc and MCInc subpopulations.
There are  186 subjects: 121 MCInc and 65 MCIc.

We use the groups from Cuingnet et al. 2010 to split the population in two samples.
There are 92 training and 94 testing subjects.

The groups are as follows:
MCInc training    60
      testing     61
MCIc  training    32
      testing     33

In the response vector (y), MCInc is coded 0, MCIc is coded 1.

Scripts
=======

00_create_population_file_and_univariate_analysis.py: create file population.csv and file for univariate analysis.
01_build_dataset.py: create dataset
