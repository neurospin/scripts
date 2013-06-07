scripts
=======

Sharing and versioning processing scripts of studies


Introduction
============

This repository aims to share scripts for Neuroimaging and Genetic data analysis.

for *studies*: **version and share scripts** to **process** individidual ( **imaging and genetic** ) **data** 
as well as scripts for group-wise statistical/machine learning analysis. 


Remarks
-------

- Scripts are not intend to work or compile for other. Just provide softwares/libraries dependances.
- Keep it lightweight: avoid binary files such as pdf, doc, etc. 

Global organization
===================

for each study create a directory such:

YYYY_study/
  doc/
  scripts/


Example:
~~~~~~~~

  2012_ABIDE/     # Study name prefixed by the year
    scripts/   
      01_unpack_and_organize_data.sh
        readme.[rst|txt]
        01_segmentation_with_spm.m
        02_segmentation_with_fsl.sh
        03_quality-control_spm-vs-fsl.py
        04_quality-control_plot_tissue-proportion-per-site.R
        ...
  doc/
