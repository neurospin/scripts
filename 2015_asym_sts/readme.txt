# Commandes au 19 jan 2015 pour travail sur STs.

#############
cd /neurospin/brainomics/2015_asym_sts/data

############# create the different pheno and cov
python ~/gits/scripts/2015_asym_sts/01-create-pheno-cov.py

############# do the plink
python ~/gits/scripts/2015_asym_sts/02-script-plink.py -p sts_asym_rightonly.phe
python ~/gits/scripts/2015_asym_sts/02-script-plink.py -p sts_asym.phe


############# performe p-val analysis
python ~/gits/scripts/2015_asym_sts/03-get-pval.py -l sts_asym_rightonly_sts_gender_centre.STs_asym.assoc.linear
python ~/gits/scripts/2015_asym_sts/03-get-pval.py -l sts_asym_sts_gender_centre.STs_asym.assoc.linear


############# Pour interroger les informations de variants
http://www.ncbi.nlm.nih.gov/gap/phegeni et entrer la liset de SNPs.



###############################################################################
# As of 15 oct 2015
# Commands to run
#

# Creation of the pheno file (local for the project) and cov (in imagen_central)
# this command use qc_subjects.py forits method 
# "_get_sulci_for_subject_with_genetics"
python  create_pheno_cov.py

# Univariate study - plink based.
python script-plink.py
python parse_pval_plink.py
