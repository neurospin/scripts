#############################################################################

weigths_map_vizu.py: Edouard Duchesnay's tool to mesh beta maps.

#############################################################################

write_cofounds_file.py:

Generation of a dataframe containing cofounds of non interest (i.e. gender,
imaging city centre, tiv_gaser and mean pds) for the 745 subjects who passed
the quality control on sulci data for further use with Plink.


BEWARE!!!
- the first two columns must be IID and FID
- categorical variables require dummy coding


INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"
    clinical data on IMAGEN population (.xls initial data file)
- "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_depthMax_df.csv":
    sulci depthMax after quality control

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/Sulci_SNPs/
    confound_Gender_Centre_TIV_PDS_745id.cov:"
    .cov file with FID - IID - Gender - Centre - TIV - PDS

#############################################################################


#############
# Phenotype #
#############

BMI_phenotype_norm-ob.py:

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects with normal or obese status, for further
use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI_norm-ob_groups.phe"

Only normal and obese teenagers (i.e. 991 subjects) among the 1.265 subjects
for whom we have both neuroimaging (i.e. masked_images) and genetic data
have been selected.

#############################################################################

BMI_phenotype_normal_group.py

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects with normal status, for further use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI_norm_group.phe"

Only normal teenagers (i.e. 910 subjects) among the 1.265 subjects for whom
we have both neuroimaging and genetic data have been selected.

#############################################################################

BMI_phenotype_overweight-obese.py

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects with overweight or obese status, for further
use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI_o-o_groups.phe"

Only the 242 subjects (overweight and obese teenagers) among the 1265 for
whom we have both neuroimaging (i.e. masked_images) and genetic data have
been selected.

#############################################################################

BMI_phenotype_sulci_norm-ob.py

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects with normal or obese status, for further
use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI_norm-ob_groups.phe"

Only normal and obese teenagers (i.e. 991 subjects) among the 1.265 subjects
for whom we have both neuroimaging (i.e. masked_images) and genetic data
have been selected.

#############################################################################

BMI_phenotype.py

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects of interest, for further use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI.phe"

Only the 1265 subjects for which we have neuroimaging data (masked_images)
are selected among the total number of subjects.

#############################################################################


############
# Template #
############

00_get_demographics.py:

This script aims at selecting the subjects for building a new template.
This template will result from as many obese, overweight and normal people,
as many girls as boys, well distributed along the imaging centre cities.

INPUT: clinical data of IMAGEN subjects, including their weight status
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"

OUTPUT: distribution of IMAGEN subjects along weight status, gender and
        imaging centre city
    "/neurospin/brainomics/2013_imagen_bmi/data/template/
    subjects_distribution.txt"

#############################################################################

01_images_copy_for_tpm.py

This script copies images in nifti compressed format selected to build a
new template in a temporary file.
Images have been selected based on the output file generated by the script
"~/gits/scripts/2013_imagen_bmi/scripts/Tools/population_demographics.py".
They will then be extracted and used to build a new template via Matlab
SPM8 module.

OUTPUT_DIR:
    "/neurospin/tmp/hl/"

#############################################################################

02_build_homogeneous_tpm.py

Compute the mean of the images obtained after new_segment segmentation (wmc)
for the six tissue classes to build the new TPM.

This TPM is then normalized and saved under:
/neurospin/tmp/hl/TPM.nii

#############################################################################