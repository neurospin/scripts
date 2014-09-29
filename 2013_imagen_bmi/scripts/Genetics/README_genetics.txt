#############################################################################

IMAGEN1265_SNPs_measurements.py:

This script generates a .csv file containing the genotype of IMAGEN subjects
for whom we have both neuroimaging and genetic data regarding SNPs of interest
at the intersection between SNPs referenced in the literature as from genes
robustly associated to BMI and SNPs read by the Illumina platform on IMAGEN
subjects.

INPUT:
- Genotype data from the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/2012_imagen_shfj/genetics/
    qc_sub_qc_gen_all_snps_common_autosome"
- SNPs referenced in the literature as strongly associated to the BMI
  (included in or near to BMI-associated genes):
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    genes_BMI_ob.xls"
- List of the 1.265 subjects for whom we have both neuroimaging and genetic
  data:
    "/neurospin/brainomics/2013_imagen_bmi/data/subjects_id.csv"

OUTPUT:
- List of SNPs referenced in the literature as strongly associated to the BMI:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    BMI_SNPs_names_list.csv"
- List of SNPs at the intersection between BMI-associated SNPs referenced in
  the literature and SNPs read by the Illumina platform:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    snp_from_liter_in_study.csv"
- Genetic measures on SNPs of interest
    "/neurospin/brainomics/2013_imagen_bmi/data/
    BMI_associated_SNPs_measures.csv"

#############################################################################

all_SNPs_included_in_BMI_genes_measurements.py:

This script generates a .csv file containing the genotype of IMAGEN subjects
for whom we have both neuroimaging and genetic data regarding SNPs of interest
at the intersection between all SNPs included in BMI-associated genes and
SNPs read by the Illumina platform on IMAGEN subjects.

NB: Difference with the script IMAGEN1265_SNPs_measurements.py is that here,
    we do not only focus on SNPs referenced in the literature as associated
    to the BMI.

INPUT:
- Genotype data from the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/2012_imagen_shfj/genetics/
    qc_sub_qc_gen_all_snps_common_autosome"
- List of all SNPs included in BMI-associated genes:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/Sulci_SNPs/
    all_SNPs_within_BMI_associated_genes.snp"
- List of the 1.265 subjects for whom we have both neuroimaging and genetic
  data:
    "/neurospin/brainomics/2013_imagen_bmi/data/subjects_id.csv"

OUTPUT:
- Intersection between all SNPs included in BMI-associated genes and SNPs
  read by the Illumina platform for the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    all_Illumina_SNPs_from_BMI_genes.csv"
- Genetic measures on SNPs of interest
    "/neurospin/brainomics/2013_imagen_bmi/data/
    BMI_associated_all_SNPs_measures.csv"

#############################################################################
