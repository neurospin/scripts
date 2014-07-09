SNP="/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/2012_imagen_shfj/genetics/qc_sub_qc_gen_all_snps_common_autosome"
#SUBSET="sorted_subject_listNonNAN.csv"
PHENO="phenotype_final.csv"
COVAR="covarGenderPds.cov"
#COVAR="covarGenderSitePds.cov"


OUT="MaxLR-dominant-covarGenderPDS"
plink --noweb --bfile $SNP --covar $COVAR --pheno $PHENO --all-pheno \
    --snp rs13107325 --out $OUT \
    --linear --dominant
