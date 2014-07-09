SNP="/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/2012_imagen_shfj/genetics/qc_sub_qc_gen_all_snps_common_autosome"
#SUBSET="sorted_subject_listNonNAN.csv"
PHENO="phenotype_final_covarGenderPDS_rs7182018.csv"
COVAR="covarGenderPds.cov"
#COVAR="covarGenderSitePds.cov"


OUT="MaxLR-rs7182018-linear-covarGenderPDS"
plink --noweb --bfile $SNP --covar $COVAR --pheno $PHENO --all-pheno \
    --snp rs7182018 --out $OUT \
    --linear
	

#--linear --dominant	#dominant model
#--linear	#linear model
