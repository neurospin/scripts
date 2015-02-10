SNP="/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/2012_imagen_shfj/genetics/qc_sub_qc_gen_all_snps_common_autosome"
#SUBSET="sorted_subject_listNonNAN.csv"
DATA=/neurospin/brainomics/2015_asym_sts/data
PHENO=$DATA/sts_asym_rightonly.phe
COVAR=$DATA/sts_gender_centre.cov


OUT=$DATA/sts-sillons_rightonly
plink --noweb --bfile $SNP --covar $COVAR --pheno $PHENO --all-pheno \
    --out $OUT \
    --linear
