COV="/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/test.cov"
PHE="/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/testLhippo.phe"
GENO="/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/qc_subjects_qc_genetics_all_snps_wave2"
SNPS="/neurospin/brainomics/2015_hippo_l1_gl_ovl/data/testList.snp"
OUT="/neurospin/brainomics/2015_hippo_l1_gl_ovl/results/test"

plink --noweb --bfile $GENO --covar $COV --pheno $PHE  --extract $SNPS \
    --out $OUT \
    --linear
