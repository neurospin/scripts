

proc makeped {working_dir pheno trait} {
    cd /neurospin/brainomics/2016_HCP/
    exec mkdir -p $working_dir
    cd $working_dir
    exec mkdir -p $pheno
    cd $pheno
    exec mkdir -p $trait  
    cd $trait 
    exec mkdir -p pedigree
    cd pedigree
    field id id
    load pedigree /neurospin/brainomics/2016_HCP/pedigree/HCP_pedigree_with_hid.csv
    cd ..
}
