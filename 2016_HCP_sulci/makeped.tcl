proc makeped {working_dir} {
    field id id
    cd /neurospin/brainomics/2016_HCP/
    exec mkdir -p $working_dir    
    cd $working_dir
    exec mkdir -p pedigree
    cd pedigree
    load pedigree /neurospin/brainomics/2016_HCP/pedigree/HCP_pedigree.csv
    cd ..
}
