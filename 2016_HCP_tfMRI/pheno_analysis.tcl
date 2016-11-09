
proc pheno_analysis {working_dir pheno trait phen_file} {
    
    cd /neurospin/brainomics/2016_HCP/$working_dir 
    exec mkdir -p $pheno/$trait
    outdir /neurospin/brainomics/2016_HCP/$working_dir/$pheno/$trait
    cd /neurospin/brainomics/2016_HCP/$working_dir/$pheno/$trait/pedigree
    field id IID
    exec python /home/yl247234/gits/scripts/2016_HCP/convert_pheno_to_solar.py -i $phen_file
    load phenotypes $phen_file
    define analyze_$trait = inormal_$trait 
    trait analyze_$trait
    # house
    covariate age^1,2#sex
    covariate sex
    covariate etiv
    polygenic -screen
}
