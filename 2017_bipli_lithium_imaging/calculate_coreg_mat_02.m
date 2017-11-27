function coregmat=calculate_coreg_mat_02(anatorigfile,anatreffile)

    coregmat=inv(spm_matrix(spm_coreg(anatreffile,anatorigfile)));
    
end
