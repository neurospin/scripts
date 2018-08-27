function coregmat=calculate_coreg_mat_02(anatorigfile,anatreffile)

    coregmat=inv(spm_matrix(spm_coreg(char(anatreffile),char(anatorigfile))));
    
end
