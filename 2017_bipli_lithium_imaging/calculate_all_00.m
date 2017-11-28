function transmat=calculate_all_00(Lifile,Lioutputdircoreg,Lioutputdirmni,Anat7Tfile,Anat3Tfile,Litranslation,TPMfile,keepniifiles)
    
    %segmentfile='C:\Users\js247994\Documents\Bipli2\Test\segmentsubjectspm12_1.mat'; %Later do a thing that finds it automatically;
    Lifile='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\03-Concentration\Patient01_KBgrid_MODULE_Echo0_filtered.nii';
    Anat7Tfile='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy7T\t1_mpr_tra_iso1_0mm.nii';
    Anat3Tfile='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy3T\t1_weighted_sagittal_1_0iso.nii';
    Litranslation='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy7T\1H_TO_Li_f.trm';
    Lioutputdir='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\06_Final';
    TPMfile='C:\Users\js247994\Documents\FidDependencies\spm12\tpm\TPM.nii';
    keepniifiles=3;
    
    [Lidir,~,~]=fileparts(Lifile);
    
    transmat=calculate_translation_mat_01(Lifile,Anat7Tfile,Litranslation);
    coregmat=calculate_coreg_mat_02(Anat7Tfile,Anat3Tfile);   
    newLimat=coregmat*transmat;
    
    Lispm=spm_vol(Lifile);
    newLispm=Lispm;
    newLispm.mat=newLimat;
    newLispm.fname=strcat(Lidir,'\tempLifile.nii');
    spm_write_vol(newLispm,spm_read_vols(Lispm));
    
    spmfilesdir=strcat(pwd,'\info_pipeline');
    segmentfile=strcat(spmfilesdir,'\segmentsubjectspm.mat');
    deform_field=calculate_deform_field_03(Anat3Tfile,segmentfile,TPMfile,keepniifiles);
    
    normfile=strcat(spmfilesdir,'\normwritespm.mat');
    apply_deform_field_04(newLispm.fname,Lioutputdir,deform_field,normfile);
    
end