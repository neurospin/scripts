function transmat=calculate_all_00(Lifiles,Lioutputdir7T,Lioutputdir3T,Lioutputdirmni,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles)
       
    transmat=calculate_translation_mat_01(Lifiles{1});%,Anat7Tfile,Litranslation);
    coregmat=calculate_coreg_mat_02(Anat7Tfile,Anat3Tfile);   
    Li3Tmat=coregmat*transmat;
    
    Anat7Tspm=spm_vol(Anat7Tfile);
    Anat7Tin3Tspm=Anat7Tspm;
    Anat7Tin3Tspm.mat=coregmat*Anat7Tspm.mat;
    [Anat7Tdir,Anat7Tfilename,ext]=fileparts(Anat7Tfile);
    Anat7Tin3Tspm.fname=strcat(Anat7Tdir,'\',Anat7Tfilename,'_3Tspace',ext);
    spm_write_vol(Anat7Tin3Tspm,spm_read_vols(Anat7Tspm));
    
    deform_field=calculate_deform_field_03(Anat3Tfile,segmentfile,TPMfile,keepniifiles);   
    Currentfolder=pwd;
    normfile=strcat(Currentfolder,'\info_pipeline\normwritespm.mat');
    for Lifile=Lifiles'
        Lispm=spm_vol(Lifile{1});
        [~,filename,ext]=fileparts(Lifile{1});
        newLispm7T=Lispm;
        newLispm7T.mat=transmat;
        newLispm7T.fname=strcat(Lioutputdir7T,'\',filename,ext);
        newLispm3T=Lispm;
        newLispm3T.mat=Li3Tmat;
        newLispm3T.fname=strcat(Lioutputdir3T,'\',filename,ext);
        spm_write_vol(newLispm7T,spm_read_vols(Lispm));
        spm_write_vol(newLispm3T,spm_read_vols(Lispm));        
        apply_deform_field_04(newLispm3T.fname,Lioutputdirmni,deform_field,normfile);
    end
    
end