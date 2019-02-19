function [transmat,coregmat,deform_field,deform_field_inv]=calculate_all_00(Lifiles,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles)
       
    if ~isempty(Lifiles)
        transmat=calculate_translation_mat_01(Lifiles{1});%,Anat7Tfile,Litranslation);
        coregmat=calculate_coreg_mat_02(Anat7Tfile,Anat3Tfile);   
        Li3Tmat=coregmat*transmat;
        Anat7Tspm=spm_vol(char(Anat7Tfile));
        Anat7Tin3Tspm=Anat7Tspm;
        Anat7Tin3Tspm.mat=coregmat*Anat7Tspm.mat;
        [Anat7Tdir,Anat7Tfilename,ext]=fileparts(Anat7Tfile);
        Anat7Tin3Tspm.fname=char(fullfile(Anat7Tdir,string(Anat7Tfilename)+'_3Tspace'+ext));
        spm_write_vol(Anat7Tin3Tspm,spm_read_vols(Anat7Tspm));
        [deform_field,deform_field_inv]=calculate_deform_field_03(string(Anat3Tfile),segmentfile,TPMfile,keepniifiles);   
        %Currentfolder=pwd;
        %if exist(fullfile(Currentfolder,'info_pipeline','normwritespm.mat'),'file')
        %    normfile=fullfile(Currentfolder,'info_pipeline','normwritespm.mat'); %Later do a thing that finds it automatically;
        %elseif exist(fullfile(Currentfolder,'info_pipeline','normwritespm.txt'),'file')
        %    movefile (fullfile(Currentfolder,'info_pipeline','normwritespm.txt'),fullfile(Currentfolder,'info_pipeline','normwritespm.mat'))
        %    normfile=fullfile(Currentfolder,'info_pipeline','normwritespm.mat');
        %end
    else
        disp('Empty subject folder, cannot launch process')
    end
end