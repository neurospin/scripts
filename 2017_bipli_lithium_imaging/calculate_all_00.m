function transmat=calculate_all_00(Lifiles,other7Tfiles,Lioutputdir7T,Lioutputdir3T,Lioutputdirmni,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles)
       
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
        deform_field=calculate_deform_field_03(string(Anat3Tfile),segmentfile,TPMfile,keepniifiles);   
        Currentfolder=pwd;
        if exist(fullfile(Currentfolder,'info_pipeline','normwritespm.mat'),'file')
            normfile=fullfile(Currentfolder,'info_pipeline','normwritespm.mat'); %Later do a thing that finds it automatically;
        elseif exist(fullfile(Currentfolder,'info_pipeline','normwritespm.txt'),'file')
            movefile (fullfile(Currentfolder,'info_pipeline','normwritespm.txt'),fullfile(Currentfolder,'info_pipeline','normwritespm.mat'))
            normfile=fullfile(Currentfolder,'info_pipeline','normwritespm.mat');
        end
        for Lifile=Lifiles'
            Lispm=spm_vol(char(Lifile{1}));
            [~,filename,ext]=fileparts(Lifile{1});
            filename=string(filename);
            ext=string(ext);
            newLispm7T=Lispm;
            newLispm7T.mat=transmat;
            newLispm7T.fname=char(fullfile(Lioutputdir7T,filename+ext));
            newLispm3T=Lispm;
            newLispm3T.mat=Li3Tmat;
            newLispm3T.fname=char(fullfile(Lioutputdir3T,filename+ext));
            spm_write_vol(newLispm7T,spm_read_vols(Lispm));
            spm_write_vol(newLispm3T,spm_read_vols(Lispm));        
            apply_deform_field_04(newLispm3T.fname,Lioutputdirmni,deform_field,normfile);
        end
        for otherfile=other7Tfiles'
            otherfilespm=spm_vol(char(otherfile{1}));
            [dir,filename,ext]=fileparts(otherfile{1});
            newotherspm3T=otherfilespm;
            newotherspm3T.mat=coregmat*otherfilespm.mat;
            newotherspm3T.fname=char(fullfile(dir,filename+'_in3T'+ext));
            spm_write_vol(newotherspm3T,spm_read_vols(otherfilespm));        
            apply_deform_field_04(newotherspm3T.fname,dir,deform_field,normfile);
        end
    else
        display('Empty subject folder, cannot launch process')
    end
    
end