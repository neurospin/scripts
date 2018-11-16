function launch_calculate_all(subjectdir)    

    %Lifile='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\03-Concentration\Patient01_KBgrid_MODULE_Echo0_filtered.nii';
    %Anat7Tfile='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy7T\t1_mpr_tra_iso1_0mm.nii';
    %Anat3Tfile='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy3T\t1_weighted_sagittal_1_0iso.nii';
    %Litranslation='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy7T\1H_TO_Li_f.trm';
    %Lioutputdircoreg='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\06_Final';
    %Lioutputdirmni='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\06_Final';
    %allsubjs=dir('V:\people\Jacques\BIPLi7\February_18\2*')
    
    %addpath("/volatile/spm12")
    %addpath("/volatile/DISTANCE")
    keepniifiles=3;
    Currentfolder=pwd;
    if contains(subjectdir,'/')
        splitfolder=strsplit(subjectdir,'/');
        splitfolder=strjoin(splitfolder(1:end-3),'/');
    elseif contains(subjectdir,'\')
        splitfolder=strsplit(subjectdir,'\');
        splitfolder=strjoin(splitfolder(1:end-3),'\');
    end
    subjectdir=string(subjectdir);
    %TPMfile='C:\Users\js247994\Documents\FidDependencies\spm12\tpm\TPM.nii'; 
    TPMfile=fullfile(splitfolder,'Masks','TPM.nii'); 
    if exist(fullfile(Currentfolder,'info_pipeline','segmentsubjectspm.mat'),'file')
        segmentfile=fullfile(Currentfolder,'info_pipeline','segmentsubjectspm.mat'); %Later do a thing that writes the .mat files;
    elseif exist(fullfile(Currentfolder,'info_pipeline','segmentsubjectspm.txt'),'file')
        movefile (fullfile(Currentfolder,'info_pipeline','segmentsubjectspm.txt'),fullfile(Currentfolder,'info_pipeline','segmentsubjectspm.mat'))
        segmentfile=fullfile(Currentfolder,'info_pipeline','segmentsubjectspm.mat');
    end
    Lioutputdirmni=fullfile(subjectdir,'TPI','Reconstruct_gridding','06-MNIspace');
    Lioutputdir7T=fullfile(subjectdir,'TPI','Reconstruct_gridding','04-7Tanatspace');
    Lioutputdir3T=fullfile(subjectdir,'TPI','Reconstruct_gridding','05-3Tanatspace');    
    %Litranslation=strcat(subjectdir,'\Anatomy7T\1H_TO_Li_f.trm');
    Anat3Tfile=fullfile(subjectdir,'Anatomy3T','t1_weighted_sagittal_1_0iso.nii');
    Anat7Tfile=fullfile(subjectdir,'Anatomy7T','t1_mpr_tra_iso1_0mm.nii');
    LifilesS=dir(fullfile(subjectdir,'TPI','Reconstruct_gridding','03-Filtered','*nii'));
    
    i=1;
    Lifiles=cell(size(LifilesS,1),1);
    for Lifile=LifilesS'
        Lifiles{i}=fullfile(subjectdir,'TPI','Reconstruct_gridding','03-Filtered',Lifile.name);
        i=i+1;
    end
    
    i=1;
    otherfiles=[];
    trufifiles=[];
    TrufifilesS=dir(fullfile(subjectdir,'Trufi','03-Filtered','*nii'));
    for trufifile=TrufifilesS'
        if ~contains(trufifile.name,'3T')
            trufifiles{i,1}=fullfile(subjectdir,'Trufi','03-Filtered',trufifile.name);
            i=i+1;
        end
    end   
    
    i=1;
    if exist(fullfile(subjectdir,'Field_mapping'),'dir')
        if exist(fullfile(subjectdir,'Field_mapping','field_mapping_rad.nii'),'file')
            otherfiles{i,1}=fullfile(subjectdir,'Field_mapping','field_mapping_rad.nii');
            i=i+1;
        end
        if exist(fullfile(subjectdir,'Field_mapping','Field_mapping_mag','field_mapping.nii'),'file')
            otherfiles{i,1}=fullfile(subjectdir,'Field_mapping','Field_mapping_mag','field_mapping.nii');
            i=i+1;
        end
        if exist(fullfile(subjectdir,'Field_mapping','Field_mapping_phase','field_mapping_phase.nii'),'file')
            otherfiles{i,1}=fullfile(subjectdir,'Field_mapping','Field_mapping_phase','field_mapping_phase.nii');
            i=i+1;
        end
    end
    
    [transmat,coregmat,deform_field,normfile]=calculate_all_00(Lifiles,otherfiles,Lioutputdir7T,Lioutputdir3T,Lioutputdirmni,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles);
    if ~isempty(trufifiles)
        trufioutput3T=fullfile(subjectdir,'Trufi','04-3Tanatspace');
        trufioutputMNI=fullfile(subjectdir,'Trufi','05-MNIspace');
        apply_transform_7TtoMNI(trufifiles,trufioutput3T,trufioutputMNI,coregmat,deform_field,normfile);
    end