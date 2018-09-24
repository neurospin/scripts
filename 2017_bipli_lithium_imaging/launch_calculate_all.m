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
    otherfilesS=dir(fullfile(subjectdir,'Trufi','*nii'));
    i=1;
    Lifiles=cell(size(LifilesS,1),1);
    for Lifile=LifilesS'
        Lifiles{i}=fullfile(subjectdir,'TPI','Reconstruct_gridding','03-Filtered',Lifile.name);
        i=i+1;
    end
    i=1;
    otherfiles=[];
    for otherfile=otherfilesS'
        if ~contains(otherfile.name,'3T')
            otherfiles{i}=fullfile(subjectdir,'Trufi',otherfile.name);
            i=i+1;
        end
    end
    
    calculate_all_00(Lifiles,otherfiles,Lioutputdir7T,Lioutputdir3T,Lioutputdirmni,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles);