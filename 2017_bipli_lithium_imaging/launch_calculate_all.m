function [transmat,coregmat,deform_field,deform_field_inv,normfile]=launch_calculate_all(subjectdir,makeQuantif)    

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
    %Litranslation=strcat(subjectdir,'\Anatomy7T\1H_TO_Li_f.trm');
    Anat3Tdir=fullfile(subjectdir,'Anatomy3T');
    Anat3Tfile=fullfile(Anat3Tdir,'t1_weighted_sagittal_1_0iso.nii');
    Anat7Tdir=fullfile(subjectdir,'Anatomy7T');
    Anat7Tfile=fullfile(Anat7Tdir,'t1_mpr_tra_iso1_0mm.nii');
    LifilesS=dir(fullfile(subjectdir,'TPI','Reconstruct_gridding','01-Raw','*nii'));
    
    i=1;
    Lifiles=cell(size(LifilesS,1),1);

    for Lifile=LifilesS'
        Lifiles{i}=fullfile(subjectdir,'TPI','Reconstruct_gridding','01-Raw',Lifile.name);
        i=i+1;
    end    
    
    trufifiles=[];
    TrufifilesS=dir(fullfile(subjectdir,'Trufi','01-Raw','*nii'));
    i=1;
    for trufifile=TrufifilesS'
        trufifiles{i,1}=fullfile(subjectdir,'Trufi','01-Raw',trufifile.name);
        i=i+1;
    end       
    
    transfoparam_file=fullfile(Anat7Tdir,'transfovariables.mat');
    if ~exist(transfoparam_file,'file')
        [transmat,coregmat,deform_field,deform_field_inv,normfile]=calculate_all_00(Lifiles,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles);
        save(char(transfoparam_file),'transmat','coregmat','deform_field','deform_field_inv','normfile');
    else
        load(transfoparam_file,'transmat','coregmat','deform_field','deform_field_inv','normfile');
    end
    
    if ~exist('makeQuantif','var')
        makeQuantif=0;
    end
    
    if makeQuantif
        %convertMasks3TtoMNI(splitfolder,Anat7Tdir,Anat3Tdir,coregmat,deform_field_inv,normfile);
        QuantifMasks=fullfile(splitfolder,'Masks','Quantifmaps'); 
        i=1;
        maskslist=dir(fullfile(QuantifMasks,'*nii'));
        MNImasks=cell(size(maskslist));
        Anat3Tmaskdir=fullfile(Anat3Tdir,'Masks');
        if ~exist(Anat3Tmaskdir,'dir')
            mkdir(Anat3Tmaskdir);
        end
        Anat7Tmaskdir=fullfile(Anat7Tdir,'Masks');
        if ~exist(Anat7Tmaskdir,'dir')
            mkdir(Anat7Tmaskdir);
        end
        for maskfile=maskslist'
            MNImasks{i,1}=fullfile(QuantifMasks,maskfile.name);
            i=i+1;
        end
        
        apply_transform_MNI_to7T(MNImasks,Anat3Tmaskdir,Anat7Tmaskdir,coregmat,deform_field_inv,normfile);
        Quantif_masks=fullfile(subjectdir,"Quantif_masks");
        if ~exist(Quantif_masks,'dir')
            mkdir(Quantif_masks);
        end
        newmaskslist=dir(fullfile(Anat7Tmaskdir,'*nii'));
        for maskfile=newmaskslist'
            maskpath=fullfile(Anat7Tmaskdir,maskfile.name);
            outputpathTPI=fullfile(Quantif_masks,"TPI_"+maskfile.name);
            outputpathtrufi=fullfile(Quantif_masks,"Trufi_"+maskfile.name);
            convertreso_spm(Lifiles{1},maskpath,outputpathTPI);
            convertreso_spm(trufifiles{1},maskpath,outputpathtrufi);
        end
        
    end