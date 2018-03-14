function launch_calculate_all(subjectdir)    

    %Lifile='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\03-Concentration\Patient01_KBgrid_MODULE_Echo0_filtered.nii';
    %Anat7Tfile='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy7T\t1_mpr_tra_iso1_0mm.nii';
    %Anat3Tfile='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy3T\t1_weighted_sagittal_1_0iso.nii';
    %Litranslation='C:\Users\js247994\Documents\Bipli2\Test2\Anatomy7T\1H_TO_Li_f.trm';
    %Lioutputdircoreg='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\06_Final';
    %Lioutputdirmni='C:\Users\js247994\Documents\Bipli2\Test2\TPI\Reconstruct_gridding\06_Final';
    %allsubjs=dir('V:\people\Jacques\BIPLi7\February_18\2*')
    
    keepniifiles=3;
    Currentfolder=pwd;
    splitfolder=strsplit(Currentfolder,'\');
    splitfolder=splitfolder(1:end);
    %TPMfile='C:\Users\js247994\Documents\FidDependencies\spm12\tpm\TPM.nii'; 
    TPMfile=strcat(strjoin(splitfolder,'\'),'\Masks\TPM.nii'); 
    segmentfile=strcat(Currentfolder,'\info_pipeline\segmentsubjectspm.mat'); %Later do a thing that finds it automatically;
    Lioutputdirmni=strcat(subjectdir,'\TPI\Reconstruct_gridding\06-MNIspace');
    Lioutputdir7T=strcat(subjectdir,'\TPI\Reconstruct_gridding\04-7Tanatspace');
    Lioutputdir3T=strcat(subjectdir,'\TPI\Reconstruct_gridding\05-3Tanatspace');    
    %Litranslation=strcat(subjectdir,'\Anatomy7T\1H_TO_Li_f.trm');
    Anat3Tfile=strcat(subjectdir,'\Anatomy3T\t1_weighted_sagittal_1_0iso.nii');
    Anat7Tfile=strcat(subjectdir,'\Anatomy7T\t1_mpr_tra_iso1_0mm.nii');
    LifilesS=dir(strcat(subjectdir,'\TPI\Reconstruct_gridding\03-Filtered\*.nii'));
    i=1;
    Lifiles=cell(size(LifilesS,1),1);
    for Lifile=LifilesS'
        Lifiles{i}=strcat(subjectdir,'\TPI\Reconstruct_gridding\03-Filtered\',Lifile.name);
        i=i+1;
    end
    
    calculate_all_00(Lifiles,Lioutputdir7T,Lioutputdir3T,Lioutputdirmni,Anat7Tfile,Anat3Tfile,TPMfile,segmentfile,keepniifiles);