function launch_applytransforms(subjectdir,transfoparam_file,reconstruct_type)

    if ~exist(transfoparam_file,'file')
        disp('Error, param file of transformations not found');
    else
        load(transfoparam_file,'transmat','coregmat','deform_field','deform_field_inv');
        deform_field=fullfile(subjectdir,'Anatomy3T',deform_field);
        normfile=fullfile(pwd,'info_pipeline','normwritespm.mat');        
    end
    if ~exist('reconstruct_type','var')
        reconstruct_type='Reconstruct_gridding';
    end
    TPIoutputdirmni=fullfile(subjectdir,'TPI',reconstruct_type,'05-MNIspace');
    TPIoutputdir3T=fullfile(subjectdir,'TPI',reconstruct_type,'04-3Tanatspace');  
    trufioutput3T=fullfile(subjectdir,'Trufi','04-3Tanatspace');
    trufioutputMNI=fullfile(subjectdir,'Trufi','05-MNIspace');
    if ~exist(TPIoutputdirmni,'dir')
        mkdir(TPIoutputdirmni);
    end
    if ~exist(TPIoutputdir3T,'dir')
        mkdir(TPIoutputdir3T);
    end    
    
    i=1;
    TPIfilesS=dir(fullfile(subjectdir,'TPI',reconstruct_type,'03-Filtered','*nii'));
    TPIfiles=cell(size(TPIfilesS,1),1);
    
    for TPIfile=TPIfilesS'
        TPIfiles{i}=fullfile(subjectdir,'TPI',reconstruct_type,'03-Filtered',TPIfile.name);
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
    
    MNIprefix="MNI_";
    runTPIfiles=0;
    i=1;
    for TPIfile=TPIfiles'
        [~,filename,ext]=fileparts(TPIfiles{i});
        MNIoutput=fullfile(TPIoutputdirmni,MNIprefix+filename+ext);
        if ~exist(MNIoutput,'file')
            runTPIfiles=1;
        end
        i=i+1;
    end
    
    runtrufifiles=0;
    i=1;
    for trufifile=trufifiles'
        [~,filename,ext]=fileparts(trufifiles{i});
        MNIoutput=fullfile(trufioutputMNI,MNIprefix+filename+ext);
        if ~exist(MNIoutput,'file')
            runtrufifiles=1;
        end
        i=i+1;
    end
    %deform_field=fullfile(Anat3Tdir,deform_field);
    normfile=fullfile(pwd,'info_pipeline','normwritespm.mat');
    
    if runTPIfiles || forcestart   
        apply_transform_7TtoMNI(TPIfiles,TPIoutputdir3T,TPIoutputdirmni,coregmat,deform_field,normfile,"MNI_");
    end
    Anat3Tdir=fullfile(subjectdir,'Anatomy3T');
    Anat3Tfile=fullfile(Anat3Tdir,'t1_weighted_sagittal_1_0iso.nii');
    if ~exist(fullfile(Anat3Tdir,MNIprefix+"t1_weighted_sagittal_1_0iso.nii"),'file')
        apply_deform_field_04(Anat3Tfile,Anat3Tdir,deform_field,normfile,MNIprefix);
    end
    %if everneeded, add another apply_transform for any other list of 7T
    %files to transfer to MNI space
    if ~isempty(trufifiles) && runtrufifiles
        apply_transform_7TtoMNI(trufifiles,trufioutput3T,trufioutputMNI,coregmat,deform_field,normfile,"MNI_");
    end
