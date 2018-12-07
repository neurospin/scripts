function getanatfiles(rawdir,processeddir) 

    Anat3Tfile=fullfile(processeddir,'Anatomy3T','t1_weighted_sagittal_1_0iso.nii');
    Anat7Tfile=fullfile(processeddir,'Anatomy7T','t1_mpr_tra_iso1_0mm.nii');

    
    if ~exist(Anat3Tfile,'file') || ~exist(Anat7Tfile,'file')
        anat3Tdir=fullfile(processeddir,'Anatomy3T');
        anat7Tdir=fullfile(processeddir,'Anatomy7T');
        anat3Trawdir=fullfile(rawdir,'DICOM3T','T1_WEIGHTED_SAGITTAL_1.0ISO');
        anat7Trawdir=fullfile(rawdir,'DICOM7T','T1_MPR_TRA_ISO1.0MM');
        if ~exist(anat3Tdir,'dir')
            mkdir(anat3Tdir)
        end
        if ~exist(anat7Tdir,'dir')
            mkdir(anat7Tdir)
        end
        dicm2nii(anat3Trawdir,anat3Tdir,'.nii');
        dicm2nii(anat7Trawdir,anat7Tdir,'.nii');
    end