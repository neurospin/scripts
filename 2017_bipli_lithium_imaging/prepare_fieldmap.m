function prepare_fieldmap(projectdir,subjname,spm_alignmentfile,ref_im)

  %  spm_alignmentfile='/volatile/scripts/2017_bipli_lithium_imaging/info_pipeline/fieldmapwritespm.mat';
%    ref_im='/neurospin/ciclops/projects/BIPLi7/Masks/Li_ref.nii';
%    ref_im='/neurospin/ciclops/projects/SIMBA/Clinicaldata/Processed_Data/2018_08_01_test/TPI/Reconstruct_gridding/01-Raw/meas162_KBgrid_MODULE_Echo0_TE500.nii';
    
    
    proc_subjdir=fullfile(projectdir,'Processed_Data',subjname);
    raw_subjdir=fullfile(projectdir,'Raw_Data',subjname);

    fsladress='/volatile/fsl/';    
    setenv('FSLDIR',fsladress);  % this to tell where FSL folder is
    setenv('FSLOUTPUTTYPE', 'NIFTI_GZ'); % this to tell what the output type would be
    
    fieldmap_folder= string(fullfile(proc_subjdir,'Field_mapping'));
    fieldmap_magdirproc=fullfile(proc_subjdir,'Field_mapping','Field_mapping_mag');
    fieldmap_phasedirproc=fullfile(proc_subjdir,'Field_mapping','Field_mapping_phase');
    magnit_file=dir(fullfile(fieldmap_magdirproc,'*nii'));
    magnit_file=fullfile(fieldmap_magdirproc,magnit_file.name);
    %magnit_file=fullfile(fieldmap_magdirproc,'field_mapping.nii');
    phase_file=dir(fullfile(fieldmap_phasedirproc,'*nii'));
    phase_file=fullfile(fieldmap_phasedirproc,phase_file.name);
    %phase_file=fullfile(fieldmap_phasedirproc,'field_mapping_phase.nii');
    if ~exist(magnit_file,'file') || exist(magnit_file,'file')==7

        mkdir(fieldmap_folder)
        mkdir(fieldmap_magdirproc)
        mkdir(fieldmap_phasedirproc)
        fieldmap_magdirraw=fullfile(raw_subjdir,'DICOM7T','FIELD_MAPPING_1');
        dicm2nii(fieldmap_magdirraw,fieldmap_magdirproc,'.nii');
        fieldmap_phasedirraw=fullfile(raw_subjdir,'DICOM7T','FIELD_MAPPING_2');
        dicm2nii(fieldmap_phasedirraw,fieldmap_phasedirproc,'.nii');
        magnit_file=dir(fullfile(fieldmap_magdirproc,'*nii'));
        magnit_file=fullfile(fieldmap_magdirproc,magnit_file.name);
        phase_file=dir(fullfile(fieldmap_phasedirproc,'*nii'));
        phase_file=fullfile(fieldmap_phasedirproc,phase_file.name);
    end

        
    bet_command= fullfile( fsladress, '/bin/fsl5.0-bet');
    fsl_prepare_command = fullfile( fsladress, '/bin/fsl5.0-fsl_prepare_fieldmap');
    
    magnit_betfile=fullfile(fieldmap_folder,'field_mapping_bet.nii');
    if ~exist(magnit_betfile,'file')
        cmd=char(strcat(bet_command, {' '}, magnit_file, {' '}, magnit_betfile, {' -f 0.5'}));
        system(cmd);
        cmd=char(strcat('gunzip', {' '}, strcat(magnit_betfile,'.gz')));
        system(cmd);    
    end
    
    fieldmap_prepared=fullfile(fieldmap_folder,'field_mapping_rad.nii');
    if ~exist(fieldmap_prepared,'file')
        cmd=strcat(fsl_prepare_command, {' SIEMENS '}, phase_file, {' '},magnit_betfile,{' '}, fieldmap_prepared, {' 1.02'});
        system(char(cmd));
    
        cmd=char(strcat('gunzip', {' '}, strcat(fieldmap_prepared,'.gz')));
        system(cmd);    
    end
    load(spm_alignmentfile)
    matlabbatch{1,1}.spm.util.imcalc.input{1}=char(ref_im + ',1');
    matlabbatch{1,1}.spm.util.imcalc.input{2}=char(fieldmap_prepared + ',1');
    matlabbatch{1,1}.spm.util.imcalc.outdir={char(fieldmap_folder)};
    matlabbatch{1,2}.spm.util.imcalc.outdir={char(fieldmap_folder)};
    %tempfile=fullfile('temp.mat');
    %save(tempfile,'matlabbatch');
    spm_jobman('run',matlabbatch);
    %delete(tempfile);
end