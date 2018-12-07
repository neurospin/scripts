function newfile_address=apply_deform_field_04(oldfile,newdir,deform_field,normfile,prefix)

    if ~exist('prefix','var')
        prefix="MNI_";
    end
    [normdir,~,~]=fileparts(normfile);
    [olddir,oldname,oldext]=fileparts(oldfile);
    load(normfile,'matlabbatch');
    matlabbatch{1,1}.spm.spatial.normalise.write.subj.def={char(deform_field)};
    matlabbatch{1,1}.spm.spatial.normalise.write.subj.resample={char(oldfile)};
    matlabbatch{1,1}.spm.spatial.normalise.write.woptions.prefix=char(prefix);
    matlabbatch{1,2}.cfg_basicio.file_dir.file_ops.file_move.files={char(fullfile(olddir,strcat(matlabbatch{1,1}.spm.spatial.normalise.write.woptions.prefix,oldname,oldext)))};
    matlabbatch{1,2}.cfg_basicio.file_dir.file_ops.file_move.action.moveto={char(newdir)};
    %tempfile=fullfile(normdir,'temp.mat');
    %save(tempfile,'matlabbatch');
    %spm_jobman('run',tempfile);
    spm_jobman('run',matlabbatch');
    newfile_address=fullfile(char(newdir),prefix+oldname+oldext);
    %delete(tempfile);
end