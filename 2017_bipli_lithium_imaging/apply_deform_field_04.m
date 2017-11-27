function apply_deform_field_04(oldfile,newdir,deform_field,normfile)

    [normdir,~,~]=fileparts(normfile);
    [olddir,oldname,oldext]=fileparts(oldfile);
    load(normfile);
    matlabbatch{1,1}.spm.spatial.normalise.write.subj.def={deform_field};
    matlabbatch{1,1}.spm.spatial.normalise.write.subj.resample={oldfile};
    matlabbatch{1,2}.cfg_basicio.file_dir.file_ops.file_move.files={strcat(olddir,'\',matlabbatch{1,1}.spm.spatial.normalise.write.woptions.prefix,oldname,oldext)};
    matlabbatch{1,2}.cfg_basicio.file_dir.file_ops.file_move.action.moveto={newdir};
    tempfile=strcat(normdir,'\temp.mat');
    save(tempfile,'matlabbatch');
    spm_jobman('run',tempfile);
    delete(tempfile);
end