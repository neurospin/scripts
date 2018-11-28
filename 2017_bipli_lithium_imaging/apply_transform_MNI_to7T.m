function apply_transform_MNI_to7T(filesMNI,output3T,output7T,coregmat,deform_field,normfile)
    for fileMNI=filesMNI'
        spm_MNI=spm_vol(char(fileMNI{1}));
        for i=1:size(spm_MNI,1)
            [~,filename,ext]=fileparts(fileMNI{1});
            if size(spm_MNI,1)>1
                ext="_"+int2str(i)+string(ext);
            end
            filename=string(filename);
            prefix="3T_";
            if ~exist(output3T,'dir')
                mkdir(output3T);
            end
            filename3T=apply_deform_field_04(spm_MNI.fname,output3T,deform_field,normfile,prefix);
            spm_3T=spm_vol(char(filename3T));
            spm_7T=spm_3T;
            spm_7T.mat=coregmat\spm_3T.mat;
            spm_7T.fname=char(fullfile(output7T,filename+ext));
            if ~exist(output7T,'dir')
                mkdir(output7T);
            end
            spm_write_vol(spm_7T,spm_read_vols(spm_3T));        
            
        end
    end    