function apply_transform_7TtoMNI(files7T,output3T,outputMNI,coregmat,deform_field,normfile)
    for file7T=files7T'
        spm_7T=spm_vol(char(file7T{1}));
        for i=1:size(spm_7T,1)
            [~,filename,ext]=fileparts(file7T{1});
            if size(spm_7T,1)>1
                ext="_"+int2str(i)+string(ext);
            end
            filename=string(filename);
            spm_3T=spm_7T;
            spm_3T.mat=coregmat*spm_7T.mat;
            spm_3T.fname=char(fullfile(output3T,filename+ext));
            if ~exist(output3T,'dir')
                mkdir(output3T);
            end
            if ~exist(outputMNI,'dir')
                mkdir(outputMNI);
            end
            spm_write_vol(spm_3T,spm_read_vols(spm_7T));        
            apply_deform_field_04(spm_3T.fname,outputMNI,deform_field,normfile);
        end
    end    