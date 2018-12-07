function apply_transform_3TtoMNI(files3T,outputMNI,deform_field,normfile,MNIprefix)
    for file3T=files3T'
        spm_3T=spm_vol(char(file3T{1}));
        for i=1:size(spm_7T,1)
            if ~exist(outputMNI,'dir')
                mkdir(outputMNI);
            end   
            apply_deform_field_04(spm_3T.fname,outputMNI,deform_field,normfile,MNIprefix);
        end
    end    