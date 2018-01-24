function deformfield=calculate_deform_field_03(anat3Tfile,segmentfile,TPMfile,keepniifiles)%Greyfile,Whitefile,CSFfile)

    %U=load('C:\Users\js247994\Documents\Bipli2\Processing\Batch\Segment2spm_OneSubject.mat');
    %keepniifiles=0=> then erase everything
    %keepniifiles=1=> then erase everything but final mask
    %keepniifiles=2=> then erase the modulated warped results
    %keepniifiles=3=> keep everything
    
    %TPMfile='C:\Users\js247994\Documents\FidDependencies\spm12\tpm\TPM.nii';
    %segmentfile='C:\Users\js247994\Documents\Bipli2\Test\segmentsubjectspm12_1.mat';
    
    [dirsegment,~,~]=fileparts(segmentfile);
    load(segmentfile);
    if size(TPMfile,1)==3
        matlabbatch{1,1}.spm.spatial.preproc.opts.tpm{1}=TPMfile(1);%Greyfile;
        matlabbatch{1,1}.spm.spatial.preproc.opts.tpm{2}=TPMfile(2);%Whitefile;
        matlabbatch{1,1}.spm.spatial.preproc.opts.tpm{3}=TPMfile(3);%CSFfile;
        matlabbatch{1,1}.spm.spatial.preproc.opts.tpm=U.matlabbatch{1,1}.spm.spatial.preproc.opts.tpm{1:3};
        matlabbatch{2}.spm.util.imcalc.expression = '(i1+i2+i3)';
        matlabbatch(2).spm.util.imcalc.input=U.matlabbatch(2).spm.util.imcalc.input(1:3);
    elseif size(TPMfile,1)==1
        for i=1:6
            matlabbatch{1,1}.spm.spatial.preproc.tissue(i).tpm={strcat(TPMfile,',',num2str(i))};
        end
    else
        error('Cannot read TPMfile')
    end
    matlabbatch{1}.spm.spatial.preproc.channel.vols =  { [anat3Tfile, ',1'] };
    [subjectdir,filename,ext]=fileparts(anat3Tfile);
    matlabbatch{2}.spm.util.imcalc.output = strcat(subjectdir,'\sumsegments.nii');    
    matlabbatch{3}.spm.util.imcalc.output = strcat(subjectdir,'\mask3T.nii');    
    
    if keepniifiles>0
        matlabbatch=matlabbatch(1:5);
        if keepniifiles>1
            matlabbatch=matlabbatch(1:4);
            if keepniifiles>2
                matlabbatch=matlabbatch(1:3);
            end
        end
    end  
    
    tempsegfile=strcat(dirsegment,'\tempmatseg.mat');
    save(tempsegfile,'matlabbatch');
    spm_jobman('run',tempsegfile);
    delete(tempsegfile);
    deformfield=strcat(subjectdir,'\y_',filename,ext);
    
end