function convertreso_spm(baseresolfile,otherresofile,outputdir,outputfile)

    
    %load('im_changeres_test.mat')
    if ~exist('outputfile','var')
        [outputdir,outputfile,ext]=fileparts(outputdir);
        outputfile=outputfile+ext;
    end
    matlabbatch{1,1}.spm.util.imcalc.input{1,1}=char(string(baseresolfile)+",1");
    matlabbatch{1,1}.spm.util.imcalc.input{2,1}=char(string(otherresofile)+",1");
    %baseresolfile2="V:\projects\BIPLi7\Clinicaldata\Processed_Data\2018_06_01\TPI\Reconstruct_gridding\03-Filtered\Patient12_21deg_MID717_B0cor_KBgrid_MODULE_Echo0_TE300_rhoSPGR.nii";
    %matlabbatch{1,1}.spm.util.imcalc.input{1,1}=char(string(baseresolfile2)+",1");
    matlabbatch{1,1}.spm.util.imcalc.output=char(outputfile);
    matlabbatch{1,1}.spm.util.imcalc.outdir={char(outputdir)};
    matlabbatch{1,1}.spm.util.imcalc.expression=char('i2');
    matlabbatch{1,1}.spm.util.imcalc.var={};
    matlabbatch{1,1}.spm.util.imcalc.options.dmtx= 0;
    matlabbatch{1,1}.spm.util.imcalc.options.mask= 0;
    matlabbatch{1,1}.spm.util.imcalc.options.interp= -7;
    matlabbatch{1,1}.spm.util.imcalc.options.dtype= 4;
    
    spm_jobman('run',matlabbatch');