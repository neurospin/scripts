function trufitoprocess(raw_dic,proc_subjdir)

    trufdirs=dir(fullfile(raw_dic,'TRUFI*'));
    if ~isempty(trufdirs)
        for trufdir=trufdirs'
            proc_trufi=fullfile(proc_subjdir,'Trufi','01-Raw');
            if ~exist(proc_trufi,'dir')
                mkdir(proc_trufi)
            end
            dicm2nii(fullfile(raw_dic,trufdir.name),proc_trufi,'.nii',trufdir.name);
        end
        endv