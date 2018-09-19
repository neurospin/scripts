 function launch_reconstruct_TPI(projectdir,subjname,codedir,T1val,pythonexe,subjectnumber)
   
    raw_subjdir=fullfile(projectdir,'Raw_Data',subjname,'twix7T');
    proc_subjdir=fullfile(projectdir,'Processed_Data',subjname);
    reconstructfile=fullfile(codedir,'reconstructionTPI','ProcessData.py');
    
    if ~exist('subjectnumber','var')
        S = dir(raw_subjdir);
        S=S(~ismember({S.name},{'.','..'}));
        %nbsubjects=sum([S.isdir]);
        i=0;
        for subj=S'
            if subj.isdir==1
                i=i+1;
                if subject==subjname
                    if ( i > 9 ) 
                        subjectnumber =  int2str(i) ;
                    else
                        subjectnumber =  [ '0', int2str(i) ] ;
                    end
                end
            end
        end
    end  
    
    fieldmap_file=fullfile(proc_subjdir,'Field_mapping','fieldmap_final.nii');
    if ~exist(fieldmap_file,'file')
        spm_alignmentfile=fullfile(codedir,'info_pipeline','fieldmapwritespm.mat');
        ref_im=dir(fullfile(projectdir,'*.nii'));
        ref_im=fullfile(projectdir,string(ref_im.name));
        prepare_fieldmap(projectdir,subjname,spm_alignmentfile,ref_im);
    end

    listdat=dir(raw_subjdir);
    for j=1:numel(listdat)
        ok=strfind(listdat(j).name,'TPI');
        if ~isempty(ok)  
            ProcessedTPIpath=fullfile(proc_subjdir,'TPI','Reconstruct_gridding','01-Raw');
            if ~exist(ProcessedTPIpath,'dir')
                mkdir(ProcessedTPIpath)
            end
            Tpifilename=listdat(j).name;
            Tpifilepath=fullfile(raw_subjdir,char(Tpifilename));
            deg=strfind(Tpifilename,'deg');
            degval=(Tpifilename(deg-2:deg-1));

            TPIresultname=strcat('Patient',(subjectnumber),'_',degval,'deg.nii');
            Reconstructpath=fullfile(char(ProcessedTPIpath),char(TPIresultname));
            codelaunch=strcat(pythonexe,{' '},reconstructfile,{' --i '},Tpifilepath,{' --fieldmap '}, fieldmap_file, {' --NSTPI --s --FISTA_CSV --o '},Reconstructpath);
            %system(codelaunch)
        end
    end
    run_create_processfolders(fullfile(projectdir,'Processed_Data'),subjname)
    run_Compute_Quantif(fullfile(projectdir,'Processed_Data'),subjname,T1val,codedir,pythonexe)
    runBTK(proc_subjdir)
    launch_calculate_all(proc_subjdir)    