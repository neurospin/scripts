function launch_reconstruct_TPI_all(projectdir)%,subjectdir,subjectnumber,reconstructfile,Pythonexe )

    %launch_reconstruct_TPI('C:\Users\js247994\Documents\Bipli2\Test8\Raw','C:\Users\js247994\Documents\Bipli2\Test8\Processed','C:\Users\js247994\Documents\Bipli2\BipliPipeline\scripts\2017_bipli_lithium_imaging\ReconstructionTPI','C:\Python27\python.exe')
    projectdir='/neurospin/ciclops/projects/BIPLi7/ClinicalData';
    pythonexe='python';
    raw_dir=fullfile(projectdir,'Raw_Data','2018*');
    currentfolder=pwd;
    reconstructfile=fullfile(pwd,'reconstructionTPI','ProcessData.py');
    reconstructfile=fullfile('/home/js247994/DocumentsN2/2017_bipli_lithium_imaging','reconstructionTPI','ProcessData.py');
    listsubj=dir(raw_dir);

    for i = 1:numel(listsubj)
        subjname=listsubj(i).name;
        raw_subjdir=fullfile(projectdir,'Raw_Data',subjname,'twix7T');
        proc_subjdir=fullfile(projectdir,'Processed_Data',subjname);
        fieldmap_file=fullfile(proc_subjdir,'Field_mapping','fieldmap_final.nii');
        if ~exist(fieldmap_file,'file')
            spm_alignmentfile=fullfile(currentfolder,'info_pipeline','fieldmapwritespm.mat');
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
                if i<10
                    subjectnumber=strcat('0',int2str(i));
                elseif i>=10
                    subjectnumber=int2str(i);
                end
                TPIresultname=strcat('Patient',(subjectnumber),'_',degval,'deg.nii');
                Reconstructpath=fullfile(char(ProcessedTPIpath),char(TPIresultname));
                codelaunch=strcat(pythonexe,{' '},reconstructfile,{' --i '},Tpifilepath,{' --fieldmap '}, fieldmap_file, {' --NSTPI --s --FISTA_CSV --o '},Reconstructpath);
                disp('hi');
            end
        end
    end
end
