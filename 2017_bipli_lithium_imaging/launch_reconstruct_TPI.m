function launch_reconstruct_TPI(Rawdatdir,Processeddir,Codedir,Pythonexe)

%aunch_reconstruct_TPI('C:\Users\js247994\Documents\Bipli2\Test8\Raw','C:\Users\js247994\Documents\Bipli2\Test8\Processed','C:\Users\js247994\Documents\Bipli2\BipliPipeline\scripts\2017_bipli_lithium_imaging\ReconstructionTPI','C:\Python27\python.exe')

if ~exist(Processeddir,'dir')
    mkdir(Processeddir)
end

if ~exist('Pythonexe','var')
    Pythonexe='python';
    if ~exist('Codedir','var')
        Codedir=strcat(pwd,'\ReconstructionTPI');
    end
end

S = dir(Rawdatdir);
S=S(~ismember({S.name},{'.','..'}));
%nbsubjects=sum([S.isdir]);
i=0;
for subj=S'
    if subj.isdir==1
        i=i+1;
        subject=subj.name;
        if ( i > 9 ) 
        subjectnumber =  int2str(i) ;
        else
            subjectnumber =  [ '0', int2str(i) ] ;
        end
        ProcessedTPIpath=strcat(Processeddir,'\',subject,'\TPI\Reconstruct_gridding\01-Raw\');
        if ~exist(ProcessedTPIpath,'dir')
            mkdir(ProcessedTPIpath)
        end

        reconstructfile=strcat(Codedir,'\ProcessData.py');
        twixdir=strcat(Rawdatdir,'\',subject,'\twix7T\');
        T7dir=strcat(Rawdatdir,'\',subject,'\7T\');
        if exist(twixdir,'dir')
            filedir=twixdir;
        elseif exist(T7dir,'dir')
            filedir=T7dir;
        end
        listdat=dir(filedir);

        for j=1:numel(listdat)
            ok=strfind(listdat(j).name,'TPI');
            if ~isempty(ok)
                Tpifilename=listdat(j).name;
                Tpifilepath=strcat(filedir,Tpifilename);
                deg=strfind(Tpifilename,'deg');
                degval=(Tpifilename(deg-2:deg-1));
                Reconstructpath=strcat(ProcessedTPIpath,'\Patient',subjectnumber,'_',degval,'deg');
                Reconstructniipath=strcat(Reconstructpath,'.nii');
                codelaunch=strcat(Pythonexe,{' '},reconstructfile,{' --i '},Tpifilepath,{' --NSTPI --s --FISTA_CSV --o '},Reconstructniipath);
                status = system(codelaunch{1});
            end
        end
    end    
end
  