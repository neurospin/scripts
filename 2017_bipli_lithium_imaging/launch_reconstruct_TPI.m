function launch_reconstruct_TPI(dirdat,ProcessedTPIpath,subjectnumber,reconstructfile,Pythonexe )

%launch_reconstruct_TPI('C:\Users\js247994\Documents\Bipli2\Test8\Raw','C:\Users\js247994\Documents\Bipli2\Test8\Processed','C:\Users\js247994\Documents\Bipli2\BipliPipeline\scripts\2017_bipli_lithium_imaging\ReconstructionTPI','C:\Python27\python.exe')

listdat=dir(dirdat);

for j=1:numel(listdat)
    ok=strfind(listdat(j).name,'TPI');
    if ~isempty(ok)
        Tpifilename=listdat(j).name;
        Tpifilepath=fullfile(dirdat,char(Tpifilename));
        deg=strfind(Tpifilename,'deg');
        degval=(Tpifilename(deg-2:deg-1));
        TPIresultname=strcat('Patient',int2str(subjectnumber),'_',degval,'deg.nii');
        Reconstructpath=fullfile(char(ProcessedTPIpath),char(TPIresultname));
        %Reconstructniipath=strcat(Reconstructpath,'.nii');
        system('python good_luck.py')
        codelaunch=strcat(Pythonexe,{' '},reconstructfile,{' --i '},Tpifilepath,{' --NSTPI --s --FISTA_CSV --o '},Reconstructpath);
        status = system(codelaunch{1});
    end
end
  
  