function run_Compute_Quantif(Processeddir,subject,T1val,Codedir,Pythonexe)
    % run_Compute_Quantif('C:\Users\js247994\Documents\Bipli2\Biplisubjects','C:\Users\js247994\Documents\ReconstructionTPI','C:\Program Files (x86)\Python36-32\python.exe')

    %Launch the reconstruction of the dat files found in raw to the processed
    %images and placed them in right location (should perhaps eventually be
    %updated to also include FISTA?
    %reconstructfile=fullfile(char(Codedir),'ProcessData.py');
    %launch_reconstruct_TPI(Subjectdirtwix,Subjectdirresult,subjectnumber,reconstructfile,Pythonexe );

    Subjectdirp=fullfile(Processeddir,subject);

    filesdirin=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw');
    filesdirout=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','02-PostQuantif');
    filesdone=[];
    filesnii=dir(fullfile(filesdirin,'*.nii'));

    ComputeVFAfile=fullfile(char(Codedir),'reconstructionTPI','ComputeDensity3D_clinic.py');
    ComputeQuantifile=fullfile(char(Codedir),'reconstructionTPI','ComputeQuantif_clinic.py');

    for file1=filesnii'
        degloc=strfind(file1.name,'deg');
        over=0;
        for filesdon=filesdone'
            if strcmp(file1.name,filesdon')
                over=1;
            end
        end
        if ~isempty(degloc)
            file1path=fullfile(filesdirin,file1.name);
            deg1=file1.name(degloc-2:degloc-1);   
            if ~over   
                for file2=filesnii'
                    VFA=~strcmp(file1.name,file2.name) && strcmp(file1.name(degloc:end),file2.name(degloc:end)); %If there is another file with the same name but a different degree value, it is treated as the second VFA file
                    if VFA
                        deg2=file2.name(degloc-2:degloc-1);
                        Computedniipath=fullfile(filesdirout,strcat(file1.name(1:degloc-3),file1.name(degloc+4:end)));
                        file2path=fullfile(filesdirin,file2.name);
                        codelaunchVFA=strcat({'"'},Pythonexe,{'" '},ComputeVFAfile,{' --i1 '},file1path,{' --i2 '},file2path,{' --deg1 '},deg1,{' --deg2 '},deg2,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
                        system(codelaunchVFA{1});
                        filesdone=[filesdone;string(file1.name);string(file2.name)];
                    end
                end
            end
            %[~,filename,ext]=fileparts(file1.name);
            %Computedniipath=strcat(filesdir,filename,ext);
            Computedniipath=fullfile(filesdirout,file1.name);
            codelaunchQuant=strcat({'"'},Pythonexe,{'" '},ComputeQuantifile,{' --i '},file1path,{' --deg '},deg1,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
            system(codelaunchQuant{1});
        end
    end
end
