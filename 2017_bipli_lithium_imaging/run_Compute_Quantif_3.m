function run_Compute_Quantif_3(Processeddir,subject,T1val)
    % run_Compute_Quantif('C:\Users\js247994\Documents\Bipli2\Biplisubjects','C:\Users\js247994\Documents\ReconstructionTPI','C:\Program Files (x86)\Python36-32\python.exe')

    %Launch the reconstruction of the dat files found in raw to the processed
    %images and placed them in right location (should perhaps eventually be
    %updated to also include FISTA?
    %reconstructfile=fullfile(char(Codedir),'ProcessData.py');
    %launch_reconstruct_TPI(Subjectdirtwix,Subjectdirresult,subjectnumber,reconstructfile,Pythonexe );

    relax_params.liquid.T1=14000;
    relax_params.liquid.T2=1670;
    relax_params.liquid.T2star=5;
    relax_params.brainmatter.T1=T1val*1000;
    relax_params.brainmatter.T2=63;
    relax_params.brainmatter.T2star=5;
    
    
    Subjectdirp=fullfile(Processeddir,subject);
    
    maskfolder=fullfile(Subjectdirp,'Quantif_masks');
    filesdirTPIin=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','05-MNIspace');
    filesdirTPIout=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','06-postQuantif-MNIspace');
    if ~exist(filesdirTPIout,'dir')
        mkdir(filesdirTPIout);
    end
    
    filesdone=[];    
    if exist(filesdirTPIin,'dir')
        filesTPInii=dir(fullfile(filesdirTPIin,'*_noquantif.nii'));
        for file1=filesTPInii'
            degloc=strfind(file1.name,'deg');
            over=0;
            for filesdon=filesdone'
                if strcmp(file1.name,filesdon')
                    over=1;
                end
            end
            if ~isempty(degloc)
                file1path=fullfile(filesdirTPIin,file1.name);
                deg1=file1.name(degloc-2:degloc-1); 
                %deg1=int2str(90);
                if ~over   
                    for file2=filesTPInii'
                        VFA=~strcmp(file1.name,file2.name) && strcmp(file1.name(degloc:end),file2.name(degloc:end)); %If there is another file with the same name but a different degree value, it is treated as the second VFA file
                        if VFA
                            deg2=file2.name(degloc-2:degloc-1);
                            Computedniipath=fullfile(filesdirTPIout,strcat(file1.name(1:degloc-3),file1.name(degloc+4:end)));
                            file2path=fullfile(filesdirTPIin,file2.name);
                            %codelaunchVFA=strcat({'"'},Pythonexe,{'" '},ComputeVFAfile,{' --i1 '},file1path,{' --i2 '},file2path,{' --deg1 '},deg1,{' --deg2 '},deg2,{' --t1 '}, num2str(T1val), {' --v --o '},Computedniipath);
                            %system(codelaunchVFA{1});
                            filesdone=[filesdone;string(file1.name);string(file2.name)];
                        end
                    end
                end
                B0cor=strfind(file1.name,'B0cor');
                noB0cor=strfind(file1.name,'no_B0cor');
                if ~isempty(B0cor) && isempty(noB0cor)
                    B0corkval=1;
                else
                    B0corkval=0;
                end
                %[~,filename,ext]=fileparts(file1.name);
                filesplit=string(strsplit(file1.name,"."));
                Computedniipath=fullfile(filesdirTPIout,filesplit(1)+"_highresquantif."+filesplit(2));
                liquidmask=dir(fullfile(maskfolder,'*MNI*csfandeyes*'));
                liquidmask=fullfile(maskfolder,liquidmask(1).name);
                mattermask=dir(fullfile(maskfolder,'*MNI*greyandwhite*'));        
                mattermask=fullfile(maskfolder,mattermask(1).name);
                ComputeSignaltoQuantif(file1path,deg1,Computedniipath,'TPI',liquidmask,mattermask,B0corkval,relax_params);                
            end
        end
    else
        display("warning, no TPI folder found at "+filesdirTPIin);
    end
    
    filesdirtrufiin=fullfile(Subjectdirp,'Trufi','05-MNIspace');
    filesdirtrufiout=fullfile(Subjectdirp,'Trufi','06-postQuantif-MNIspace');
    if ~exist(filesdirtrufiout,'dir') && exist(filesdirtrufiin,'dir')
        mkdir(filesdirtrufiout);
    end
    
    if exist(filesdirtrufiin,'dir')
        filestrufinii=dir(fullfile(filesdirtrufiin,'*_noquantif.nii'));
        if ~exist(filesdirtrufiout,'dir')
            mkdir(filesdirtrufiout);
        end
        for file1=filestrufinii'
            
            %B0cor=strfind(file1.name,'B0cor');
            %noB0cor=strfind(file1.name,'no_B0cor');
            %if ~isempty(B0cor) && isempty(noB0cor)
            %    B0corkval=(' --B0cor ');
            %else
            %    B0corkval=('');
            %end
            %[~,filename,ext]=fileparts(file1.name);
            %Computedniipath=strcat(filesdir,filename,ext);
            
            B0corkval='';
            filepath=fullfile(filesdirtrufiin,file1.name);
            %%%%%% IMPORTANT! TO CHANGE SO AS TO INCLUDE REAL FLIP ANGLES
            deg1='30';
            %%%%%%%
            filesplit=string(strsplit(file1.name,"."));
            Computedniipath=fullfile(filesdirtrufiout,filesplit(1)+"_highresquantif."+filesplit(2));
            %Computedniipathtest=fullfile(filesdirtrufiout,filesplit(1)+"_classic."+filesplit(2));
            liquidmask=dir(fullfile(maskfolder,'*MNI*csfandeyes*'));
            liquidmask=fullfile(maskfolder,liquidmask(1).name);
            mattermask=dir(fullfile(maskfolder,'*MNI*greyandwhite*'));        
            mattermask=fullfile(maskfolder,mattermask(1).name);
            ComputeSignaltoQuantif(filepath,deg1,Computedniipath,'trufi',liquidmask,mattermask,B0corkval,relax_params);
        end
    else
        display("warning, no TPI folder found at "+filesdirtrufiin);
    end    
    
end