function run_create_processfolders(Processeddir,subject)

    if ~exist(Processeddir,'dir')
        mkdir(char(Processeddir))
        mkdir(fullfile(char(Processeddir,subject)))
    else
        if ~exist(fullfile(char(Processeddir),char(subject)),'dir')
            mkdir(fullfile(char(Processeddir),char(subject)));
        end
    end

 %   Subjectdirr=fullfile(char(Rawdatdir),char(subject));
   % Subjectdirtwix=fullfile(Subjectdirr,'twix7T');
    Subjectdirp=fullfile(char(Processeddir),char(subject));
 %   Subjectdirresult=fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw');

    %Create the processeddir subject folders if they are not already created
    if ~exist(fullfile(Subjectdirp,'Anatomy3T'),'dir')
        mkdir(fullfile(Subjectdirp,'Anatomy3T'));
    end
    if ~exist(fullfile(Subjectdirp,'Anatomy7T'),'dir')
        mkdir(fullfile(Subjectdirp,'Anatomy7T'));
    end
    if ~exist(fullfile(Subjectdirp,'Anatomy3T'),'dir')
        mkdir(fullfile(Subjectdirp,'Anatomy3T'));
    end
    if ~exist(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw'),'dir');
        mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','01-Raw'));
    end
    if ~exist(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','06-MNIspace'),'dir')
        mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','02-Post-Quantif'));
        mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','03-Filtered'));
        mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','04-7Tanatspace'));
        mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','05-3Tanatspace'));
        mkdir(fullfile(Subjectdirp,'TPI','Reconstruct_gridding','06-MNIspace'));
    end
    if ~exist(fullfile(Subjectdirp,'Trufi'),'dir')
        mkdir(char(fullfile(Subjectdirp,'Trufi')));
    end
end