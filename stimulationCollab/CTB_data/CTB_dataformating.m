% Set working directory and add folders to path
cd \Users\F_r_e\PycharmProjects\TVBsim-py3.8\CTB_data
addpath(genpath("C:\Users\F_r_e\PycharmProjects\TVBsim-py3.8\CTB_data"))

% Load functional data to work on region names
load("Conectividades_2-4Hz_suj2.mat")

% Order regions as in TVB, to match path length matrices order and centres.
indexes = [34:38 64 39:47 49 48 50:63 65:66 1:5 31 6:14 16 15 17:30 32:33];

namesOrdered = atlasbis.table(indexes);

% Load centres.txt file to update with CTB names. 
raw=fileread("tvb-connectivity_66/centres_model.txt");
raw=textscan(raw, "%s %s %s %s %s");
centres=[namesOrdered raw{2} raw{3} raw{4} raw{5}];

% CENTRES
fid = fopen('centres.txt','wt');
for i = 1:size(centres,1)
    fprintf(fid,'%s %s %s %s %s',centres{i,:});
    fprintf(fid,'\n');
end
fclose(fid);

% loop over subjects
for i = 2:10
    % Load mat file containing structural connectivity
    sname="SCreord_suj"+ sprintf('%02d',i)+".mat";
    load(sname)
    
    % Reorder Anatomical Connectivity matrix
    ctbWeights=AC(indexes,indexes);
    
    % Write weights.txt file
    fid = fopen('weights.txt','wt');
    for ii = 1:size(ctbWeights,1)
        fprintf(fid,'%.18e ',ctbWeights(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);

    % Work on functional matrices
    s=dir("ctb-data\Functional connectivity");
    foldernames=struct2cell(s);
    bands=foldernames(1,3:end);
    bands(2,:) = {"2-4","4-8","8-12","12-30","30-45"};
    
    % Create folder to include FC measures x subject
    fcFname="output/FC_subj"+int2str(i)
    mkdir(fcFname)
    % Copy centres.txt and weights.txt to the new folder for reference
    copyfile("centres.txt", fcFname)
    copyfile("weights.txt", fcFname)
    
    % Loop over frequency bands
    for j = 1:length(bands)
        % Load mat file containing structural connectivity
        fname="Conectividades_"+bands{2,j}+"Hz_suj"+int2str(i)+".mat";
        load(fname)
        
        % Reorder Functional matrices
        corramp=corramp(indexes,indexes);
        pli=pli(indexes,indexes);
        plv=plv(indexes,indexes);
        wpli=wpli(indexes,indexes);

        % Write .txt files with connectivity measures matrices
        dirname=fcFname+"/"+bands{1,j}+"corramp.txt";
        fid = fopen(dirname,'wt');
        for ii = 1:size(corramp,1)
            fprintf(fid,'%f ',corramp(ii,:));
            fprintf(fid,'\n');
        end
        fclose(fid);

        dirname=fcFname+"/"+bands{1,j}+"pli.txt";
        fid = fopen(dirname,'wt');
        for ii = 1:size(pli,1)
            fprintf(fid,'%.18e ',pli(ii,:));
            fprintf(fid,'\n');
        end
        fclose(fid);
        
        dirname=fcFname+"/"+bands{1,j}+"plv.txt";
        fid = fopen(dirname,'wt');
        for ii = 1:size(plv,1)
            fprintf(fid,'%.18e ',plv(ii,:));
            fprintf(fid,'\n');
        end
        fclose(fid);
    end
    zipname="output/CTB_connx66_subj"+int2str(i);
    zip(zipname, ["centres.txt", "info.txt", "tract_lengths.txt", "weights.txt"])
end

clear
delete centres.txt weights.txt