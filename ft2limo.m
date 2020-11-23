%% Call LIMO functions from FieldTrip files

%% (DATA SPECIFIC) Set the paths and names
PATH_TO_RAW_EEG             = 'D:\__EEG-data';
PATH_TO_PROCESSED_EEG       = 'D:\__EEG-data\EEG_Erika_format\EEG';
PATH_TO_ELEC                = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Data';
PATH_TO_FIELDTRIP           = 'D:\FieldTrip';
PATH_TO_LIMO                = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\3-Dynamic-Analysis\limo_tools';
PATH_TO_ROOT                = 'D:\__EEG-data\BIDS_files';

% specific variables
raw_eeg_common_name         = 'ARC_J*';
processed_eeg_common_name   = 'J_*';
elec_mat                    = 'elec_tmp.mat';
processed_eeg_mat           = 'clean_eeg.mat';
task_name                   = 'semantic-priming';
nb_elec                     = 64;

% useful subfolders
PATH_TO_DERIV               = fullfile(PATH_TO_ROOT, 'derivatives');
PATH_TO_TEMPLATE_ELEC       = fullfile(PATH_TO_FIELDTRIP,'template\electrode', sprintf('GSN-HydroCel-%d.sfp',nb_elec));
if ~exist(PATH_TO_TEMPLATE_ELEC,'file')
    PATH_TO_TEMPLATE_ELEC = fullfile(PATH_TO_FIELDTRIP,'template\electrode', sprintf('GSN-HydroCel-%d_1.0.sfp',nb_elec));
end
PATH_TO_TEMPLATE_NEIGHBOURS = fullfile(PATH_TO_FIELDTRIP,'template\neighbours',sprintf('biosemi%d_neighb.mat',nb_elec));
if ~exist(PATH_TO_TEMPLATE_NEIGHBOURS,'file')
    sprintf('ERROR in ft2limo: no neighbours template for %d electrodes...\nPlease find another template or create your own neighbouring matrix in model.defaults.neighbouring_matrix',nb_elec)
end

% add toolboxes to path
addpath(PATH_TO_FIELDTRIP)
addpath(PATH_TO_LIMO)
addpath(genpath(fullfile(PATH_TO_LIMO,'external')))
addpath(genpath(fullfile(PATH_TO_LIMO,'limo_cluster_functions')))

%% (DATA SPECIFIC) Create derivatives files and set model.cat_files
dinfo = dir(fullfile(PATH_TO_PROCESSED_EEG,processed_eeg_common_name));
subj = {dinfo.name};

model.cat_files = {};
subj_ID = 1;
for subj_name = drange(subj)
    processed_eeg = load(fullfile(PATH_TO_PROCESSED_EEG,subj_name{1},processed_eeg_mat));
    processed_eeg = struct2cell(processed_eeg);
    processed_eeg = processed_eeg{1};
%     processed_eeg = processed_eeg.(processed_eeg_mat(1:strfind(processed_eeg_mat,'.mat')-1)); %other way to extract the struct
    
    cfg = [];
    deriv_mat = ft_appenddata(cfg,processed_eeg{:});
    
    model.cat_files{subj_ID,1} = [];
    for i = 1:length(processed_eeg)
        N = length(processed_eeg{i}.trial);
        model.cat_files{subj_ID,1} = [model.cat_files{subj_ID,1}; i*ones(N,1)];
    end
    
    if subj_ID >= 10
        subfolder = 'sub-0%d';
    else
        subfolder = 'sub-00%d';
    end
    derivatives_path = fullfile(PATH_TO_DERIV,sprintf(subfolder,subj_ID),'eeg', [sprintf(subfolder,subj_ID) '_task-' task_name '_eeg.mat']);
    [root,name,ext] = fileparts(derivatives_path);
    if ~exist(root,'dir')
        mkdir(root)
    end
    % add electrode positions
    elec = tdfread(fullfile(PATH_TO_ROOT,sprintf(subfolder,subj_ID),'eeg', [sprintf(subfolder,subj_ID) '_task-' task_name '_channels.tsv']));
% channels is on the hardware part, not the scalp => electrode positions
% have to be defined elsewhere
    elec.name = cellstr(elec.name);
    deriv_mat.chanpos = zeros(length(deriv_mat.label),3);
    for i = 1:length(deriv_mat.label)
        lab = deriv_mat.label{i};
        deriv_mat.chanpos(i,:) = elec.position(strcmpi(elec.name,lab),:);
%         deriv_mat.elec.elecpos = ...
    end
    
    save(derivatives_path,'deriv_mat');
    
    subj_ID = subj_ID + 1;
end

%% (REQUIRED EEG_JSON FILE FOR SUB-001 TO BE MANUALLY CREATED) Create BIDS files
json_path = fullfile(PATH_TO_ROOT,sprintf('sub-00%d',1),'eeg',[sprintf('sub-00%d',1) '_task-' task_name '_eeg.json']);
channel_path = fullfile(PATH_TO_ROOT,sprintf('sub-00%d',1),'eeg', [sprintf('sub-00%d',1) '_task-' task_name '_channels.tsv']);
if ~exist(json_path,'file')
    disp('ERROR ! NEED EEG_JSON.TSV FILE FOR SUB-001 TO BE MANUALLY CREATED')
    return;
end

dinfo = dir(fullfile(PATH_TO_RAW_EEG,raw_eeg_common_name));
fileNames = { dinfo.name };
for iFile = 1 : numel( dinfo )
    if iFile >= 10
        subfolder = 'sub-0%d';
    else
        subfolder = 'sub-00%d';
    end
    output_path = fullfile(PATH_TO_ROOT,sprintf(subfolder,iFile),'eeg', [sprintf(subfolder,iFile) '_task-' task_name '_eeg.bdf']);
    [root,name,ext] = fileparts(output_path);
    if ~exist(root,'dir')
        mkdir(root)
    end
    copyfile(fullfile(PATH_TO_RAW_EEG,dinfo(iFile).name),output_path);

    % create json & channels file
    if iFile ~= 1
        copyfile(json_path,[output_path(1:end-3) 'json']);
%         copyfile(channel_path,[output_path(1:end-7) 'channels.tsv']);
    end
    
    % create channels file
    elec = load(fullfile(PATH_TO_ELEC,sprintf(subfolder, iFile),elec_mat),'-mat');
    elec = struct2cell(elec);
    elec = elec{1};
    
    name = elec.label;
    type = repmat('EEG',length(elec.label),1);
    units = repmat('uV',length(elec.label),1);
    position = num2str(elec.chanpos);
    T = table(name,type,units,position);
    channel_path = [output_path(1:end-7) 'channels.txt'];
    writetable(T,channel_path,'Delimiter','\t');
    movefile(channel_path, [channel_path(1:end-3) 'tsv'])
    
    % create events file
    event = ft_read_event(fullfile(PATH_TO_RAW_EEG,dinfo(iFile).name));
    type = {event.type}';
    sample = {event.sample}';
    value = {event.value}';
    offset = {event.offset}';
    duration = {event.duration}';
    T = table(type,sample,value,offset,duration);
    event_path = [output_path(1:end-7) 'events.txt'];
    writetable(T,event_path,'Delimiter','\t');
    movefile(event_path, [event_path(1:end-3) 'tsv'])
end

%% (DATA SPECIFIC) Call limo_batch function
option = 'both';

% model is a structure that specifiy information to build a model
dinfo = dir(fullfile(PATH_TO_DERIV,'sub-*'));
model.set_files = {};
model.cat_files = {};
model.cont_files = {};
for i = 1 : numel( dinfo )
    if i > 10
        break;
    end
    if i >= 10
        subfolder = 'sub-0%d';
    else
        subfolder = 'sub-00%d';
    end
    model.set_files{i,1} = fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_eeg.mat']);
    eeg = load(model.set_files{i,1});
    eeg = struct2cell(eeg);
    eeg = eeg{1};
    model.cat_files{i,1} = eeg.trialinfo.condition;
    model.cat_files{i,1}(isnan(model.cat_files{i,1})) = 0;
    fields = eeg.trialinfo.Properties.VariableNames;
    fields = fields(5:end);
    cont = [];
    for f = fields
        tmp = eeg.trialinfo.(f{1});
        cont = [cont,limo_split_continuous(model.cat_files{i,1},tmp)];
    end
    model.cont_files{i,1} = cont;
    model.cont_files{i,1}(isnan(model.cont_files{i,1})) = 0;
%     model.cat_files{i,1}(isnan(model.cat_files{i,1})) = 0;
%     model.set_files{1,1} = fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_eeg.mat']);
%     eeg = load(model.set_files{1,1});
%     eeg = struct2cell(eeg);
%     eeg = eeg{1};
%     model.cat_files{1,1} = eeg.trialinfo.condition;
end
% model.defaults: specifiy the parameters to use for each subject
model.defaults.type = 'Channels'; %or 'Components'
model.defaults.analysis = 'Time'; %'Frequency' or 'Time-Frequency'
model.defaults.method = 'OLS'; %'IRLS' 'WLS'
model.defaults.type_of_analysis = 'Mass-univariate'; %or 'Multivariate'
model.defaults.fullfactorial = 0; %or 1
model.defaults.zscore = 0; %or 1
model.defaults.start = -200; %starting time in ms
model.defaults.end = 500; %ending time in ms
model.defaults.bootstrap = 0; %or 1
model.defaults.tfce = 0; %or 1

elec = load(fullfile(PATH_TO_ELEC,sprintf(subfolder, i),elec_mat),'-mat');
elec = struct2cell(elec);
elec = elec{1};
elec_neighb = elec;
elec_neighb.pnt = elec_neighb.chanpos;
data_neighb = elec;
data_neighb.elec = elec_neighb;
cfg = [];
cfg.elec = elec_neighb;
cfg.neighbourdist = 40; %defined in cm in limo_ft_neighbourselection
[neighbours,channeighbstructmat] = limo_ft_neighbourselection(cfg,data_neighb);
model.defaults.neighbouring_matrix = channeighbstructmat;

% model.defaults.neighbouring_matrix = template2neighbmat(PATH_TO_TEMPLATE_NEIGHBOURS,nb_elec); %neighbouring matrix use for clustering (necessary if bootstrap = 1)
%neighbouring matrix format: [n_chan x n_chan] of 1/0 (neighbour or not)
% model.defaults.template_elec = ft_read_sens(PATH_TO_TEMPLATE_ELEC);
model.defaults.template_elec = elec;

% contrast.mat = [1 0 0 0 -1 0 0 0 0;
%                 0 1 0 0 0 -1 0 0 0;
%                 0 0 1 0 -1 0 0 0 0;
%                 0 0 0 1 0 -1 0 0 0];

%t-test
% contrast.mat = [1 0 1 0 1 0 1 0 0;
%                 0 1 0 1 0 1 0 1 0];

%1 sample t-test nat vs man and ANOVA semantic vs non-semantic and nat vs man
% contrast.mat = [0 1 -1 1 -1 1 -1 0 0 0];
contrast.mat = [0 1 -1 1 -1  1 -1 0 0 0;
                0 1  1 1  1 -1 -1 0 0 0];
            
% save(fullfile(PATH_TO_DERIV,'model.mat'),'model')
% save(fullfile(PATH_TO_DERIV,'contrast.mat'),'contrast')

% %uncomment if you want to load an existing model/contrast
% model = load(fullfile(PATH_TO_DERIV,'model.mat'));
% model = model.model;
% contrast = load(fullfile(PATH_TO_DERIV,'contrast.mat'));
% contrast = contrast.contrast;

cd(PATH_TO_ROOT)
[LIMO_files, procstatus] = limo_batch(option,model,contrast);
% option = 'model specification';
% [LIMO_files, procstatus] = limo_batch(option,model);

%% call limo_random_select
clc;
LIMOfiles = fullfile(pwd,'Beta_files_GLM_OLS_Time_Channels.txt');

%expected_chanlocs = mean chanlocs
expected_chanlocs = limo_avg_expected_chanlocs(PATH_TO_DERIV, model.defaults);

% LIMOPath = limo_random_select('Repeated Measures ANOVA',expected_chanlocs,'LIMOfiles',... 
%     LIMOfiles,'analysis_type','Full scalp analysis','parameters',{[1 2],[3 4]},...
%     'factor names',{'semantic_relation', 'type_of_object'},'type','Channels','nboot',1000,'tfce',1,'skip design check','yes');


LIMOPath = limo_random_select('Repeated Measures ANOVA',expected_chanlocs,'LIMOfiles',... 
    LIMOfiles,'analysis_type','Full scalp analysis','parameters',{[1 2],[5 6]},...
    'factor names',{'semantic_relation', 'type_of_object'},'type','Channels','nboot',1000,'tfce',1,'skip design check','yes');

if ~exist('one_sample_t_test','dir')
    mkdir('one_sample_t_test')
end
LIMOfiles = fullfile(pwd,'con1_files_GLM_OLS_Time_Channels.txt');
cd('one_sample_t_test')

LIMOPath = limo_random_select('one sample t-test',expected_chanlocs,'LIMOfiles',... 
    LIMOfiles,'analysis_type','Full scalp analysis','parameters',{1},...
    'type','Channels','nboot',1000,'tfce',1,'skip design check','yes');



%% Source analysis

%% Compute source for each condition
subfolder = 'sub-0%d';
i = 30; %subject ID
eeg = load(fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_eeg.mat']));
eeg = struct2cell(eeg);
eeg = eeg{1};

timelock = cell(1,max(eeg.trialinfo.condition));
for i = 1:max(eeg.trialinfo.condition)
    cfg = [];
    cfg.trials = find(eeg.trialinfo.condition==i)';
    cfg.covariance = 'yes';
    timelock{i} = ft_timelockanalysis(cfg, eeg);
end

cfg = [];
timelock = ft_appenddata(cfg,timelock{:});

cfg                     = [];
cfg.method              = 'mne';                    %specify minimum norm estimate as method
% cfg.latency             = 0.025;                    %latency of interest
cfg.sourcemodel         = sourcemodel_and_leadfield;%the precomputed leadfield
cfg.headmodel           = vol;                      %the head model
cfg.elec                = elec_aligned;                 %the electrodes
cfg.channel             = elec_aligned.label(1:64);           %the useful channels
cfg.mne.prewhiten       = 'yes';                    %prewhiten data
cfg.mne.lambda          = 0.01;                     %regularisation parameter
cfg.mne.scalesourcecov  = 'yes';                    %scaling the source covariance matrix
cfg.rawtrial            = 'yes';
cfg.keeptrials          = 'yes';
source_dipole = ft_sourceanalysis(cfg, timelock);

cfg=[];
test = ft_appenddata(cfg,source_dipole)
%% Compute ROI-by-ROI source activity and neighbouring matrix

% subfolder = 'sub-0%d';
% i = 30; %subject ID
% source_dipole = load(fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_source.mat']));
% source_dipole = struct2cell(source_dipole);
% source_dipole = source_dipole{1};

% select target ROIs
roi_mat = {[3,5];[4,6];[7,9];[8,10];[11,13,15];[12,14,16];61;62;63;64;65;66;67;68;31;32;[83,87];[84,88];85;86};
[roi_atlas] = select_roi(sourcemodel_atlas,roi_mat);

source_roi = dipole2roi(source_dipole,roi_atlas);
source_roi.trial = permute(source_roi.mom, [[1,2],3])
source_roi.trial = NaN(size(source_roi.mom,1)*size(source_roi.mom,2),size(source_roi.mom,3));
for i = 1:size(source_roi.mom,2)
    source_roi.trial = 

neighbouring_matrix = source_neighbmat(roi_atlas);

%% (DATA SPECIFIC) neighbourhood correction following regions properties
for i = 1:length(neighbouring_matrix)
    % forbid left/right neighbours
    if mod(i,2)
        neighbouring_matrix(i,2:2:end)=0;
    else
        neighbouring_matrix(i,1:2:end)=0;
    end
    % ensure left/right symmetry
    for j = 1:i
        if neighbouring_matrix(i,j)==1
            if mod(i,2)
                neighbouring_matrix(i+1,j+1)=1;
                neighbouring_matrix(j+1,i+1)=1;
            else
                neighbouring_matrix(i-1,j-1)=1;
                neighbouring_matrix(j-1,i-1)=1;
            end
        end
    end
end
figure()
imagesc(neighbouring_matrix)

%% Run limo_batch on sources


