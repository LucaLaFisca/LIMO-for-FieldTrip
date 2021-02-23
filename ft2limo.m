%% Call LIMO functions from FieldTrip files

%% (DATA SPECIFIC) Set the paths and names
PATH_TO_RAW_EEG             = 'D:\__EEG-data';
PATH_TO_PROCESSED_EEG       = 'D:\__EEG-data\EEG_Erika_format\EEG';
PATH_TO_ELEC                = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Data';
PATH_TO_FIELDTRIP           = 'D:\FieldTrip';
PATH_TO_LIMO                = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\3-Dynamic-Analysis\limo_tools';
PATH_TO_FT2LIMO             = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\3-Dynamic-Analysis\LIMO-for-FieldTrip';
PATH_TO_CUSTOM_FUNCTIONS    = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\0-General-Pipeline\EEG-Source-Analysis-Pipeline';
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
addpath(PATH_TO_FT2LIMO)
addpath(PATH_TO_CUSTOM_FUNCTIONS)


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
for i = numel( dinfo ):-1:1
    if i >= 10
        subfolder = 'sub-0%d';
    else
        subfolder = 'sub-00%d';
    end
    model.set_files{i,1} = fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_eeg.mat']);
    eeg = load(model.set_files{i,1});
    eeg = eeg.(cell2mat(fieldnames(eeg)));
    regressors = load(fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg','combined_trialinfo.mat'));
    regressors = regressors.(cell2mat(fieldnames(regressors)));
    model.cat_files{i,1} = eeg.trialinfo.condition;
%     model.cat_files{i,1}(model.cat_files{i,1}==0) = NaN;
    model.cat_files{i,1}(isnan(model.cat_files{i,1})) = 0;
    fields = regressors.Properties.VariableNames;
    fields = fields(4:end);
    % note: we combined complexity, imageability and concreteness (pca)
    cont = [];
    for f = fields
        tmp = regressors.(f{1});
        cont = [cont,limo_split_continuous(model.cat_files{i,1},tmp)];
    end
    for j = 1:size(cont,2)
        idx = find(isnan(cont(:,j)));
        for row = idx'
            if ~all(isnan(cont(row,:)))
                cont(row,j) = 0;
            end
        end
    end
    cont(isnan(cont)) = 0;
    model.cont_files{i,1} = cont;
end
% model.defaults: specifiy the parameters to use for each subject
model.defaults.type = 'Channels'; %or 'Components'
model.defaults.analysis = 'Time'; %'Frequency' or 'Time-Frequency'
model.defaults.method = 'OLS'; %'IRLS' 'WLS'
model.defaults.type_of_analysis = 'Multivariate'; %or 'Mass-univariate'
model.defaults.fullfactorial = 0; %or 1
model.defaults.zscore = 0; %or 1
model.defaults.start = -200; %starting time in ms
model.defaults.end = 500; %ending time in ms
model.defaults.bootstrap = 0; %or 1
model.defaults.tfce = 0; %or 1

% elec = load(fullfile(PATH_TO_ELEC,sprintf(subfolder, i),elec_mat),'-mat');
% elec = struct2cell(elec);
% elec = elec{1};
elec_neighb = eeg.elec;
elec_neighb.pnt = elec_neighb.chanpos;
data_neighb = eeg;
data_neighb.elec = elec_neighb;
cfg = [];
cfg.elec = elec_neighb;
cfg.neighbourdist = 40; %defined in cm in limo_ft_neighbourselection
[neighbours,channeighbstructmat] = limo_ft_neighbourselection(cfg,data_neighb);
model.defaults.neighbouring_matrix = channeighbstructmat;

% model.defaults.neighbouring_matrix = template2neighbmat(PATH_TO_TEMPLATE_NEIGHBOURS,nb_elec); %neighbouring matrix use for clustering (necessary if bootstrap = 1)
%neighbouring matrix format: [n_chan x n_chan] of 1/0 (neighbour or not)
% model.defaults.template_elec = ft_read_sens(PATH_TO_TEMPLATE_ELEC);
model.defaults.template_elec = eeg.elec;

% contrast.mat = [1 0 0 0 -1 0 0 0 0;
%                 0 1 0 0 0 -1 0 0 0;
%                 0 0 1 0 -1 0 0 0 0;
%                 0 0 0 1 0 -1 0 0 0];

%t-test
% contrast.mat = [1 0 1 0 1 0 1 0 0;
%                 0 1 0 1 0 1 0 1 0];

%1 sample t-test nat vs man and ANOVA semantic vs non-semantic and nat vs man
% contrast.mat = [0 1 -1 1 -1 1 -1 0 0 0];
contrast.mat = [1 -1 1 -1  1 -1 0 0 0;
                1  1 1  1 -2 -2 0 0 0];
            
% save(fullfile(PATH_TO_DERIV,'new_regressed_model.mat'),'model')
% save(fullfile(PATH_TO_DERIV,'contrast.mat'),'contrast')

% %uncomment if you want to load an existing model/contrast
% model = load(fullfile(PATH_TO_DERIV,'model.mat'));
% model = load(fullfile(PATH_TO_DERIV,'regressed_model.mat'));
% model = model.model;
% contrast = load(fullfile(PATH_TO_DERIV,'contrast.mat'));
% contrast = contrast.contrast;

cd(PATH_TO_ROOT)
option = 'both';
[LIMO_files, procstatus] = limo_batch(option,model,contrast);
% option = 'model specification';
% [LIMO_files, procstatus] = limo_batch(option,model);
% option = 'contrast only';
% [LIMO_files, procstatus] = limo_batch(option,model,contrast);


% ft_plot_mesh(vol,'facealpha',0.5)
% hold on
% scatter3(eeg.elec.elecpos(1:32,1),eeg.elec.elecpos(1:32,2),eeg.elec.elecpos(1:32,3),500,'g','filled')

%% call limo_random_select
clc;
LIMOfiles = fullfile(pwd,'Beta_files_GLM_OLS_Time_Channels.txt');

%expected_chanlocs = mean chanlocs
expected_chanlocs = limo_avg_expected_chanlocs(PATH_TO_DERIV, model.defaults);

% LIMOPath = limo_random_select('Repeated Measures ANOVA',expected_chanlocs,'LIMOfiles',... 
%     LIMOfiles,'analysis_type','Full scalp analysis','parameters',{[1 2],[3 4]},...
%     'factor names',{'semantic_relation', 'type_of_object'},'type','Channels','nboot',1000,'tfce',1,'skip design check','yes');

%Anova
cd(PATH_TO_ROOT)
LIMOfiles = fullfile(pwd,'simple_Beta_files_GLM_OLS_Time_Channels.txt');
if ~exist('simple_anova','dir')
    mkdir('simple_anova')
end
cd('simple_anova')

if ~exist('test_anova','dir')
    mkdir('test_anova')
end
cd('test_anova')

% LIMOfiles = fullfile(pwd,'regressed_Beta_files_GLM_OLS_Time_Channels.txt');
% if ~exist('regressed_anova','dir')
%     mkdir('regressed_anova')
% end
% cd('regressed_anova')

LIMOPath = limo_random_select('Repeated Measures ANOVA',expected_chanlocs,'LIMOfiles',... 
    LIMOfiles,'analysis_type','Full scalp analysis','parameters',{[1 2],[5 6]},...
    'factor names',{'semantic_relation', 'type_of_object'},'type','Channels','nboot',100,'tfce',1,'skip design check','yes');

%one sample t-test
cd(PATH_TO_ROOT)
my_con = 'con2';
LIMOfiles = fullfile(pwd,sprintf('%s_files_GLM_OLS_Time_Channels.txt',my_con));
if ~exist(['one_sample_t_test_' my_con],'dir')
    mkdir(['one_sample_t_test_' my_con])
end
cd(sprintf('one_sample_t_test_%s',my_con))

LIMOPath = limo_random_select('one sample t-test',expected_chanlocs,'LIMOfiles',... 
    LIMOfiles,'analysis_type','Full scalp analysis','parameters',{1},...
    'type','Channels','nboot',100,'tfce',1,'skip design check','yes');

%paired t-test
cd(PATH_TO_ROOT)
my_con = 'con1';
LIMOfiles = {fullfile(pwd,sprintf('%s_files_GLM_OLS_Time_Channels.txt',my_con)); fullfile(pwd,sprintf('%s_files_GLM_OLS_Time_Channels_simple.txt',my_con))};
if ~exist(['paired_t_test_' my_con],'dir')
    mkdir(['paired_t_test_' my_con])
end
cd(sprintf('paired_t_test_%s',my_con))

% LIMOPath = limo_random_select('paired t-test',expected_chanlocs,'LIMOfiles',... 
%     LIMOfiles,'analysis_type','Full scalp analysis','parameters',{[1:2]},...
%     'type','Channels','nboot',100,'tfce',1,'skip design check','yes');
LIMOPath = limo_random_select('paired t-test',expected_chanlocs,'LIMOfiles',... 
    LIMOfiles,'analysis_type','Full scalp analysis',...
    'type','Channels','nboot',100,'tfce',1,'skip design check','yes');

%% display results

load(fullfile(LIMOPath,'LIMO.mat'))
limo_review(LIMO)
% test = LIMO;
% test.design.weights = zeros(size(LIMO.design.weights));
% test.design.weights(:,LIMO.design.X(:,1)==1) = 1;
% test.design.weights(:,LIMO.design.X(:,2)==1) = 1;
% imagesc(test.design.weights)

limo_semi_partial_coef(LIMO); %output dim: channel*time*(R2,F-value,p-value)

figure
% hold on
test = squeeze(semi_partial_coef(:,:,1));
plot(LIMO.data.timevect,test')
[~,idx] = min(abs(LIMO.data.timevect-41.75));
[~,idx] = min(test(:,idx));
figure;plot(LIMO.data.timevect,test(idx,:)')

%7 at 248.8; 60 at 41.75; 49 at 412.8

%% Plot partial coef at time of interest
signif_effect = [];
for i=1:128
    cov_effect = load(['D:\__EEG-data\BIDS_files\derivatives\sub-001\eeg\regressed_GLM_OLS_Time_Channels\Covariate_effect_' num2str(i) '.mat']);
    cov_effect = cov_effect.(cell2mat(fieldnames(cov_effect)));
    
    effect = mean(mean(cov_effect([4:12 38:47],179:183,1),1));
    test = mean(cov_effect([4:12 38:47],179:183,1),1);
    bar(effect)
    
    [elec, time, subj, cov_effect] % bar plot mean, then std
    
    if min(TOI)<0.95
        signif_effect = [signif_effect i];
%         signif_effect = [signif_effect min(TOI)];
%         figure();plot(TOI)
    end
end

%% Plot Betas distribution
group_weight = [];
for subj = length(model.set_files):-1:1
    if subj >= 10
        subfolder = ['sub-0' num2str(subj)];
    else
        subfolder = ['sub-00' num2str(subj)];
    end
    betas = load(['D:\__EEG-data\BIDS_files\derivatives\' subfolder '\eeg\GLM_OLS_Time_Channels\Betas.mat']);
    betas = betas.(cell2mat(fieldnames(betas)));
    
    weight_eval = squeeze(mean(mean(abs(betas([4:12 38:47],179:183,9:end-1)),1),2));
    % [weights,idx] = sort(weight_eval,'ascend');
    % figure;barh(weights)
    % yticks(1:105)
    % yticklabels(idx)

    weight_eval = reshape(weight_eval,length(trialinfo.Properties.VariableNames)-3,8);
    group_weight(:,subj) = mean(weight_eval,2);
    
end

mean_weight = mean(group_weight,2);
[weights,idx] = sort(mean_weight,'ascend');

fields = trialinfo.Properties.VariableNames(4:end);
for i = 1:length(fields)
    fields{i} = strrep(fields{i},'_','-');
end
figure;barh(weights)
yticklabels(fields(idx));

%% Source analysis
PATH_TO_SOURCE = 'D:\__EEG-data\BIDS_source';

%% Compute source ROI activity for each condition
% subfolder = 'sub-0%d';
% i = 30; %subject ID
% eeg = load(fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_eeg.mat']));
% eeg = struct2cell(eeg);
% eeg = eeg{1};

conditions = unique(eeg.trialinfo.condition);
nb_conditions = nnz(conditions(~isnan(conditions)));
source_all_cond = cell(1,nb_conditions);

% select target ROIs
roi_mat = {[3,5];[4,6];[7,9];[8,10];[11,13,15];[12,14,16];61;62;63;64;65;66;67;68;31;32;[83,87];[84,88];85;86;[45,49,51,53];[46,50,52,54]};
[roi_atlas] = select_roi(sourcemodel_atlas,roi_mat);

for i = 1:nb_conditions
    cfg = [];
    cfg.trials = find(eeg.trialinfo.condition == i)';
    eeg_single_cond = ft_preprocessing(cfg,eeg);

    cfg                     = [];
    cfg.method              = 'mne';                    %specify minimum norm estimate as method
    cfg.sourcemodel         = sourcemodel_and_leadfield;%the precomputed leadfield
    cfg.headmodel           = vol;                      %the head model
    cfg.elec                = eeg.elec;                 %the electrodes
    cfg.channel             = eeg.label;                %the useful channels
    cfg.mne.prewhiten       = 'yes';                    %prewhiten data
    cfg.mne.lambda          = 0.01;                     %regularisation parameter
%     cfg.mne.scalesourcecov  = 'yes';                    %scaling the source covariance matrix
    cfg.rawtrial            = 'yes';
    cfg.keeptrials          = 'yes';
    source_dipole = ft_sourceanalysis(cfg, eeg_single_cond);
    
    % Compute ROI-by-ROI source activity
    source_roi = dipole2roi(source_dipole,roi_atlas);
    source_all_cond{i} = source_roi;
end

% Concatenate ROI activity of all the conditions
cfg            = [];
cfg.parameter  = 'mom';
source_roi = ft_appendsource(cfg,source_all_cond{:});

% Add label and trial fields (required for LIMO) and save
source_roi.label = source_all_cond{1}.label;
R = ones(1,size(source_roi.mom,1));
source_roi.trial = (mat2cell(source_roi.mom,R))';
source_roi.trial = cellfun(@(x) reshape(x,[],size(source_roi.mom,3)),source_roi.trial,'un',0);

subfolder = 'sub-0%d';
i=30;
output_path = fullfile(PATH_TO_SOURCE,'derivatives',sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_source-roi.mat']);
save(output_path,'source_roi')

%% Set the model and compute the neighbouring matrix
% subfolder = 'sub-0%d';
% i = 30; %subject ID
% source_dipole = load(fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_source.mat']));
% source_dipole = struct2cell(source_dipole);
% source_dipole = source_dipole{1};

model.set_files{1} = output_path;
model.cat_files = {};
model.cont_files = {};
tmp = [];
for i = 1:nb_conditions
    tmp = [tmp; i*ones(nnz(eeg.trialinfo.condition==i),1)];
end
model.cat_files{1} = tmp;

neighbouring_matrix = source_neighbmat(roi_atlas,0);

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
colormap(gray)

%% Model design
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

model.defaults.neighbouring_matrix = neighbouring_matrix;
%% Run limo_batch on sources
option = 'both';
% option = 'model specification';
% contrast.mat = [1 -1 1 -1  1 -1 0 0 0];
contrast.mat = [1 -1 1 -1  1 -1 0 0 0;
                1  1 1  1 -2 -2 0 0 0];

cd(PATH_TO_SOURCE)
[LIMO_files, procstatus] = limo_batch(option,model,contrast);
% [LIMO_files, procstatus] = limo_batch(option,model);

%% Display results
load(LIMO_files.mat{1})
load(LIMO_files.Beta{1})
my_con = 2;
load(LIMO_files.con{1}{my_con})
% close all
% region = 7;
% tmp = squeeze(Betas(region,:,:));
tmp = squeeze(con(:,:,1));
% tmp = (tmp-mean(tmp,2))./std(tmp')';
tmp = zscore(tmp,[],2);
for i = 1:size(tmp,1)
    tmp(i,:) = smooth(tmp(i,:),10);
end

figure
hold on
my_legend = [];
% for i = 1:size(tmp,1)
% for i = 1:2:size(tmp,1) %left hemisphere
for i = 2:2:size(tmp,1) %right hemisphere
% for i = [1,3,4,6,7,9,12,13,14,15,16,17,18,19,22]
% for i = [22,19,18,17,15,14,13,7,1]
%     figure
%     plot(LIMO.data.timevect,tmp(:,[1:i-1, i+1:end]),'LineWidth',1)
%     hold on
    plot(LIMO.data.timevect,tmp(i,:),'LineWidth',2)
%     hold on
%     line(LIMO.data.timevect,zeros(1,length(LIMO.data.timevect)),'LineWidth',2,'Color','black')
    region = strrep(LIMO.data.chanlocs(i).labels,'_','-');
    my_legend = [my_legend; {region}];
%     title(sprintf('Trimmed Mean Difference\nRegion: %s', region))
%     pause()
%     close all
%     plot(source_roi.time(154:513),Betas(region,:,i),'LineWidth',2)
%     plot(source_roi.time(154:513),source_roi.trial{i}(region,154:513),'LineWidth',2)
end
line(LIMO.data.timevect,zeros(1,length(LIMO.data.timevect)),'LineWidth',2,'Color','black')
grid on
lgd = legend(my_legend);
lgd.FontSize = 7;
% title(sprintf('Trimmed Mean Difference\nNatural vs. Manufactured'))
title(sprintf('Trimmed Mean Difference\nSemantic vs. Non-semantic'))
xlabel('time (ms)')
ylabel('Amplitude (z-score)')

% plot(source_roi.time(154:513),squeeze(Betas(region,:,:)),'LineWidth',2)
% title(sprintf('region: %s',source_roi.label{region}))
% legend
% figure;
% plot(LIMO.data.timevect,tmp(:,1),'LineWidth',2)
% grid on
% title('Trimmed Mean Difference')
