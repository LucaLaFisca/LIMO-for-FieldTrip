%% Call LIMO functions from FieldTrip files

%% (DATA SPECIFIC) Set paths and variable names
% required data (set to true for the first use of ft2limo)
CREATE_DERIV                = false;
CREATE_BIDS                 = false;
SOURCE_ANALYSIS             = true;
ANOVA                       = false;
T_TEST                      = false;
PAIRED_T_TEST               = false;

% paths
PATH_TO_RAW_EEG             = 'D:\__EEG-data';
PATH_TO_PROCESSED_EEG       = 'D:\__EEG-data\EEG_Erika_format\EEG';
PATH_TO_SOURCE              = 'D:\__EEG-data\BIDS_source';
PATH_TO_ELEC                = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Data';
PATH_TO_HEADMODEL           = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Data';
PATH_TO_FIELDTRIP           = 'D:\FieldTrip';
PATH_TO_LIMO                = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\3-LIMO\limo_tools';
PATH_TO_FT2LIMO             = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\3-LIMO\LIMO-for-FieldTrip';
PATH_TO_CUSTOM_FUNCTIONS    = 'C:\Users\luca-\OneDrive - UMONS\_PhD\_Matlab\0-General-Pipeline\EEG-Source-Analysis-Pipeline';
PATH_TO_ROOT                = 'D:\__EEG-data\BIDS_files';

% specific variables
raw_eeg_common_name         = 'ARC_J*';
processed_eeg_common_name   = 'J_*';
headmodel_subfolder         = 'J_';
elec_mat                    = 'elec_tmp.mat';
processed_eeg_mat           = 'clean_eeg.mat';
task_name                   = 'semantic-priming';
nb_elec                     = 64;

% output folder (derivatives)
PATH_TO_DERIV               = fullfile(PATH_TO_ROOT, 'derivatives');

% add toolboxes to path
addpath(PATH_TO_FIELDTRIP)
addpath(PATH_TO_LIMO)
addpath(genpath(fullfile(PATH_TO_LIMO,'external')))
addpath(genpath(fullfile(PATH_TO_LIMO,'limo_cluster_functions')))
addpath(genpath(PATH_TO_FT2LIMO))
addpath(PATH_TO_CUSTOM_FUNCTIONS)

BIDS_FOLDER = PATH_TO_ROOT;

cd(PATH_TO_ROOT)
%% (DATA SPECIFIC) Create derivatives files
if CREATE_DERIV
    create_deriv(PATH_TO_ROOT,PATH_TO_DERIV, PATH_TO_PROCESSED_EEG, task_name, processed_eeg_common_name, processed_eeg_mat)
end

%% (REQUIRED EEG_JSON FILE FOR SUB-001 TO BE MANUALLY CREATED) Create BIDS files
if CREATE_BIDS
    create_bids(PATH_TO_ROOT,PATH_TO_RAW_EEG,PATH_TO_ELEC,task_name,raw_eeg_common_name,elec_mat)
end

%% Compute source activity
if SOURCE_ANALYSIS
    % Compute source ROI activity for each condition

    conditions = unique(eeg.trialinfo.condition);
    conditions(conditions==reject_cat) = [];
    conditions(isnan(conditions)) = [];

    subj=30;
    if subj >= 10
        subfolder = 'sub-0%d';
    else
        subfolder = 'sub-00%d';
    end

    eeg = load(fullfile(PATH_TO_DERIV,sprintf(subfolder,i),'eeg',[sprintf(subfolder,i) '_task-' task_name '_eeg.mat']));
    eeg = eeg.(cell2mat(fieldnames(eeg)));
    eeg.elec.elecpos = eeg.elec.chanpos;
    eeg.elec.tra = eeg.elec.tra(:,1:64);
    sourcemodel_atlas = load(fullfile(PATH_TO_HEADMODEL,[headmodel_subfolder,num2str(subj)],'sourcemodel_atlas.mat'));
    sourcemodel_atlas = sourcemodel_atlas.(cell2mat(fieldnames(sourcemodel_atlas)));
    vol = load(fullfile(PATH_TO_HEADMODEL,[headmodel_subfolder,num2str(subj)],'vol.mat'));
    vol = vol.(cell2mat(fieldnames(vol)));
    sourcemodel_and_leadfield = load(fullfile(PATH_TO_HEADMODEL,[headmodel_subfolder,num2str(subj)],'leadfield.mat'));
    sourcemodel_and_leadfield = sourcemodel_and_leadfield.(cell2mat(fieldnames(sourcemodel_and_leadfield)));

    roi_mat = {[3,5];[4,6];[7,9];[8,10];[11,13,15];[12,14,16];61;62;63;64;65;66;67;68;31;32;[83,87];[84,88];85;86;[45,49,51,53];[46,50,52,54]};
    [source_roi,roi_atlas,output_path] = compute_source(PATH_TO_SOURCE,subj,eeg,roi_mat,reject_cat,sourcemodel_atlas,vol,sourcemodel_and_leadfield);

%     % Set the model and compute the neighbouring matrix
%     model.set_files{1} = output_path;
%     model.cat_files = {};
%     model.cont_files = {};
%     tmp = [];
%     for i = conditions'
%         tmp = [tmp; i*ones(nnz(eeg.trialinfo.condition==i),1)];
%     end
%     model.cat_files{1} = tmp;
% 
%     neighbouring_matrix = source_neighbmat(roi_atlas,0);
%     model.defaults.neighbouring_matrix = neighbouring_matrix;
end

%% (DATA SPECIFIC) Create the model and the contrast

%%% /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\ %%%
%%% CAREFULLY READ THE REQUESTED PARAMATER SYNTAX %%%
%%% /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\ %%%

% design the desired contrasts & regressor combination
%%% contrast syntax %%%
% [first_factor_combination_vector ;
%  second_factor_combination_vector;
%  nth_factor_combination_vector    ]
%%% regressor syntax %%%
% {first_regressor_merging, corresponding_value ;
%  second_regressor_merging, corresponding_value;
%  nth_regressor_merging, corresponding_value    }

my_contrast = 1; %in this example: nat vs man = 1; semantic vs non semantic = 2; both = 0
if my_contrast == 0
    % both conditions
    contrast.mat = [1 -1 1 -1  1 -1 0 0 0;
                    1  1 1  1 -2 -2 0 0 0];
    regress_cat = { 1:6 ,1;
                   [7,8],0};
elseif my_contrast == 1
    % natural vs manufactured
    contrast.mat = [1 -1 1 -1  1 -1 0 0 0];
    regress_cat = {[1,3,5] ,1;
                   [2,4,6] ,2;
                   [7,8]   ,0};
    
elseif my_contrast == 2
    % semantic vs non semantic
    contrast.mat = [1  1 1  1 -2 -2 0 0 0];
    regress_cat = { 1:4 ,1;
                   [5,6],2;
                   [7,8],0};
end

% select the desired regressors
my_trialinfo = 'combined_trialinfo.mat';
selected_regressors = [4:7,10,11]; %selection from trialinfo.Properties.VariableNames
trial_start = -200; %starting time of the trial in ms
trial_end = 500; %ending time of the trial in ms


% define rejected categories
%%% /!\/!\/!\/!\/!\ %%%
%%% WE NEED A CATEGORY TO REJECT TO FIX NaN ISSUES %%%
%%% BE SURE YOU DIDN'T ALLOCATE THIS LABEL TO A USEFUL CATEGORY %%%
reject_cat = 0; %we do not use the 0 category in our study (only there for experimental purpose)


% model is a structure that specifiy information to build a model
model = create_model(PATH_TO_DERIV,PATH_TO_SOURCE,SOURCE_ANALYSIS,task_name,my_trialinfo,trial_start,trial_end,selected_regressors,regress_cat,reject_cat);


%% Call limo_batch function
if SOURCE_ANALYSIS
    cd(PATH_TO_SOURCE)
else
    cd(PATH_TO_ROOT)
end
option = 'both'; % or 'model specification' or 'contrast only'
[LIMO_files, procstatus] = limo_batch(option,model,contrast);
% [LIMO_files, procstatus] = limo_batch(option,model); %if option=='model specification'

%% call limo_random_select
% select the desired contrast
my_con = 'con1';

if SOURCE_ANALYSIS
    cd(PATH_TO_SOURCE)
else
    cd(PATH_TO_ROOT)
end    
expected_chanlocs = limo_avg_expected_chanlocs(PATH_TO_DERIV, model.defaults);
    
%Anova
if ANOVA
    LIMOfiles = fullfile(pwd,'Beta_files_GLM_OLS_Time_Channels.txt');
    if ~exist('anova','dir')
        mkdir('anova')
    end
    cd('anova')
    LIMOPath = limo_random_select('Repeated Measures ANOVA',expected_chanlocs,'LIMOfiles',... 
        LIMOfiles,'analysis_type','Full scalp analysis','parameters',{[1,2],[5,6]},...
        'factor names',{'type_of_object', 'semantic_relation'},'type','Channels','nboot',100,'tfce',1,'skip design check','yes');
end

%one sample t-test
if T_TEST
    LIMOfiles = fullfile(pwd,sprintf('%s_files_GLM_OLS_Time_Channels.txt',my_con));
    if ~exist(['t_test_' my_con],'dir')
        mkdir(['t_test_' my_con])
    end
    cd(sprintf('t_test_%s',my_con))
    LIMOPath = limo_random_select('one sample t-test',expected_chanlocs,'LIMOfiles',... 
        LIMOfiles,'analysis_type','Full scalp analysis',...
        'type','Channels','nboot',100,'tfce',1,'skip design check','yes');
end

%paired t-test
if PAIRED_T_TEST
    LIMOfiles = {fullfile(pwd,sprintf('%s_files_GLM_OLS_Time_Channels_model1.txt',my_con)); fullfile(pwd,sprintf('%s_files_GLM_OLS_Time_Channels_model2.txt',my_con))};
    if ~exist(['paired_t_test_' my_con],'dir')
        mkdir(['paired_t_test_' my_con])
    end
    cd(sprintf('paired_t_test_%s',my_con))
    LIMOPath = limo_random_select('paired t-test',expected_chanlocs,'LIMOfiles',... 
        LIMOfiles,'analysis_type','Full scalp analysis',...
        'type','Channels','nboot',100,'tfce',1,'skip design check','yes');
end

%% Plot Betas distribution
if SOURCE_ANALYSIS
    deriv_path = fullfile(PATH_TO_SOURCE,'derivatives');
    my_time = [284,454];
    my_channel = {'Frontal_Mid_R and Frontal_Mid_Orb_R'};
    target_channel = source_roi.label;
else
    deriv_path = PATH_TO_DERIV;
    my_time = [284,454; 37,89; -58,-18];
    my_channel = [{'PO3'}; {'PO3'};{'CP4'}];
    target_channel = eeg.label;
end

timevect = LIMO.data.timevect;
[~,start_time] = find(timevect>=my_time(1),1);
[~, end_time] = find(timevect<=my_time(2),1,'last');

range_time = [];
range_channel = [];
for i = 1:size(my_channel,1)
    range_time = [range_time find((timevect>=my_time(i,1)) & (timevect<=my_time(i,2)))];
    channel_idx = find(strcmp(target_channel,my_channel(i,:)'));
    if length(channel_idx) == 1
        range_channel = [range_channel, channel_idx-1:channel_idx+1];
    else
        range_channel = [range_channel, channel_idx];
    end
end


group_weight = [];
i=1;
for my_path = model.set_files
    my_path = char(my_path);
    [root,~,~] = fileparts(my_path);
    betas = load([root '\GLM_OLS_Time_Channels\Betas.mat']);
    betas = betas.(cell2mat(fieldnames(betas)));
    weight_eval = squeeze(mean(mean(abs(betas(range_channel,range_time,1:end-1)),1),2));
    load([root '\reduced_trialinfo.mat'])
    weight_reshaped = reshape(weight_eval(7:end),length(trialinfo.Properties.VariableNames),2);
    group_weight(:,i) = [weight_eval(1:6);mean(weight_reshaped,2)];
    i=i+1;
end

mean_weight = mean(group_weight,2);
[weights,idx] = sort(mean_weight,'ascend');

cat_names = [];
for i = 1:6
    cat_names = [cat_names {['cat_' num2str(i)]}];
end
f = [cat_names trialinfo.Properties.VariableNames];
for i = 1:length(f)
    f{i} = strrep(f{i},'_','-');
end
figure;barh(weights)
xlabel('weights')
yticks(1:length(f))
yticklabels(f(idx));
title('variable weights at time-region of interest')


%% Display results on sources
load(LIMO_files.mat{1})
load(LIMO_files.Beta{1})
my_con = 1;
hemisphere = 1; %left(1) or right(2)
load(LIMO_files.con{1}{my_con})
tmp = squeeze(con(:,:,1));
tmp = zscore(tmp,[],2);
for i = 1:size(tmp,1)
    tmp(i,:) = smooth(tmp(i,:),10);
end

figure
hold on
my_legend = [];
for i = hemisphere:2:size(tmp,1) %right hemisphere
    plot(LIMO.data.timevect,tmp(i,:),'LineWidth',2)
    region = strrep(LIMO.data.chanlocs(i).labels,'_','-');
    my_legend = [my_legend; {region}];
end
line(LIMO.data.timevect,zeros(1,length(LIMO.data.timevect)),'LineWidth',2,'Color','black')
grid on
lgd = legend(my_legend);
lgd.FontSize = 7;
if my_con == 1
    title(sprintf('Trimmed Mean Difference\nNatural vs. Manufactured'))
elseif my_con == 2
    title(sprintf('Trimmed Mean Difference\nSemantic vs. Non-semantic'))
end
xlabel('time (ms)')
ylabel('Amplitude (z-score)')
