function [model] = create_model(PATH_TO_DERIV,PATH_TO_SOURCE,SOURCE_ANALYSIS,task_name,my_trialinfo,trial_start,trial_end,selected_regressors,regress_cat,reject_cat)
% model is a structure that specifiy information to build a model
if SOURCE_ANALYSIS
    deriv_path = fullfile(PATH_TO_SOURCE,'derivatives');
    data_name = ['_task-' task_name '_source-roi.mat'];
else
    deriv_path = PATH_TO_DERIV;
    data_name = ['_task-' task_name '_eeg.mat'];
end
dinfo = dir(fullfile(deriv_path,'sub-*'));
model.set_files = {};
model.cat_files = {};
model.cont_files = {};
i=1;
for subj = {dinfo.name}
    subj = char(subj);
    model.set_files{i,1} = fullfile(deriv_path,subj,'eeg',[subj data_name]);
    eeg = load(fullfile(PATH_TO_DERIV,subj,'eeg',[subj '_task-' task_name '_eeg.mat']));
    eeg = eeg.(cell2mat(fieldnames(eeg)));
    trialinfo = load(fullfile(PATH_TO_DERIV,subj,'eeg',my_trialinfo));
    trialinfo = trialinfo.(cell2mat(fieldnames(trialinfo)));
    model.cat_files{i,1} = eeg.trialinfo.condition;
    model.cat_files{i,1}(isnan(model.cat_files{i,1})) = reject_cat;
    regressor = trialinfo.Properties.VariableNames(selected_regressors);
    cont = [];
    for r = regressor
        tmp = trialinfo.(r{1});
        tmp_cat = model.cat_files{i,1};
        for j = 1:size(regress_cat,1)
            tmp_cat(ismember(tmp_cat,regress_cat{j,1})) = regress_cat{j,2};
        end        
        cont = [cont,limo_split_continuous(tmp_cat,tmp)];
    end
    
    if exist('reject_cat','var') && ~isempty(reject_cat)
        tmp = sort(cell2mat(regress_cat(:,2)));
        [reject_idx, ~] = find(tmp==reject_cat);
        cont(:,reject_idx:size(regress_cat,1):end) = [];
    end
    
    % fix unwanted NaN values issues
    cont(sum(cont,2) == 0,:) = nan;
    for j = 1:size(cont,2)
        idx = find(isnan(cont(:,j)));
        for row = idx'
            if ~all(isnan(cont(row,:)))
                cont(row,j) = 0;
            end
        end
    end
    if SOURCE_ANALYSIS
        cont(model.cat_files{i,1}==0, :) = [];
        model.cat_files{i,1}(model.cat_files{i,1}==0) = [];
    end
    model.cont_files{i,1} = cont;
    i=i+1;
end
% model.defaults: specifiy the parameters to use for each subject
model.defaults.type = 'Channels'; %or 'Components'
model.defaults.analysis = 'Time'; %'Frequency' or 'Time-Frequency'
model.defaults.method = 'OLS'; %'IRLS' 'WLS'
model.defaults.type_of_analysis = 'Mass-univariate'; %or 'Multivariate'
model.defaults.fullfactorial = 0; %or 1
model.defaults.zscore = 0; %or 1
model.defaults.start = trial_start; %starting time in ms
model.defaults.end = trial_end; %ending time in ms
model.defaults.bootstrap = 0; %or 1
model.defaults.tfce = 0; %or 1

% uncomment if you wanna use a specific elec structure
% elec = load(fullfile(PATH_TO_ELEC,sprintf(subfolder, i),elec_mat),'-mat');
% elec = struct2cell(elec);
% elec = elec{1};
% eeg.elec = elec
if SOURCE_ANALYSIS
    roi_atlas = load(fullfile(deriv_path,subj,'eeg',[subj '_task-' task_name '_roi-atlas.mat']));
    roi_atlas = roi_atlas.(cell2mat(fieldnames(roi_atlas)));
    neighbouring_matrix = source_neighbmat(roi_atlas,0);
    model.defaults.neighbouring_matrix = neighbouring_matrix;
else
    elec_neighb = eeg.elec;
    elec_neighb.pnt = elec_neighb.chanpos;
    data_neighb = eeg;
    data_neighb.elec = elec_neighb;
    cfg = [];
    cfg.elec = elec_neighb;
    cfg.neighbourdist = 40; %defined in cm in limo_ft_neighbourselection
    [neighbours,channeighbstructmat] = limo_ft_neighbourselection(cfg,data_neighb); %REQUIRES last version of LIMO toolbox
    model.defaults.neighbouring_matrix = channeighbstructmat;
    % neighbouring matrix format: [n_chan x n_chan] of 1/0 (neighbour or not)
end

model.defaults.template_elec = eeg.elec;