function [source_roi,roi_atlas,output_path] = compute_source(PATH_TO_SOURCE,task_name,subj,eeg,roi_mat,conditions,sourcemodel_atlas,vol,sourcemodel_and_leadfield)
nb_conditions = nnz(conditions);
source_all_cond = cell(1,nb_conditions);

% select target ROIs
[roi_atlas] = select_roi(sourcemodel_atlas,roi_mat);
save(fullfile(PATH_TO_SOURCE,'derivatives',sprintf(subfolder,subj),'eeg',[sprintf(subfolder,subj) '_task-' task_name '_roi-atlas.mat']),'roi_atlas')

for i = conditions'
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

if subj >= 10
    subfolder = 'sub-0%d';
else
    subfolder = 'sub-00%d';
end
    
output_path = fullfile(PATH_TO_SOURCE,'derivatives',sprintf(subfolder,subj),'eeg',[sprintf(subfolder,subj) '_task-' task_name '_source-roi.mat']);
save(output_path,'source_roi')