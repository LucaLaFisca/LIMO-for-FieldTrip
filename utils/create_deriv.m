function create_deriv(PATH_TO_ROOT, PATH_TO_DERIV, PATH_TO_PROCESSED_EEG, task_name, processed_eeg_common_name, processed_eeg_mat)
%% (DATA SPECIFIC) Create derivatives files
dinfo = dir(fullfile(PATH_TO_PROCESSED_EEG,processed_eeg_common_name));
subj = {dinfo.name};

subj_ID = 1;
for subj_name = drange(subj)
    processed_eeg = load(fullfile(PATH_TO_PROCESSED_EEG,subj_name{1},processed_eeg_mat));
    processed_eeg = processed_eeg.(cell2mat(fieldnames(processed_eeg)));
    
    cfg = [];
    deriv_mat = ft_appenddata(cfg,processed_eeg{:});
    
    if subj_ID >= 10
        subfolder = 'sub-0%d';
    else
        subfolder = 'sub-00%d';
    end
    derivatives_path = fullfile(PATH_TO_DERIV,sprintf(subfolder,subj_ID),'eeg', [sprintf(subfolder,subj_ID) '_task-' task_name '_eeg.mat']);
    [root,~,~] = fileparts(derivatives_path);
    if ~exist(root,'dir')
        mkdir(root)
    end
    % add electrode positions
    elec = tdfread(fullfile(PATH_TO_ROOT,sprintf(subfolder,subj_ID),'eeg', [sprintf(subfolder,subj_ID) '_task-' task_name '_channels.tsv']));
    % channels is on the hardware part, not the scalp => electrode positions have to be defined elsewhere
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