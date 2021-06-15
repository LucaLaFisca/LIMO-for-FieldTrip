function create_bids(PATH_TO_ROOT,PATH_TO_RAW_EEG,PATH_TO_ELEC,task_name,raw_eeg_common_name,elec_mat)
json_path = fullfile(PATH_TO_ROOT,sprintf('sub-00%d',1),'eeg',[sprintf('sub-00%d',1) '_task-' task_name '_eeg.json']);
channel_path = fullfile(PATH_TO_ROOT,sprintf('sub-00%d',1),'eeg', [sprintf('sub-00%d',1) '_task-' task_name '_channels.tsv']);
if ~exist(json_path,'file')
    error('ERROR ! NEED EEG_JSON.TSV FILE FOR SUB-001 TO BE MANUALLY CREATED')
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
    [root,~,~] = fileparts(output_path);
    if ~exist(root,'dir')
        mkdir(root)
    end
    copyfile(fullfile(PATH_TO_RAW_EEG,dinfo(iFile).name),output_path);

    % create json & channels file
    if iFile ~= 1
        copyfile(json_path,[output_path(1:end-3) 'json']);
        copyfile(channel_path,[output_path(1:end-7) 'channels.tsv']);
    end
    
    % create channels file
    elec = load(fullfile(PATH_TO_ELEC,sprintf(subfolder, iFile),elec_mat),'-mat');
    elec = elec.(cell2mat(fieldnames(elec)));
    
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