    protocol =  'inlabStr';
    win = 4;
    stride = 2;
    dict_size = 8;
    
    raw_fCol = 9;
    raw_nfCol = 10;
    engy_fCol = 8;
    engy_nfCol = 11;
    % define folder
    folder = '../../inlabStr/subject/';
    subjectname = [subj];
    subjfolder = [folder, subjectname,'/'];
    energyfolder = [subjfolder,'feature/energy/'];
    segfolder = [subjfolder, 'segmentation/'];
%     act_rootfolder = [segfolder, 'activity/'];
    featfolder = [subjfolder, 'feature/'];
    gtHtFolder = [folder, subj,'/segmentation/rawdata_gt/'];
    
