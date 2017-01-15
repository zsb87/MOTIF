protocol =  'inlabUnstr';
    win = 4;
    stride = 2;
    dict_size = 5;
    
    raw_fCol = 9;
    raw_nfCol = 10;
    engy_fCol = 8;
    engy_nfCol = 11;
    % define folder
    folder = '../../inlabUnstr/subject/';
    subjectname = [subj];
    subjfolder = [folder, subjectname,'/'];
    energyfolder = [subjfolder,'feature/energy/'];
    segfolder = [subjfolder, 'segmentation/'];
    featfolder = [subjfolder, 'feature/'];
    gtHtFolder = [folder, subj,'/segmentation/rawdata_gt/'];

    