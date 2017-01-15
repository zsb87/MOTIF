protocol =  'inlabHiStr';
    win = 4;
    stride = 2;
    dict_size = 5;
    
    raw_fCol = 7;
    raw_nfCol = 8;
    engy_fCol = 8;
    engy_nfCol = 9;
    % define folder
    folder = '../../inlabHiStr/subject/';
    subjectname = [subj];
    subjfolder = [folder, subjectname,'/'];
    energyfolder = [subjfolder,'feature/energy/'];
    segfolder = [subjfolder, 'segmentation/'];
    featfolder = [subjfolder, 'feature/'];
    gtHtFolder = [folder, subj,'/segmentation/rawdata_gt/'];
%     
%     activities = { 
%         'answerPhone';           
%         'bottle';     
%         'brushTeeth';
%         'cheek';        
%         'chips';        
%         'chopsticks';     
%         'comb';            
%         'cup';
%         'drinkStraw';        
%         'forehead';
%         'fork';
%         'knifeFork';
%         'nose'; 
%         'pizza';
%         'restNearMouth';
%         'Phone';    
%         'smoke_im'; 
%         'smoke_ti';
%         'soupSpoon';
%     };

    