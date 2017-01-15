protocol =  'inlabUnstr';
    win = 4;
    stride = 2;
    dict_size = 15;
    
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

    