function FG_save_engy_gt_headtail(subj, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file!')
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % save all activities' ground truth in 'gt_feeding_headtail.csv'
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sigfolder = [folder, subj,'/feature/energy/'];
    sigfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(strcat(sigfolder,sigfile),1);
    
    disp(size(data));
    fClass = data(:,engy_fCol);
    disp(engy_nfCol)
    nfClass = data(:,engy_nfCol);
    
    gtHtFolder = [folder, subj,'/segmentation/engy_gt/'];
    if ~exist(gtHtFolder, 'dir')    mkdir(gtHtFolder),   end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save all activities' ground truth in one file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gt_headtail = pointwise2headtail(fClass);
    save_headtail(gt_headtail, strcat(gtHtFolder,'gt_feeding_headtail.csv'));
    
    nf_headtail = pointwise2headtail(nfClass);
    save_headtail(nf_headtail, strcat(gtHtFolder,'gt_nonfeeding_headtail.csv'));
    
end
