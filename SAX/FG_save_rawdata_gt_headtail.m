function FG_save_rawdata_gt_headtail(subj, config_file)
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
    testdata_labeled = [folder, subj,'/testdata_labeled.csv'];
    disp(testdata_labeled);
    data = csvread(testdata_labeled,1,1);
    
    fClass = data(:,raw_fCol);
    nfClass = data(:,raw_nfCol);
    if ~exist(gtHtFolder, 'dir')    mkdir(gtHtFolder),   end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save all activities' ground truth in one file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    f_gt_headtail = pointwise2headtail(fClass);
    %   position:  folder, subj,'/segmentation/rawdata_gt/',
    %   here subj = testShibo
    save_headtail(f_gt_headtail, strcat(gtHtFolder,'gt_feeding_headtail.csv'));
    
%     disp(calcLabelStat(f_gt_headtail));
    
    nf_gt_headtail = pointwise2headtail(nfClass);
    %   position:  folder, subj,'/segmentation/rawdata_gt/'
    save_headtail(nf_gt_headtail, strcat(gtHtFolder,'gt_nonfeeding_headtail.csv'));
    
%     disp(calcLabelStat(nf_gt_headtail));
    
end
