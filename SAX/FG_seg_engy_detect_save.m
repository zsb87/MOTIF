% --------------------------------------------------------------------
%   Unstructured
% 
%   usage: use motifs in ptns_cell to detect segments
% --------------------------------------------------------------------

function [test_pred_htcell, num_pred] = FG_seg_engy_detect_save(save_subj, motif_SAX_cell, test_sig_cell, std_thres, dist_thres, run, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file!_seg_detect')
    end
    
    num_pred = 0;
    test_pred_htcell = [];
    
    for i = 1:size(test_sig_cell,2)
        
        predfolder = [folder,save_subj, '/segmentation/engy_run',int2str(run),'_pred/'];
        if ~exist(predfolder, 'dir')   mkdir(predfolder),  end
        
        % define predict result file path
        tmpfolder = [folder,save_subj, '/segmentation/engy_run',int2str(run),'_tmp/'];
        if ~exist(tmpfolder, 'dir')   mkdir(tmpfolder),  end
        
        predfilepath = strcat(tmpfolder, ['pred_headtail_test', num2str(i), '.csv']);
        % save in multiple files, each piece of test signal in a file
        pred_reduce_filepath = strcat(predfolder, ['pred_headtail_reduced_', num2str(i), '.csv']);
        dist_folder = strcat(predfolder, 'dist_all/');
        if ~exist(dist_folder, 'dir')   mkdir(dist_folder),  end
        %-------------------------------------------------------------------------------
        % core function for judging a prediction
        % detect among each piece of test data and save in 'pred_acc_headtail_reduced_i.csv'
        %-------------------------------------------------------------------------------     
        [test_pred_ht, num_pred_tmp] = finding_segmentation(motif_SAX_cell, test_sig_cell{i}, std_thres, dist_thres, dict_size, predfilepath, pred_reduce_filepath, dist_folder);
        num_pred = num_pred + num_pred_tmp;
        test_pred_htcell{i} = test_pred_ht;
    end
end
