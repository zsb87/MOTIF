function [test_pred_htcell, num_pred] = FG_seg_engy_detect_read(save_subj,run, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!_seg_detect')
    end
    
    num_pred = 0;
    test_pred_htcell = [];
    
    for i = 1
        
        predfolder = [folder,save_subj, '/segmentation/engy_run',num2str(run),'_pred/'];
        if ~exist(predfolder, 'dir')   mkdir(predfolder),  end
        pred_reduce_filepath = strcat(predfolder, ['pred_headtail_reduced_', num2str(i), '.csv']);
        
        %-------------------------------------------------------------------------------
        % core function for judging a prediction
        % detect among each piece of test data and save in 'pred_acc_headtail_reduced_i.csv'
        %-------------------------------------------------------------------------------    
        headtaillendist = csvread(pred_reduce_filepath);
        test_pred_ht = headtaillendist(:,1:3);
        num_pred_tmp = size(headtaillendist,1);
        
        num_pred = num_pred + num_pred_tmp;
        test_pred_htcell{i} = test_pred_ht;
    end
end
