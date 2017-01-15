function [train_sig_cell, train_gt_htcell, test_sig_cell, test_gt_global_htcell, test_gt_local_htcell] = FG_set_split(subj, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!_set_split')
    end
        
    train_sig_cell = [];
    train_gt_htcell = [];
    test_sig_cell = [];
    test_gt_global_htcell = [];
    
    %% read energy file
    %% check the column number
    sigfolder = [folder, subj,'(8Hz)/feature/energy/'];
    sigfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(strcat(sigfolder,sigfile),1);
    sig_all = data(:,2);
    
    
    % read headtail files generated in FG_save_gt(subj, config_file) function    
    gt_headtail = csvread([folder, subj,'(8Hz)/segmentation/gt/gt_feeding_headtail.csv']);% it has (n,3) shape now
    
    
    % expend gt_headtail from 3 columns to 4 columns
    append_row = gt_headtail(2:end,1)-1;
    gt_headtail = [gt_headtail,[append_row;length(sig_all)]]; % gt_headtail is a 4 column matrix
    
    
    % randomly split 'gt matrix' to 7:3 train test set
    rand_gt = gt_headtail(randperm(size(gt_headtail,1)),:); % a random permutation of your data
    train_HT = rand_gt(1:floor(size(gt_headtail,1)*0.7),:);
    test_HT = rand_gt(floor(size(gt_headtail,1)*0.7)+1:end,:);
    
    
    train_sig_cell = [];
    for i = 1: size(train_HT,1)            
        train_sig_cell{i} = sig_all(train_HT(i,1): train_HT(i,4));            
        train_gt_htcell{i} = train_HT(i,:);        
    end
    
    test_sig_cell = [];
    for i = 1: size(test_HT,1)            
        test_sig_cell{i} = sig_all(test_HT(i,1): test_HT(i,4));            
        test_gt_global_htcell{i} = test_HT(i,:);        
    end
    
    % make all test/train_gt_htcell start from 1
    for i = 1: size(train_gt_htcell,2)            
        tmp_cell = train_gt_htcell{i};
        train_gt_htcell{i} = [1,tmp_cell(2)-tmp_cell(1)+1,tmp_cell(3),tmp_cell(4)-tmp_cell(1)+1];        
    end
    % make all test/train_gt_htcell start from 1
    for i = 1: size(test_gt_global_htcell,2)            
        tmp_cell = test_gt_global_htcell{i};
        test_gt_local_htcell{i} = [1,tmp_cell(2)-tmp_cell(1)+1,tmp_cell(3),tmp_cell(4)-tmp_cell(1)+1];        
    end
    
end
