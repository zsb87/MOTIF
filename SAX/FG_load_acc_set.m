function [train_sig_cell, train_gt_htcell, test_sig_cell, test_gt_global_htcell, test_gt_local_htcell] = FG_load_acc_set(test_subj,train_subj)

    win = 4;
    stride = 2;
    dict_size = 5;
    folder = '../../inlabStr/subject/';
    
    sigfile = [folder, train_subj,'(8Hz)/','testdata.csv'];
%     sigfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(sigfile,1,1);
    % z-normalization across each column.
    data = zscore(data);
    train_sig_cell{1} = data(:,4);    
    
    train_gt_htcell{1} = csvread([folder, train_subj,'(8Hz)/segmentation/rawdata_gt/gt_feeding_headtail.csv']);% it has (n,3) shape now

    
    
    folder = '../../inlabStr/subject/';
    sigfile = [folder, test_subj,'(8Hz)/','testdata.csv'];
    data = csvread(sigfile,1,1);
    data = zscore(data);
    test_sig_cell{1} = data(:,4);
    
    test_gt_global_htcell{1} = csvread([folder, test_subj,'(8Hz)/segmentation/rawdata_gt/gt_feeding_headtail.csv']);% it has (n,3) shape now
    test_gt_local_htcell{1} = test_gt_global_htcell{1};

end
