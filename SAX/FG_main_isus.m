clear;
%'Shibo','Dzung',  'JC', 'Cao','Jiapeng','Eric','Rawan','Gleb','Will','Matt'
subjs = {'Dzung'}; %'Shibo','Dzung',  'JC', 'Cao','Jiapeng','Eric','Rawan'
%problem subject: 'Matt','Will','Gleb' data missing
protocol =  'inlabUnstr';%'inlabStr';
config_file = 'config_file_us';%'config_file_us';
motif_sel_mode = 3;

for i = 1:size(subjs,2)
    train_subj = ['train',subjs{i}];
%     test_subj = ['test',subjs{i}];
    
    result = [];
    recall_all=[];
    precision_all = [];
    num_gt_all = [];
    num_ptn_all = [];
%     num_TP_all = [];
    % num_det_acc_all = [];
    % num_det_ang_all = [];
    num_pred_all = [];
    dist_thre_all = [];
    
    %=================================================================
    %   save in 'gt_feeding_headtail.csv'
    %=================================================================
%     FG_save_gt(['test',subjs{i}], config_file);
    
    % for Rawan, 0.01 is better; for jiapeng, 0 is better
    for std_thres = 0.01

        for dist_thres = 0.7
%         for run = 1:1
            
            std_thres = 0.05;%0.005
%             dist_thres = 0.6;%0.3

            %=================================================================
            % select motifs and split train test set
            % ***train_sig_cell, test_sig_cell are not SAX
            %=================================================================
%             [train_sig_cell, train_gt_htcell, test_sig_cell, test_gt_global_htcell, test_gt_local_htcell] = FG_set_split(subj, config_file);
            [test_sig_cell, test_gt_global_htcell, test_gt_local_htcell, train_sig_cell, train_gt_htcell] = FG_load_engy_set(train_subj,train_subj);
            
            
            % save test signals gt global head tails
            
%             test_gt_global_ht = [];
%             for nn=1:size(test_gt_global_htcell,2)
%                 test_gt_global_ht = [test_gt_global_ht; test_gt_global_htcell{nn}];
%             end
%             
%             folder = ['../../',protocol,'/subject/',train_subj,'(8Hz)/segmentation/test_data_global_ht/'];
%             if ~exist(folder,'dir') mkdir(folder), end
%             csvwrite([folder,'test_data_global_ht.csv'],test_gt_global_ht);
                       
            
            %=================================================================
            % select motifs and split train test set
            %=================================================================
            [motif_SAX_cell] = FG_motif_sel( train_sig_cell, train_gt_htcell, config_file, motif_sel_mode);
            num_motif = size(motif_SAX_cell,2);
            
            
            %=================================================================
            % detect segments and save in 'pred_acc_headtail_reduced_(i).csv'
            %=================================================================
            % test_pred_htcell: each element test_pred_ht is a N*3 matrix
            [test_pred_htcell, num_pred] = FG_seg_detect_save(train_subj, motif_SAX_cell, test_sig_cell, std_thres, dist_thres, config_file);
            
            
            %=================================================================
            % save in file 'AllGestures/ACTIVITY/SUBJECT/pred_ang_headtail_ACTIVITY.csv'
            %=================================================================
%             FG_detection_for_activity_energyang(subj, std_thres, dist_thres, 'config_file_us');
            
            
            % remove the repeated segments detected
            %=================================================================
            % save in file 'activity/pred_headtail_all.csv'
            %=================================================================
%             [num_det_accang] = FG_remove_repeatedSeg(subj, 'config_file_us');
            
            %-----------------------------------------------------------------------
            % in the label's view, also save the 'T/F' label of prediction
            % segment when check if the gt is covered
            %-----------------------------------------------------------------------
            %=================================================================
            % save in file 'activity/pred_accang_label.csv'
            %=================================================================
            if motif_sel_mode == 1
                [num_gt, num_TP, recall] = FG_seg_pred_trueOrFalse_accang(subj, config_file);
            else % 2 or 3
                [seg_label_cell, recall] = FG_seg_measure(test_pred_htcell, test_gt_local_htcell, config_file);
            end
            
            % write test set labels to csv file
            labels = [];
            for n= 1:size(seg_label_cell,2)  labels=[labels;seg_label_cell{n}];  end
            
            folder = ['../../',protocol,'/subject/',train_subj,'(8Hz)/segmentation/pred_label'];
            if ~exist(folder,'dir') mkdir(folder), end   
            csvwrite([folder,'/seg_labels.csv'],labels);            
            
            num_gt = 0;
            for n = 1:size(test_gt_local_htcell, 2)
                num_gt = num_gt + size(test_gt_local_htcell{n}, 1);
            end
            
            num_gt_all = [num_gt_all, num_gt];
            num_ptn_all = [num_ptn_all, num_motif];
%             num_det_acc_all = [num_det_acc_all, num_det_acc];
%             num_det_ang_all = [num_det_ang_all, num_det_ang];
            num_pred_all = [num_pred_all, num_pred];
%             num_TP_all = [num_TP_all, num_TP];
            dist_thre_all = [dist_thre_all, dist_thres];
            recall_all = [recall_all, recall];
        end
    end

    result = [result; dist_thre_all];
    result = [result; num_gt_all];
    % result = [result; num_det_acc_all];
    % result = [result; num_det_ang_all];
    
    result = [result; num_ptn_all];
    result = [result; num_pred_all];
    result = [result; recall_all];
    disp(result');
    folder = ['../../',protocol,'/result/segmentation/'];
    if ~exist(folder,'dir')
        mkdir(folder);
    end    
    resultfile_all = ['result_',train_subj,'_Msel',int2str(motif_sel_mode),'_setsplit_thre',int2str(dist_thres),'_new.csv'];
    resultfile_allpath = [folder, resultfile_all];
    csvwrite(resultfile_allpath, result');

end
