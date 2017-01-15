% 
% INPUT FILE:   testdata_labeled.csv
%               engy_ori_win4_str2_labeled.csv
% 
%  for US, qualified subjs: Dzung Shibo Rawan JC Jiapeng Matt
%  for US, problematic subjs:  Dzung Shibo 

function brecall = FG_main_engy_0107(subj, run, dist_thres, n_motif, config_file)

% #           0       1       2      3      4       5        6      7     8   9(no HS)    10
% subjs = ['Eric','Dzung','Gleb','Will','Shibo','Rawan','Jiapeng','JC','Cao','Matt', 'MattSmall']
% subjs = {'Eric','Dzung','Gleb','Will'}; %'Rawan','JC','Cao','Jiapeng','Eric',
%problem subject: 'Matt','Will','Gleb' data missing
try
    eval(config_file);
catch
    disp('config file!_main')
end

motif_sel_mode = 3;

if 1
        train_subj = ['train',subj];
        test_subj = ['test',subj];
        
        result = [];
        meas_thres_all=[];
        recall_all=[];
        precision_all = [];
        num_gt_all = [];
        num_ptn_all = [];
        num_pred_all = [];
        dist_thre_all = [];

        %=================================================================
        %   save in 'gt_feeding_headtail.csv'
        %=================================================================
        FG_save_rawdata_gt_headtail(test_subj, config_file);
        FG_save_engy_gt_headtail(test_subj, config_file);
        FG_save_engy_gt_headtail(train_subj, config_file);
        [test_sig_cell, test_gt_global_htcell, test_gt_local_htcell, train_sig_cell, train_gt_htcell ] = FG_load_engy_set(test_subj, train_subj, config_file);
        [motif_SAX_cell] = FG_motif_sel(train_sig_cell, train_gt_htcell, config_file, motif_sel_mode, n_motif, subj);
        num_motif = size(motif_SAX_cell,2);
        
%         draw standard motifs generated
        motiffig = figure();
        for c = 1:num_motif
            figure(motiffig);
            subplot(ceil(num_motif/4), 4, c);
            plot(motif_SAX_cell{c});
            title(strcat(subj,' :generated std motif'));
        end
        
                
                
        std_thres = 0.01;
        [train_pred_htcell, num_pred_train] = FG_seg_engy_detect_save(train_subj, motif_SAX_cell, train_sig_cell, std_thres, dist_thres, run, config_file);  
        [test_pred_htcell, num_pred_test] = FG_seg_engy_detect_save(test_subj, motif_SAX_cell, test_sig_cell, std_thres, dist_thres, run, config_file);      

        % for Rawan, 0.01 is better; for jiapeng, 0 is better
        for meas_thres = 0.5:0.1:0.9
                std_thres = 0.01;
                
                if motif_sel_mode == 1
%                     [num_gt, num_TP, recall] = FG_seg_pred_trueOrFalse_accang(subj, config_file);
                else % 2 or 3
                    [seg_label_cell, recall] = FG_seg_measure(test_pred_htcell, test_gt_local_htcell, meas_thres, config_file);
                end

                % save test set labels to csv file
                labels = [];

                % modify here "test/train"
                for n= 1:size(seg_label_cell,2)  labels=[labels;seg_label_cell{n}];  end            
                folder = ['../../',protocol,'/subject/',test_subj,'/segmentation/engy_run',num2str(run),'_pred_label_thre',num2str(meas_thres)];
                if ~exist(folder,'dir') mkdir(folder), end   
                csvwrite([folder,'/seg_labels.csv'],labels);


                num_gt = 0;
                for n = 1:size(test_gt_local_htcell, 2)
                    num_gt = num_gt + size(test_gt_local_htcell{n}, 1);
                end

                meas_thres_all = [meas_thres_all, meas_thres];
                num_gt_all = [num_gt_all, num_gt];
                num_pred_all = [num_pred_all, num_pred_test];
                dist_thre_all = [dist_thre_all, dist_thres];
                recall_all = [recall_all, recall];
        end

        for meas_thres = 0.5:0.1:0.9
                std_thres = 0.01;
                
%                 [train_pred_htcell, num_pred_train] = FG_seg_engy_detect_read(train_subj, run, config_file);

                if motif_sel_mode == 1
%                     [num_gt, num_TP, recall] = FG_seg_pred_trueOrFalse_accang(subj, config_file);
                else % 2 or 3
                    [seg_label_cell, recall] = FG_seg_measure(train_pred_htcell, train_gt_htcell, meas_thres, config_file);
                end

                % save test set labels to csv file
                labels = [];
                for n= 1:size(seg_label_cell,2)  labels=[labels;seg_label_cell{n}];  end            
                folder = ['../../',protocol,'/subject/',train_subj,'/segmentation/engy_run',num2str(run),'_pred_label_thre',num2str(meas_thres)];
                if ~exist(folder,'dir') mkdir(folder), end   
                csvwrite([folder,'/seg_labels.csv'],labels);

                num_gt = 0;
                for n = 1:size(train_gt_htcell, 2)
                    num_gt = num_gt + size(train_gt_htcell{n}, 1);
                end

                meas_thres_all = [meas_thres_all, meas_thres];
                num_gt_all = [num_gt_all, num_gt];
                num_pred_all = [num_pred_all, num_pred_train];
                dist_thre_all = [dist_thre_all, dist_thres];
                recall_all = [recall_all, recall];
        end

        result = [result; meas_thres_all];
        result = [result; dist_thre_all];
        result = [result; num_gt_all];
        result = [result; num_pred_all];
        result = [result; recall_all];
        disp(result');
        folder = ['../../',protocol,'/result/segmentation/'];
        if ~exist(folder,'dir')     mkdir(folder),    end    

        result = result';
        brecall = result(1,end);
        
        resultfile_all = ['engy_run',num2str(run),'_result_',test_subj,'_dict',int2str(dict_size),'_thre',num2str(dist_thres),'_f',num2str(brecall),'.csv'];
        resultfile_allpath = [folder, resultfile_all];
        
        csvwrite(resultfile_allpath, double(result));
        
end
