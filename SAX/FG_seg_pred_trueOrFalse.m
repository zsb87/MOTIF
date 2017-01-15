% Copyright 
% 
% --------------------------------------------------------------------
% Foodwatch is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Foodwatch is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with HARPS. If not, see <http://www.gnu.org/licenses/>.
% --------------------------------------------------------------------

%
% Input:
%   data              is the raw time series. 
%   N                 is the length of sliding window (use the length of the raw time series
%                     instead if you don't want to have sliding windows)
%   n                 is the number of symbols in the low dimensional approximation of the sub sequence.
%   alphabet_size     is the number of discrete symbols. 2 <= alphabet_size <= 10, although alphabet_size = 2 is a special "useless" case.
%   NR_opt            1: no numerosity reduction (record everything)
%                     2: numerosity reduction (record only if the string is different from the last recorded string)
%                        (default)
%                     3: advanced numerosity reduction (record only if the mindist between current string and 
%                        last recorded string > 0)
%                     4: more reduction (record only if the subsequence is NOT monotonic)
%
% Output:
%   symbolic_data:    matrix of symbolic data (no-repetition).  If consecutive subsequences
%                     have the same string, then only the first occurrence is recorded, with
%                     a pointer to its location stored in "pointers"
%   pointers:         location of the first occurrences of the strings
% 
% save file:
%   
% 

function [num_gt, num_segments_all, recall] = FG_seg_pred_trueOrFalse(subj,config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!')
    end

    if ~exist(act_rootfolder, 'dir')
        mkdir(act_rootfolder);
    end

    % read ground truth for all activities in one file
    gtfile_all = 'gt_headtail_all_activities.csv';
    gtfile_allpath = [act_rootfolder, gtfile_all];
    
    
    %% --------------------------------------------------------------------
    %  read all activities'  GT  in a matrix
    %  --------------------------------------------------------------------
    if ~exist(gtfile_all, 'file')
        gt_all = [];
        for act_ind = 1:9
            % define ground truth file path
            gtfile_head = 'gt_headtail_';
            gtfile = [gtfile_head, activities{act_ind}, '.csv'];
            % define folder
            actfolder = [act_rootfolder, activities{act_ind},'/'];
            gtfilepath = strcat(actfolder, gtfile);
            
            if ~exist(gtfilepath, 'file')
                gt_act = [];
            else
                gt_act = csvread(gtfilepath);
            end
            
            gt_act = [gt_act, ones(size(gt_act,1),1)*act_ind];
            gt_all = [gt_all; gt_act];
            csvwrite(gtfile_allpath, gt_all);
        end
    else
        gt_all = csvread(gtfile_allpath);
    end
    

    %% --------------------------------------------------------------------
    %  read all pred results in cell
    %  --------------------------------------------------------------------
    for act_ind = 1:9

        % define folder
        actfolder = [segfolder, 'activity/', activities{act_ind},'/'];

        % define predict result file path
        pred_file_prefix = 'pred_acc_headtail_';
        predfile = [pred_file_prefix, activities{act_ind}, '.csv'];
        predfilepath = strcat(actfolder, predfile);

        % pattern file
        ptnfolder = [actfolder, 'pattern/'];
        ptnfile = ['patterns_acc_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
        ptnfilepath = strcat(ptnfolder, ptnfile);


        % return the number of patterns/motifs for each activity
        if ~exist(ptnfilepath, 'file')
            num_gt_act(act_ind) = 0;
        else
            import_struct = load(ptnfilepath);
            num_ptns(act_ind) = length(import_struct.ptns_cell);

            pred_act = csvread(predfilepath);
            pred_act = pred_act(:,1:2);
            pred_act = pred_act';
            pred_cell{act_ind} = pred_act(:);
            
        end
    end


    %% count the true positive and number of gt
    gt_all = gt_all(:,1:2);
    gt_all = gt_all';
    gt_all = gt_all(:);
    
    
    num_segments_all = 0;
    true_flg_concat = [];
    gt_detected_flg_all = zeros(length(gt_all)/2,1);
    
    % based on the (ref_ind)th activity patterns
    for ref_ind = 1:9
        % find similar patterns that actually belong to (test_ind)th activity
        pred = pred_cell{ref_ind};
        
        % count all the detected segments
        num_segments_all = num_segments_all + length(pred)/2;
        disp(num_segments_all);
        pred_true_flg_all = zeros(length(pred)/2,1);    
        [num_gt, gt_detected_flg, pred_true_flg] = seg_pred_trueOrFalse(pred, gt_all);

        gt_detected_flg_all = gt_detected_flg_all + gt_detected_flg;
        pred_true_flg_all = pred_true_flg_all + pred_true_flg;
        
        true_flg_concat = [true_flg_concat;pred_true_flg_all];

        actfolder = [segfolder, 'activity/', activities{ref_ind},'/'];
        csvwrite(strcat(actfolder,'pred_acc_label.csv'),pred_true_flg_all);
    end
    
    gt_detected_flg_all = sign(gt_detected_flg_all);
    recall = sum(gt_detected_flg_all)/length(gt_detected_flg_all);
    
    csvwrite(strcat(featfolder,'pred_acc_segment_label.csv'),true_flg_concat);
    
%     detection_flg_all = sign(detection_flg_all);
%     recall = sum(detection_flg_all)/length(detection_flg_all);
%     precision = sum(detection_flg_all)/num_detect_all;
    
%     num_tp_header = [0, num_gt_act];
%     num_tp_index = num_ptns';
%     num_tp_all = [num_tp_header;[num_tp_index, num_tp]];
%   
%     num_tp_perc = mutrixdividebyvector(num_tp, num_gt_act);
%     num_tp_perc_all = [num_tp_header;[num_tp_index, num_tp_perc]];
% 
%     csvwrite(resnum_filepath,num_tp_all);
%     csvwrite(resperc_filepath,num_tp_perc_all);
end 