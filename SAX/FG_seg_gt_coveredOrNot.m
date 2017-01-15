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


function [num_gt, num_segments_all, recall, precision] = FG_seg_gt_coveredOrNot(subj, config_file)
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
    %  read all activities'   GT  in a matrix
    %  --------------------------------------------------------------------
    if ~exist(gtfile_allpath, 'file')
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
        %=========================================================================
        % read ground truth headtail file
        %=========================================================================
        gt_all = csvread(gtfile_allpath);
    end
    

    %% --------------------------------------------------------------------
    %  read all pred results in cell
    %  --------------------------------------------------------------------
    for act_ind = 1:9

        % define folder
        actfolder = [segfolder, 'activity/', activities{act_ind},'/'];

        % define predict result file path
        pred_file_prefix = 'pred_headtail_';
        predfile = [pred_file_prefix, activities{act_ind}, '.csv'];
        predfilepath = strcat(actfolder, predfile);
        
        % pattern file
        ptnfolder = [actfolder, 'pattern/'];
        ptnfile = ['patterns_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
        ptnfilepath = strcat(ptnfolder, ptnfile);

        % return the number of patterns/motifs for each activity
        if ~exist(ptnfilepath, 'file')
            num_gt_act(act_ind) = 0;
        else
            import_struct = load(ptnfilepath);
            num_ptns(act_ind) = length(import_struct.ptns_cell);
            
            %=========================================================================
            % read prediction headtail file
            %=========================================================================
            pred_act = csvread(predfilepath);
            pred_act = pred_act(:,1:2);
            pred_act = pred_act';
            pred_cell{act_ind} = pred_act(:);
        end
    end


    %% count the true positive and number of gt
    
    num_tp = zeros(9);
    gt_all = gt_all(:,1:2);
    gt_all = gt_all';
    gt_all = gt_all(:);
            
    detection_flg_all = zeros(length(gt_all)/2,1);
    
    num_segments_all = 0;

    % based on the (ref_ind)th activity patterns
    for ref_ind = 1:9
        % find similar patterns that actually belong to (test_ind)th activity
        pred = pred_cell{ref_ind};
        
        num_segments_all = num_segments_all + length(pred)/2;
        
%         disp(num_segments_all);
        
        for test_ind = 1:9
            %-------------------------------------------------------------------------------
            % this is the core function for judging a prediction
            % detection_flg:
            % true_flg: 
            
            [true_positive, num_gt, detection_flg, true_flg] = seg_gt_coveredOrNot(pred, gt_all);
            %-------------------------------------------------------------------------------
            
            detection_flg_all = detection_flg_all + detection_flg;
            
            num_tp(ref_ind,test_ind) = true_positive;
        end
    end
    
    detection_flg_all = sign(detection_flg_all);
    recall = sum(detection_flg_all)/length(detection_flg_all);
    precision = sum(detection_flg_all)/num_segments_all;
    
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