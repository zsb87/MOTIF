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
%
% Output:
% 
% save file:
%   
% 

function [num_gt, num_detect, recall] = FG_seg_pred_trueOrFalse_accang(subj,config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!')
    end

    if ~exist(act_rootfolder, 'dir')
        mkdir(act_rootfolder);
    end

    %% read ground truth for all activities in one file
    gtfile_all = 'gt_headtail_all_activities.csv';
    disp(act_rootfolder);
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

            gtfilepath = strcat(act_rootfolder, activities{act_ind},'/', gtfile);
            
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
    disp(segfolder);
    predpath = [segfolder, 'activity/pred_headtail_reduced_all.csv'];
    pred_all = csvread(predpath);
    pred_all = pred_all(:,1:2);    pred_all = pred_all';    pred_all = pred_all(:);
    

    %% count the true positive and number of gt
    gt_all = gt_all(:,1:2);    gt_all = gt_all';    gt_all = gt_all(:);
    
    num_detect = length(pred_all)/2;
    
    %% --------------------------------------------------------------------
    %  core function
    %  --------------------------------------------------------------------
    [num_gt, gt_detected_flg, pred_true_flg] = seg_pred_trueOrFalse(pred_all, gt_all);
    csvwrite([segfolder, 'activity/pred_accang_label.csv'],pred_true_flg);
    
    recall = sum(gt_detected_flg)/length(gt_detected_flg);
    
end 



