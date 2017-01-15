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

function [pred_AccAng, num_det_acc, num_det_ang, num_detect_accang] = FG_remove_repeatedSeg(subj, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!')
    end
    
    
    %% --------------------------------------------------------------------
    %  read all acc based pred results in cell
    %  --------------------------------------------------------------------
    
    pred_act_all = [];
    
    for act_ind = 1:9

        % define folder
        actfolder = [segfolder, 'activity/', activities{act_ind},'/'];

        % define predict result file path
        pred_file_prefix = 'pred_acc_headtail_';
        predfile = [pred_file_prefix, 'reduced_', activities{act_ind}, '.csv'];
        predfilepath = strcat(actfolder, predfile);
        
        % pattern file
        ptnfolder = [actfolder, 'pattern/'];
        ptnfile = ['patterns_acc_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
        ptnfilepath = strcat(ptnfolder, ptnfile);

        % return the number of patterns/motifs for each activity
        if ~exist(ptnfilepath, 'file')
        else
            %=========================================================================
            % read prediction headtail file
            %=========================================================================
            pred_act_headtaildist = csvread(predfilepath);
            pred_act_all = [pred_act_all; pred_act_headtaildist];
        end
    end
    pred_actAccAll = sortrows(pred_act_all);
    pred_actAccAllSorted = pred_actAccAll(:,1:2);
    [~ ,tmp_ind]= unique(pred_actAccAllSorted,'rows');
    pred_actAccAllIdentical = pred_actAccAll(tmp_ind,:);
    
    
    
    
    %% --------------------------------------------------------------------
    %  read all ang based pred results in cell
    %  --------------------------------------------------------------------
    
    pred_act_all = [];
    
    for act_ind = 1:9

        % define folder
        actfolder = [segfolder, 'activity/', activities{act_ind},'/'];

        % define predict result file path
        pred_file_prefix = 'pred_ang_headtail_';
        predfile = [pred_file_prefix, 'reduced_', activities{act_ind}, '.csv'];
        predfilepath = strcat(actfolder, predfile);
        
        % pattern file
        ptnfolder = [actfolder, 'pattern/'];
        ptnfile = ['patterns_ang_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
        ptnfilepath = strcat(ptnfolder, ptnfile);

        % return the number of patterns/motifs for each activity
        if ~exist(ptnfilepath, 'file')
        else
            
            %=========================================================================
            % read prediction headtail file
            %=========================================================================
            pred_act = csvread(predfilepath);
            pred_act = pred_act(:,1:2);
            pred_act_all = [pred_act_all; pred_act];
        end
    end
    pred_actAngAllSorted = sortrows(pred_act_all);
    pred_actAngAllIdentical = unique(pred_actAngAllSorted,'rows');
    
    pred_actAccHTIdentical = pred_actAccAllIdentical(:,1:2);
    [pred_AccAng, pred_acc_ind] = intersect(pred_actAccHTIdentical, pred_actAngAllIdentical,'rows');
    
    num_det_acc = size(pred_actAccAllIdentical,1);
    num_det_ang = size(pred_actAngAllIdentical,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save all activities' ground truth in one file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    num_detect_accang = size(pred_AccAng,1);
   

    savepath = [segfolder, 'activity/pred_headtail_all.csv'];
    pred_AccAng = pred_actAccAllIdentical(pred_acc_ind,:);
    pred_AccAng = pred_AccAng';
    pred_AccAng = pred_AccAng(:);
    headtaillendist = save_headtaildist(pred_AccAng, savepath);
    
    savepath = [segfolder, 'activity/pred_headtail_reduced_all.csv'];
    
    headtaillendist = grouping(headtaillendist);
    
    csvwrite(savepath, headtaillendist)
end



   