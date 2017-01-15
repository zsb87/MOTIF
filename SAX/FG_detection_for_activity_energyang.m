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

function FG_detection_for_activity_energyang(subj, std_thres, dist_thres, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!')
    end
    
    if ~exist(act_rootfolder, 'dir')
        mkdir(act_rootfolder);
    end
    
    %% read energy file
    %% check the column number    
    energyfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(strcat(energyfolder,energyfile),1);
    energy_ang_xyz = data(:,7);
    fClass = data(:,8);
    activity = data(:,10);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save all activities' ground truth in one file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gt_headtail = pointwise2headtail(fClass);  
%     gtfile_all = 'gt_headtail_all_activities.csv';
%     save_headtail(gt_headtail, strcat(act_rootfolder,gtfile_all));


    %%  loop for activities
    for act_ind = 1:9
        disp(act_ind);
        
        % define folder
        actfolder = [act_rootfolder, activities{act_ind},'/'];

        % define predict result file path
        pred_file_prefix = 'pred_ang_headtail_';
        predfile = [pred_file_prefix, activities{act_ind}, '.csv'];

        if ~exist(actfolder, 'dir')
            mkdir(actfolder);
        end

        predfilepath = strcat(actfolder, predfile);
        pred_reduce_filepath = strcat(actfolder, [pred_file_prefix, 'reduced_', activities{act_ind}, '.csv']);

        %--------------------------------------------------------
        % define each activity's ground truth file path
        %--------------------------------------------------------
        gtfile_head = 'gt_headtail_';
        gtfile = [gtfile_head, activities{act_ind}, '.csv'];
        gtfilepath = strcat(actfolder, gtfile);

        % pattern file
        ptnfolder = [actfolder, 'pattern/'];
        ptnfile = ['patterns_ang_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
        ptnfilepath = strcat(ptnfolder, ptnfile);
        ptncsvfilepath = strcat(ptnfolder, 'patterns_ang_headtail.csv');


        %% define predefined pattern
        %% this could be replaced by automatic pattern selection method
        %  save patterns by column - vstack convention
        act_feeding = fClass.*activity;

        gt_headtail_act = pointwise2headtail_c(act_feeding, act_ind);

        if isempty(gt_headtail_act)
            disp(['no gesture of ',activities(act_ind)]);
        else
            headtaillen_act = save_headtail(gt_headtail_act, gtfilepath);

            ptns_cell = [];
            disp(act_ind);
            
            [maxlen, maxind] = max(headtaillen_act(:,3));
            [minlen, minind] = min(headtaillen_act(:,3));

            % sort to find the median
            [s_headtaillen_act, ind_headtaillen_act] = sort(headtaillen_act(:,3));

            ptn_headtail_list = [];
            ptcell_ind = 1;

            if mod(maxlen,2) == 0
                    %number is even
                    ptn_headtail_list = [ptn_headtail_list; headtaillen_act(maxind,:)];
                    e_ptn = energy_ang_xyz(headtaillen_act(maxind,1):headtaillen_act(maxind,2));
                    ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, maxlen, maxlen/2, dict_size,1);
                    ptcell_ind = ptcell_ind + 1;
            else 
                    %number is odd
                    ptn_headtail_list = [ptn_headtail_list; headtaillen_act(maxind,:)];
                    e_ptn = energy_ang_xyz(headtaillen_act(maxind,1):headtaillen_act(maxind,2)-1);
                    ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, maxlen-1, (maxlen-1)/2, dict_size,1);
                    ptcell_ind = ptcell_ind + 1;
            end
                
            if isempty(ind_headtaillen_act) == 0
                tmp = floor(length(ind_headtaillen_act)/2);
                if tmp ~= 0
                    medlen = s_headtaillen_act(tmp);
                    medind = ind_headtaillen_act(tmp);
                    if mod(medlen,2) == 0
                            %number is even
                            ptn_headtail_list = [ptn_headtail_list; headtaillen_act(medind,:)];
                            e_ptn = energy_ang_xyz(headtaillen_act(medind,1):headtaillen_act(medind,2));
                            ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, medlen, medlen/2, dict_size,1);
                        
                            ptcell_ind = ptcell_ind + 1;
                    else
                            %number is odd
                            ptn_headtail_list = [ptn_headtail_list; headtaillen_act(medind,:)];
                            e_ptn = energy_ang_xyz(headtaillen_act(medind,1):headtaillen_act(medind,2)-1);
                            ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, medlen-1, (medlen-1)/2, dict_size,1);
                        
                            ptcell_ind = ptcell_ind + 1;
                    end
                end
            end
                

            if minlen>4

                if mod(minlen,2) == 0
                        %number is even
                        ptn_headtail_list = [ptn_headtail_list; headtaillen_act(minind,:)];
                        e_ptn = energy_ang_xyz(headtaillen_act(minind,1):headtaillen_act(minind,2));
                        ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, minlen, minlen/2, dict_size,1);
                else
                        %number is odd
                        ptn_headtail_list = [ptn_headtail_list; headtaillen_act(minind,:)];
                        e_ptn = energy_ang_xyz(headtaillen_act(minind,1):headtaillen_act(minind,2)+1);
                        ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, minlen+1, (minlen+1)/2, dict_size,1);
                end
            end
    

            %% save patterns
            if ~exist(ptnfolder, 'dir')
                mkdir(ptnfolder);
            end
            save(ptnfilepath,'ptns_cell');
            csvwrite(ptncsvfilepath, ptn_headtail_list)
            
            %-------------------------------------------------------------------------------
            % this is the core function for judging a prediction
            pw_unit = finding_segmentation(energy_ang_xyz, std_thres, dist_thres, ptns_cell, dict_size, predfilepath, pred_reduce_filepath);
            %-------------------------------------------------------------------------------

            %% sort and plot
%             figure;
%             subplot(2,1,1);
%             plot(energy_ang_xyz,'g');
%             hold on;
%             plot(pw_unit , 'b');
%             title([activities(act_ind), 'prediction']);
% 
%             subplot(2,1,2);
%             plot(energy_ang_xyz,'g');
%             hold on;
%             plot(fClass,'r');
%             title([activities(act_ind),'ground truth']);
        end
    end

end
