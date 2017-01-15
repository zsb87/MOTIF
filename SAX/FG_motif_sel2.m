% --------------------------------------------------------------------
%
%   Unstructured
%
% --------------------------------------------------------------------

function [motif_SAX_cell] = FG_motif_sel2( train_sig_cell, train_gt_htcell, config_file, motif_sel_mode)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!_motif_sel')
    end
        
    motif_SAX_cell = [];

    if motif_sel_mode == 1
    %----------------------------------------------------------------------------------
    %  pick up the min-med-max motifs from all feeding gestures
    %----------------------------------------------------------------------------------
%     for act_ind = 1:size(feeding,2)
%         
%         energyfolder = [folder, feeding{act_ind},'/',subj,'(8Hz)/'];
%         disp(energyfolder)
%         
%         %% read energy file
%         %% check the column number
%         energyfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_10gestures.csv'];
%         labelfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_10gestures_label.csv'];
%         data = csvread(strcat(energyfolder,energyfile),1);
%         energy_acc_xyz = data(:,2);
%         label = csvread(strcat(energyfolder,labelfile),1);
%         fClass = label(:,1);
%         
%         segfolder = [energyfolder, 'segmentation/'];
%         act_rootfolder = [segfolder, 'motif_activity/'];
%         if ~exist(act_rootfolder, 'dir')
%             mkdir(act_rootfolder);
%         end
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % save all activities' ground truth in one file
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         gt_headtail = pointwise2headtail(fClass);
%         save_headtail(gt_headtail, strcat(act_rootfolder,'gt_feeding_headtail.csv'));
%         
%         %--------------------------------------------------------
%         % define folder
%         %--------------------------------------------------------
%         actfolder = [act_rootfolder, feeding{act_ind},'/'];
%         if ~exist(actfolder, 'dir')
%             mkdir(actfolder);
%         end
%         
%         %--------------------------------------------------------
%         % define each activity's ground truth file path
%         %--------------------------------------------------------
%         gtfile = ['gt_headtail_', feeding{act_ind}, '.csv'];
%         gtfilepath = strcat(actfolder, gtfile);
% 
%         % pattern file
%         ptnfolder = [actfolder, 'pattern/'];
%         ptnfile = ['patterns_acc_struc_',subj,'_win', num2str(win), '_str', num2str(stride), feeding{act_ind},'.mat'];
%         ptnfilepath = strcat(ptnfolder, ptnfile);
%         ptncsvfilepath = strcat(ptnfolder, 'patterns_acc_headtail.csv');
% 
% 
%         %% define predefined pattern
%         %% this could be replaced by automatic pattern selection method
%         %  save patterns by column - vstack convention
%         gt_headtail_act = pointwise2headtail(fClass);
% 
%         if isempty(gt_headtail_act)
%             disp(['no gesture of ',feeding(act_ind)]);
%         else
%             headtaillen_act = save_headtail(gt_headtail_act, gtfilepath);
%             
% %             disp(act_ind);
%             [maxlen, maxind] = max(headtaillen_act(4:end,3));
%             [minlen, minind] = min(headtaillen_act(4:end,3));
% 
%             % sort to find the median
%             [s_headtaillen_act, ind_headtaillen_act] = sort(headtaillen_act(4:end,3));
%             ptn_headtail_list = [];
% 
%             if mod(maxlen,2) == 0
%                     %number is even
%                     ptn_headtail_list = [ptn_headtail_list; headtaillen_act(maxind,:)];
%                     e_ptn = energy_acc_xyz(headtaillen_act(maxind,1):headtaillen_act(maxind,2));
%                     ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, maxlen, (maxlen/2), dict_size,1);
%                     ptcell_ind = ptcell_ind + 1;
%             else 
%                     %number is odd
%                     ptn_headtail_list = [ptn_headtail_list; headtaillen_act(maxind,:)];
%                     e_ptn = energy_acc_xyz(headtaillen_act(maxind,1):headtaillen_act(maxind,2)-1);
%                     ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, maxlen-1, (maxlen-1)/2, dict_size,1);
%                     ptcell_ind = ptcell_ind + 1;
%             end
%                 
%             if isempty(ind_headtaillen_act) == 0
%                 tmp = floor(length(ind_headtaillen_act)/2);
%                 if tmp ~= 0
%                     medlen = s_headtaillen_act(tmp);
%                     medind = ind_headtaillen_act(tmp);
%                     if mod(medlen,2) == 0
%                             %number is even
%                             ptn_headtail_list = [ptn_headtail_list; headtaillen_act(medind,:)];
%                             e_ptn = energy_acc_xyz(headtaillen_act(medind,1):headtaillen_act(medind,2));
%                             ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, medlen, (medlen/2), dict_size,1);
%                         
%                             ptcell_ind = ptcell_ind + 1;
%                     else
%                             %number is odd
%                             ptn_headtail_list = [ptn_headtail_list; headtaillen_act(medind,:)];
%                             e_ptn = energy_acc_xyz(headtaillen_act(medind,1):headtaillen_act(medind,2)-1);
%                             ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, medlen-1, (medlen-1)/2, dict_size,1);
%                         
%                             ptcell_ind = ptcell_ind + 1;
%                     end
%                 end
%             end
%                 
% 
%             if minlen>4
% 
%                 if mod(minlen,2) == 0
%                         %number is even
%                         ptn_headtail_list = [ptn_headtail_list; headtaillen_act(minind,:)];
%                         e_ptn = energy_acc_xyz(headtaillen_act(minind,1):headtaillen_act(minind,2));
%                         ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, minlen,(minlen/2), dict_size,1);
%                         ptcell_ind = ptcell_ind + 1;
%                 else
%                         %number is odd
%                         ptn_headtail_list = [ptn_headtail_list; headtaillen_act(minind,:)];
%                         e_ptn = energy_acc_xyz(headtaillen_act(minind,1):headtaillen_act(minind,2)+1);
%                         ptns_cell{ptcell_ind} = timeseries2symbol(e_ptn, minlen+1, (minlen+1)/2, dict_size,1);
%                         ptcell_ind = ptcell_ind + 1;
%                 end
%             end
% 
%             %% save patterns
%             if ~exist(ptnfolder, 'dir')
%                 mkdir(ptnfolder);
%             end
%             save(ptnfilepath,'ptns_cell');
%             csvwrite(ptncsvfilepath, ptn_headtail_list)
%             
%         end
%     end
    end
    
    if motif_sel_mode == 3
        % format-  engy_test_cell: {engy_xyz(1), engy_xyz(2), ... ,}
            
        nn=4;
        [motif_SAX_cell] = FG_KcentriodSC_27(train_sig_cell, train_gt_htcell, nn, dict_size);
    end
    
            %% sort and plot
%             figure;
%             subplot(2,1,1);
%             plot(energy_acc_xyz,'g');
%             hold on;
%             plot(pw_unit , 'b');
%             title([activities(act_ind), 'prediction']);
% 
%             subplot(2,1,2);
%             plot(energy_acc_xyz,'g');
%             hold on;
%             plot(fClass,'r');
%             title([activities(act_ind),'ground truth']);
        

end
