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

function FG_plot_seg(subj)

    win = 4;
    stride = 2;
    dict_size = 10;

    activities = { 
        'Spoon';            %1
        'HandBread';        %2
        'Cup';              %3
        'Chopstick';        %4
        'KnifeFork';        %5
        'Bottle';           %6
        'SaladFork';        %7
        'HandChips';        %8
        'Straw';            %9
        'SmokeMiddle';      %10
        'SmokeThumb';       %11
        'ChinRest';         %12
        'Phone';            %13
        'Mirror';           %14
        'Scratches';        %15
        'Nose';             %16
        'Teeth';            %17
        };

    % define folder
    folder = '../../inlabStr/subject/';
    subjectname = [subj,'(8Hz)'];
    subjfolder = [folder, subjectname,'/'];
    featfolder = [subjfolder,'feature/'];

    %% read energy file
    %% check the column number
    energyfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
    data = csvread(strcat(featfolder,energyfile),1);
    energy_acc_xyz = data(:,2);
    Class = data(:,7);  
    activity = data(:,9);


    % save ground truth for all activities in one file
    gt_headtail = pointwise2headtail(Class);  
    gtfile_all = 'gt_headtail_all_activities.csv';
    save_headtail(gt_headtail, strcat(subjfolder,gtfile_all));


    %%  loop for activities
    for act_ind = 1:9
        
        % define folder
        segfolder = [subjfolder, 'segmentation/'];
        act_dir = [segfolder, 'activity/'];
        actfolder = [act_dir, activities{act_ind},'/'];

        % define predict result file path
        pred_file_prefix = 'pred_headtail_';
        predfile = [pred_file_prefix, activities{act_ind}, '.csv'];

        predfilepath = strcat(actfolder, predfile);

        % define ground truth file path
        gtfile_head = 'gt_headtail_';
        gtfile = [gtfile_head, activities{act_ind}, '.csv'];
        gtfilepath = strcat(actfolder, gtfile);

        % pattern file
        ptnfolder = [actfolder, 'pattern/'];
%         ptnfile = ['patterns_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
%         ptnfilepath = strcat(ptnfolder, ptnfile);
        ptncsvfilepath = strcat(ptnfolder, 'patterns_headtail.csv');

        act_feeding = Class.*activity;
        gt_headtail_act = pointwise2headtail_c(act_feeding, act_ind);
        
        if isempty(gt_headtail_act)
            disp(['no gesture of ',activities(act_ind)]);
        else
%             ptns_cell = load(ptnfilepath,'ptns_cell');
%             ptns_cell = ptns_cell.ptns_cell;

            ptn_headtail_list = csvread(ptncsvfilepath);
            ptn_headtail_list = ptn_headtail_list(:,1:2)';
            ptn_headtail_list = ptn_headtail_list(:);

            pred_seg = csvread(predfilepath);
            pred_headtail = pred_seg(:,1:2)';
            pred_headtail = pred_headtail(:);
            
            pw = headtail2pointwise(pred_headtail, length(energy_acc_xyz));
            
            %% sort and plot
            figure;
            subplot(3,1,1);
            plot(energy_acc_xyz,'g');
            hold on;
            plot(pw , 'b');
            title([activities(act_ind), 'prediction']);

            subplot(3,1,2);
            plot(energy_acc_xyz,'g');
            hold on;
            plot(headtail2pointwise(gt_headtail_act, length(energy_acc_xyz)),'r');
            title([activities(act_ind),'ground truth']);
            
            
            subplot(3,1,3);
            plot(energy_acc_xyz,'g');
            hold on;
            plot(headtail2pointwise(ptn_headtail_list, length(energy_acc_xyz)),'r');
            title([activities(act_ind),'ground truth']);
        end
    end
end


%% energy_acc_xyz is test data
%  ptns_cell is predefined data saved in mat file
function pw_unit = activity_segmentation(energy_acc_xyz, ptns_cell, dict_size, predfilepath)
    
    %  this param is important
    std_thres = 0;%1e-2
    
    %for each pattern select how many candidates
%     topN = 100; 
    
    % convert to symbolic representation
    % save in cell
    pred_list = [];
    
    for i = 1:size(ptns_cell,2)
        % symbolic pattern length
        n = size(ptns_cell{i},2);
        
        [sd,b,std_dev] = timeseries2symbol(energy_acc_xyz, n*2, n, dict_size,1);
        sd_cell{i} = sd;
        std_cell{i} = std_dev;
    end


    %% find similar patterns to predefined patterns 
    %  pw means pointwise, in contrary to head-tail representation
    pw = zeros([1,length(energy_acc_xyz)]);

    for j = 1:size(ptns_cell,2) 
        % raw signal pattern length 
        n = size(ptns_cell{j},2);
        % symbolic pattern length
        N = n*2;

        dists = [];
        sd = sd_cell{j};
        std_dev = std_cell{j};
    %     noise_fg indicate this substring is noise if 1
        noise_fg = std_dev*0;
        noise_fg(find(std_dev<std_thres)) = 1;

        % calculate similarity with predefined pattern
        for i = 1: length(sd)
            disttmp = min_dist(sd(i,:), ptns_cell{j}(1,:), dict_size,1);
            dists = [dists;disttmp];
        end

        % convert distance of substrings with tiny std_dev, that is noise, to
        % inf
        % from [0,1,0,0,1] to [NaN, inf, NaN, NaN, inf]
        factor = noise_fg*inf; 
        factor(isnan(factor)) = 1;
        dists_wo_noise = dists.*factor;

        [sdists, sind] = sort(dists_wo_noise);

        sind_selected = sind(find(sdists<0.5));
        
        for ii = 1:length(sind_selected)
            pred_list = [pred_list, sind_selected(ii),sind_selected(ii)+N-1];
        end
        
        pw = pw + head2pointwise(sind_selected, N, length(energy_acc_xyz));
    end
    
    save_headtail(pred_list, predfilepath);
    
    pw_unit = sign(pw);

end


%% internal function
function headtaillen = save_headtail(headtail_rpr, savepath)
    head = headtail_rpr(1:2:end);
    head = head(:);
    tail = headtail_rpr(2:2:end);
    tail = tail(:);
    headtaillen = [head, tail, tail-head+1];
    fid = fopen(savepath,'w');
    fprintf(fid,'%f, %f, %f\n',headtaillen');
    fclose(fid);
end



%% internal function 
function headtail_rpr = pointwise2headtail(pointwise_rpr)
    pw = pointwise_rpr(:)';
    diff = pw-[0,pw(1:end-1)];
    ind_head = find(diff == 1);
    ind_tail = find(diff == -1)-1;
    headtail_rpr = [];
    for i = 1:length(ind_head)
        headtail_rpr = [headtail_rpr, ind_head(i), ind_tail(i)];
    end
end


%% internal function
function headtail_rpr = pointwise2headtail_c(pointwise_rpr, c)
    pw = pointwise_rpr(:)';
    diff = pw-[0,pw(1:end-1)];
    ind_head = find(diff == c);
    ind_tail = find(diff == -c)-1;
    headtail_rpr = [];
    for i = 1:length(ind_head)
        headtail_rpr = [headtail_rpr, ind_head(i), ind_tail(i)];
    end
end

%% internal function 
function pointwise_rpr = headtail2pointwise(headtail, length)
    pointwise_rpr = zeros(1, length);
    head = headtail(1:2:end);
    head = head(:);
    
    tail = headtail(2:2:end);
    tail = tail(:);
    for i = 1:size(head,1)
        pointwise_rpr(head(i):tail(i)) = 1;
    end
end


%% internal function 
function pointwise_rpr = head2pointwise(head_list, ptnN, length)
    pointwise_rpr = zeros(1, length);
    for i = 0:ptnN-1
        pointwise_rpr(head_list+i) = 1;
    end
end

function recall = measure(gt_headtail,pred_headtail)
    gt_detected = zeros(1, len(gt_headtail));
    for i = 1:len(gt_headtail)
        
    end
end
