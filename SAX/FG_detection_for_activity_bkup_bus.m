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

act_ind = 2;

% define folder
subjfolder = '../../inlabStr/subject/Dzung(8Hz)/';
segfolder = [subjfolder, 'segmentation/'];
actfolder = [segfolder, 'activity/', activities{act_ind},'/'];

% define predict result file path
pred_file_head = 'pred_headtail_';
savefile = [pred_file_head, '_', activities{act_ind}, '.csv'];

if ~exist(actfolder, 'dir')
    mkdir(actfolder);
end

savepath = strcat(actfolder, savefile);

% define ground truth file path
gtfile_head = 'gt_headtail_';
gtfile = [gtfile_head, '_', activities{act_ind}, '.csv'];

% pattern file
ptnfolder = [actfolder, '/pattern/'];
PTN_FILE = ['patterns','_struc_Dzung(8Hz)_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];


%% read energy file
energyfile = 'enrg_class_activity_win4_str2.csv';
data = csvread(strcat(subjfolder,energyfile),1);
energy_acc_xyz = data(:,2);
Class = data(:,3);  
activity = data(:,5);


%% define predefined pattern
%% this could be replaced by automatic pattern selection method
%  save patterns by column - vstack convention
enrg_col = energy_acc_xyz(:);


%%  loop for activities
for act_ind = 1:2
    
    if act_ind==1 % spoon
        ptns_cell{1} = timeseries2symbol(enrg_col(954:954+10-1), 10, 10/2, dict_size,1);
        ptns_cell{2} = timeseries2symbol(enrg_col(1336:1336+10-1), 10, 10/2, dict_size,1);
    elseif act_ind==2 % bread
        ptns_cell{1} = timeseries2symbol(enrg_col(1621:1628), 8, 8/2, dict_size,1);
    % ptns_cell{3} = timeseries2symbol(enrg_col(2313:2313+24-1), 24, 24/2, dict_size,1);
    % ptns_cell{4} = timeseries2symbol(enrg_col(5529:5546), 18, 18/2, dict_size,1);
    % ptns_cell{5} = timeseries2symbol(enrg_col(5611:5624), 14, 14/2, dict_size,1);
    % ptns_cell{6} = timeseries2symbol(enrg_col(4840:4845), 6, 6/2, dict_size,1);
    end
    
    %% save patterns
    ptnpath = [ptnfolder,PTN_FILE];
    if ~exist(ptnfolder, 'dir')
        mkdir(ptnfolder);
    end
    save(ptnpath,'ptns_cell');
    
    pw_unit = activity_segmentation(energy_acc_xyz, ptns_cell, dict_size, savepath);
    


%% sort and plot

figure;
subplot(2,1,1);
plot(energy_acc_xyz,'g');
hold on;
plot(pw_unit , 'b');
title('prediction');

subplot(2,1,2);
plot(energy_acc_xyz,'g');
hold on;
plot(Class,'r');
title('ground truth');


%% energy_acc_xyz is test data
%  ptns_cell is predefined data saved in mat file
function pw_unit = activity_segmentation(energy_acc_xyz, ptns_cell, dict_size, savepath)
    
    %  this param is important
    std_thres = 1e-2;
    
    %for each pattern select how many candidates
    topN = 100; 
    
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
            disttmp = min_dist(sd(i,:), ptns_cell{j}, dict_size,1);
            dists = [dists;disttmp];
        end

        % convert distance of substrings with tiny std_dev, that is noise, to
        % inf
        % from [0,1,0,0,1] to [NaN, inf, NaN, NaN, inf]
        factor = noise_fg*inf; 
        factor(isnan(factor)) = 1;
        dists_wo_noise = dists.*factor;

        [sdists, sind] = sort(dists_wo_noise);

        % save searched candidates in terms of head
        sind_topN = sort(sind(1:topN));
        for ii = 1: topN
            pred_list = [pred_list, sind_topN(ii),sind_topN(ii)+N-1];
        end
        pw = pw + head2pointwise(sind_topN, N, length(energy_acc_xyz));
    end
    
    
    csvwrite(savepath, pred_list);
    
    pw_unit = sign(pw);

%     headtail_rpr = pointwise2headtail(pw_unit);
%     csvwrite(strcat(folder,savefile),headtail_rpr);
end


%% internal function
function save_headtail(headtail_rpr, savepath)
    head = headtail_rpr(1:2:end);
    head = head(:);
    tail = headtail_rpr(2:2:end);
    tail = tail(:);
    htd = [head, tail, tail-head];
    disp(htd)
    fid = fopen(savepath,'w');
    fprintf(fid,'%f, %f, %f\n',htd');
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