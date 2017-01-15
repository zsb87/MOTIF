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
function FG_seg_measurement(subj)

    win = 4;
    stride = 2;
    dict_size = 10;
    if subj == 'Rawan'
        activities = { 
        'Spoon';            %1
        'HandBread';        %2
        'Chopstick';        %3
        'KnifeFork';        %4
        'SaladFork';        %5
        'HandChips';        %6
        'Cup';              %7
        'Straw';            %8
        'Phone';            %9
        'SmokeMiddle';      %10
        'SmokeThumb';       %11
        'Bottle';           %12
        'Nose';             %13
        'ChinRest';         %14
        'Scratches';        %15
        'Mirror';           %16
        'Teeth';            %17
        };
    else
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
    end
    
    % define folder
    folder = '../../inlabStr/subject/';
    subjectname = strcat(subj, '(8Hz)');
    subjfolder = [folder, subjectname,'/'];
    segfolder = [subjfolder, 'segmentation/'];
    resfolder = [segfolder, 'result/'];

    %% save patterns
    if ~exist(resfolder, 'dir')
        mkdir(resfolder);
    end

    % define result save folder
    resnumfile = 'num_truepositive.csv';
    resnum_filepath = [resfolder, resnumfile];
    respercfile = 'perc_truepositive.csv';
    resperc_filepath = [resfolder, respercfile];

    % save ground truth for all activities in one file
    gtfile_all = 'gt_headtail_all_activities.csv';
    

    %  --------------------------------------------------------------------
    %  read all gt and pred results in cell
    %  --------------------------------------------------------------------
    for act_ind = 1:9

        % define folder
        actfolder = [segfolder, 'activity/', activities{act_ind},'/'];

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
        ptnfile = ['patterns_struc_',subjectname,'_win', num2str(win), '_str', num2str(stride), activities{act_ind},'.mat'];
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


            gt_act = csvread(gtfilepath);
            gt_act = gt_act(:,1:2);
            gt_act = gt_act';
            gt_cell{act_ind} = gt_act(:);
            num_gt_act(act_ind) = length(gt_cell{act_ind})/2;        

        end


    end




    %% count the true positive and number of gt
    num_tp = zeros(9);

    % based on the (ref_ind)th activity patterns
    for ref_ind = 1:9
        % find similar patterns that actually belong to (test_ind)th activity
        for test_ind = 1:9

            pred = pred_cell{ref_ind};
            gt = gt_cell{test_ind};
            [true_positive, n, detection_flg] = seg_measurement(pred, gt);

            num_tp(ref_ind,test_ind) = true_positive;

        end
    end

    num_tp_header = [0, num_gt_act];

    num_tp_index = num_ptns';

    num_tp_all = [num_tp_header;[num_tp_index, num_tp]];

    num_tp_perc = mutrixdividebyvector(num_tp, num_gt_act);

    num_tp_perc_all = [num_tp_header;[num_tp_index, num_tp_perc]];

    csvwrite(resnum_filepath,num_tp_all);
    csvwrite(resperc_filepath,num_tp_perc_all);
end 