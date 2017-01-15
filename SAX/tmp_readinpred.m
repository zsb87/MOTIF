subjs = {'Jiapeng'}; %'Dzung','Cao', 'Jiapeng','JC','Shibo','Matt'
protocol = 'inlabUnstr'; %'inlabUnstr';

for i =1:size(subjs,2)
    test_subj = strcat('test',subjs{i});

    run = 11184;

    folder = strcat('../../',protocol,'/subject/',test_subj,'/segmentation/engy_run',num2str(run),'_pred_label_thre0.5');
    disp(strcat(folder,'/seg_labels.csv'));
    labels = csvread(strcat(folder,'/seg_labels.csv'));
    labels = labels(:,1);

    folder = strcat('../../',protocol,'/subject/',test_subj,'/segmentation/engy_run',num2str(run),'_pred');
    segments = csvread(strcat(folder,'/pred_headtail_reduced_1.csv'));

    pred_ind = find(labels == 1);
    segment_p = segments(pred_ind,:);
    segment_p = sortrows(segment_p);

    pred_f_moments = [];
    for i = 1:size(segment_p,1)
        pred_f_moments = [ pred_f_moments, segment_p(i,1):segment_p(i,2)];
    end
    figure;
    plot(pred_f_moments);
    
    % segment_p_head
    segment_p_h = segment_p(:,1);
    % segment_p_h=unique(segment_p_h,'rows');
    pw = zeros(1,max(segment_p(:,2)));

    for i = 1:size(segment_p,1)
        for j = segment_p(i,1):segment_p(i,2)
            pw(j)= pw(j)+1;
        end
    end

    figure;
    subplot(311);
    plot(pw);
    % disp(segment_p)




    folder = strcat('../../',protocol,'/subject/',test_subj,'(8Hz)/segmentation/engy_gt');
    gt_seg = csvread(strcat(folder,'/gt_feeding_headtail.csv'));
    gt_pw = zeros(1,gt_seg(end,2));
    for i = 1:size(gt_seg,1)
        for j = gt_seg(i,1):gt_seg(i,2) %(i,1)
        gt_pw(j)= gt_pw(j)+1;
        end
    end
    subplot(313);
    plot(gt_pw);


    [C, ptsC, centres] = dbscan(pred_f_moments, 2, 1);
    pw_dbsc = zeros(1,max(segment_p(:,2)));
    pw_dbsc(floor(centres)) = 1;

    subplot(312);
    plot(pw_dbsc);
    disp(size(gt_seg,1));
    disp(size(C,2));
    
end
