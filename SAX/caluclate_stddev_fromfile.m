clear;
subjs = {'Shibo','Dzung',  'JC', 'Cao','Jiapeng','Eric','Rawan'}; %
%problem subject: 'Matt','Will','Gleb' data missing
protocol =  'inlabStr';%'inlabUnstr';
motif_sel_mode = 3;
meas_thres = 0.9;
dist_thres= 0.7;
result = [];
recall_thr5 = [];
recall_thr6 = [];
recall_thr7 = [];
recall_thr8 = [];
recall_thr9 = [];

for i = 1:size(subjs,2)
    train_subj = ['train',subjs{i}];
    test_subj = ['test',subjs{i}];
    folder = ['../../',protocol,'/result/segmentation/'];
    if ~exist(folder,'dir')     mkdir(folder),    end    
    
    resultfile_all = ['engy_run2_result_',test_subj,'_Msel',int2str(motif_sel_mode),'_thre',num2str(dist_thres),'_meas',num2str(meas_thres),'.csv'];
    resultfile_allpath = [folder, resultfile_all];
    result{i} = csvread(resultfile_allpath);
    disp(i)
    
    recall_thr5 = [recall_thr5, result{i}(1,5)];
    recall_thr6 = [recall_thr6, result{i}(2,5)];
    recall_thr7 = [recall_thr7, result{i}(3,5)];
    recall_thr8 = [recall_thr8, result{i}(4,5)];
    recall_thr9 = [recall_thr9, result{i}(5,5)];
    
end

mean(recall_thr5)
std(recall_thr5)
mean(recall_thr6)
std(recall_thr6)
mean(recall_thr7)
std(recall_thr7)
mean(recall_thr8)
std(recall_thr8)
mean(recall_thr9)
std(recall_thr9)