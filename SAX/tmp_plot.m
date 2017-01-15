% clc;
% clear all;
% figure; 
% hold on;
% 
% for n=1:3
% for i=1:10
%     if 10*(n-1) + i<24
%         motif  = csvread(['C:\Users\szh702\Documents\FoodWatch\inlabStr\subject\testDzung\segmentation\engy_run5_pred\dist_all\pred_motif',num2str(10*(n-1) + i),'.csv']);
%         subplot(3,10,10*(n-1) + i), plot(motif)
%     end
% end
% end

clc;
clear all;
figure; 
hold on;

energyfile  = csvread(['C:\Users\szh702\Documents\FoodWatch\inlabStr\subject\Gleb\feature\energy\engy_ori_win4_str2_labeled.csv'],1);
engy = energyfile(:,2);
l = energyfile(:,8);
figure;
plot(engy);
hold on 
plot(l)
