clc;
clear all;
figure; 
hold on;

for n=1:3
for i=1:10
    if 10*(n-1) + i<24
        disp(['C:\Users\szh702\Documents\FoodWatch\inlabStr\subject\testDzung\segmentation\engy_run2_pred\dist_all\pred_motif',num2str(10*(n-1) + i),'.csv']);
        motif  = csvread(['C:\Users\szh702\Documents\FoodWatch\inlabStr\subject\testDzung\segmentation\engy_run2_pred\dist_all\pred_motif',num2str(10*(n-1) + i),'.csv']);
        subplot(3,10,10*(n-1) + i), plot(motif)
    end
end
end


