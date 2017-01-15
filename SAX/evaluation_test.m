clear all;
close all;
clc;

pred = [0,2;1,3;5,6]
gt = [1,3; 1,2; 2,4]

% precision, recall, fscore = eventBasedEvaluate(pred, gt, 0.5,true)

[precision, recall, fscore] = eventBasedEvaluate(pred, gt, 0.5, true)