clear all;
close all;
clc;

segs = dlmread('pred_headtail_all.csv');

threshold = 0.7;
N = size(segs,1);

marked = zeros(N,1);

groups = {};
for i = 1:N - 1
    if marked(i) == 1
        continue
    end
    interval1 = segs(i,1:2);
    group = interval1;
    for j = i+1:N
        interval2 = segs(j,1:2);
        if interval2(1) > interval1(2)
            break;
        end
        if overlap(interval1, interval2) > threshold
             group = [group; interval2];
             marked(j) = 1;
        end
    end
    groups{end + 1} = group;
end

size(groups)

