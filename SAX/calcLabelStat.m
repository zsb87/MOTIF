function [num, lenMean, lenDev] = calcLabelStat(gt_headtail)
    
    num = size(gt_headtail,1);
    len = gt_headtail(:,3);
    lenMean = mean(len);
    lenDev = std(len);
    
end
