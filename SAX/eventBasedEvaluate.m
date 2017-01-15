function [gt_retrieved, pred_true, precision, recall, fscore, false_negative] = eventBasedEvaluate(pred, gt, threshold, verbose)
% pred and gt is matrix of Nx2 and Mx2. 
% Each row is a pair of interval [start, end]
% The starting point start must be increasing

if nargin < 4
    verbose = false;
end

if nargin < 3
    threshold = 0.5;
end

assert(size(pred,2) == 2 && size(gt,2) == 2, 'Second dimension must be 2');
assert(issorted(pred(:,1)) && issorted(gt(:,1)), 'Start time need to be sorted');

pred_true = zeros(size(pred,1),1);
gt_retrieved = zeros(size(gt,1), 1);

for i = 1:size(gt,1)
    for j = 1:size(pred,1)
        if pred(j,2) < gt(i,1)
            continue
        end
        if pred(j,1) > gt(i,2)
            break
        end
        overlap = min(pred(j,2), gt(i,2)) - max(pred(j,1), gt(i,1)) + 1;
       
        if (overlap/(max(pred(j,2), gt(i,2)) - min(pred(j,1), gt(i,1)) + 1) >= threshold)
            if verbose
                fprintf('[%d, %d] intersects with [%d, %d]\n', gt(i,1), gt(i,2), pred(j,1), pred(j,2)); 
            end
            pred_true(j) = 1;
            gt_retrieved(i) = 1;
        end
    end
end

true_positive = sum(pred_true);
false_negative = size(gt,1) - sum(gt_retrieved);

precision = true_positive / size(pred,1);
recall = sum(gt_retrieved) / size(gt,1);

if (precision + recall) > 0
    fscore = 2*precision*recall/(precision + recall);
else
    fscore = 0;
end
end