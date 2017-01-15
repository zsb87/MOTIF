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

function [true_positive, num_gt, detection_flg, true_flg] = seg_gt_coveredOrNot(pred, gt)
    
    detection_flg = zeros(length(gt)/2,1);
    true_flg = zeros(length(pred)/2,1);
    
	true_positive = 0;

	if mod(length(pred),2) == 1
		disp('input pred not pairs');
        return;
    end

	if mod(length(gt),2) == 1
		disp('input gt not pairs');
        return;
    end

	for i = 1:length(gt)/2
        gt_found_times = 0;
		for j = 1:length(pred)/2
			overlap = 0;

			g0 = gt(2*i-1);
			g1 = gt(2*i);
			p0 = pred(2*j-1);
			p1 = pred(2*j);

			% first judge if pred's one point(head or tail) comes in the middle 
			% and then the equation will automatically judge the other one's position

			if (g1 - p1)*(g0 - p1) < 0 
				overlap = (p1 - g0) - ((sign(p0-g0))/2+0.5)*abs(p0-g0);
            elseif (g1 - p0)*(g0 - p0) < 0 
				overlap = (p1 - g0) - ((sign(p0-g0))/2+0.5)*abs(p0-g0);
            elseif (p1 > g1)
                if (p0 < g0) 
                    overlap = g1 - g0;
                end
            else
				overlap = 0;
            end
            
			if overlap > (g1-g0)/2
				gt_found_times = gt_found_times+1;
            end
            
        end
        
        if gt_found_times > 0
            true_positive = true_positive + 1;
            detection_flg(i) = 1;
        end
        
    end
        
    num_gt = length(gt)/2;
%     recall = true_positive/(length(gt)/2);
end
