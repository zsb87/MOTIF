
%% inner function:
%     A=[ 0     2     0     0     0
%         0     1     1     0     1
%         0     1     0     0     2
%         0     0     0     0     1
%         0     0     0     0     0]
% 
%     B=[0 4 1 0 4];
% 
%     C=[0   0.5   0   0  0
%       0   0.25  1   0  0.25
%       0   0.25  0   0  0.5
%       0   0     0   0  0.25 
%       0   0     0   0  0   ]

function C = mutrixdividebyvector(A,b)
    [rows, columns] = size(A);
    % Get sum of columns and replicate vertically.
    denom = b;
    % Do the division.
    C = A ./ denom;
    % Set infinities (where denom == 0) to 0
    C(denom==0) = 0;
end