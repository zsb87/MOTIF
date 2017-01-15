

%% internal function 
function pointwise_rpr = head2pointwise(head_list, ptnN, length)
    pointwise_rpr = zeros(1, length);
    for i = 0:ptnN-1
        pointwise_rpr(head_list+i) = 1;
    end
end