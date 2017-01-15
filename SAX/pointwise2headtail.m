%% internal function 
function headtail_rpr = pointwise2headtail(pointwise_rpr)
    pw = pointwise_rpr(:)';
    diff = [pw,0]-[0,pw(1:end)];
    ind_head = find(diff == 1);
    ind_tail = find(diff == -1)-1;
    headtail_rpr = [];
    for i = 1:length(ind_head)
        headtail_rpr = [headtail_rpr, ind_head(i), ind_tail(i)];
    end
end