%% internal function
function headtail_rpr = pointwise2headtail_c(pointwise_rpr, c)
    pw = pointwise_rpr(:)';
    diff = pw-[0,pw(1:end-1)];
    ind_head = find(diff == c);
    ind_tail = find(diff == -c)-1;
    headtail_rpr = [];
    for i = 1:length(ind_head)
        headtail_rpr = [headtail_rpr, ind_head(i), ind_tail(i)];
    end
end
