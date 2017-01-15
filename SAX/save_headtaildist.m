%% internal function
function headtaillen = save_headtaildist(headtaildist_rpr, savepath)
    head = headtaildist_rpr(1:4:end);
    head = head(:);
    tail = headtaildist_rpr(2:4:end);
    tail = tail(:);
    dist = headtaildist_rpr(4:4:end);
    dist = dist(:);
    len = headtaildist_rpr(3:4:end);
    len = len(:);
    headtaillen = [head, tail, len, dist];
    fid = fopen(savepath,'w');
    fprintf(fid,'%f, %f, %f, %f\n',headtaillen');
    fclose(fid);
end
