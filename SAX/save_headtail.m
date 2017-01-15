%% internal function
function headtaillen = save_headtail(headtail_rpr, savepath)
    head = headtail_rpr(1:2:end);
    head = head(:);
    tail = headtail_rpr(2:2:end);
    tail = tail(:);
    headtaillen = [head, tail, tail-head+1];
    fid = fopen(savepath,'w');
    fprintf(fid,'%f, %f, %f\n',headtaillen');
    fclose(fid);
end
