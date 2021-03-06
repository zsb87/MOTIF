function [motif_SAX_cell] = FG_KcentriodSC_27(train_sig_cell, train_gt_htcell, nn, dict_size)
        
    motif_SAX_cell = [];
    
    train_HT = [];    
    for i = 1:size(train_gt_htcell,2)
        train_HT = [train_HT; train_gt_htcell{i}];
    end
    
    
%     ------------------------------------------------------------------------------------
%      add function: remove gestures longer than mean absolute deviation of
%      the durations -11/01
%     ------------------------------------------------------------------------------------
    
    disp(strcat('lengths are: ',num2str(train_HT(:,3)')));
    
    meandevthres = mad(train_HT(:,3),0);
    disp(strcat('total number is: ',num2str(size(train_HT,1))));
%     disp(strcat('mean dev threshold is: ',num2str(meandevthres)));
%     disp(strcat('mean is: ',num2str(mean(train_HT(:,3)))));
    
%     rm_ind = find(train_HT(:,3)>meandevthres);
    rm_ind = find(train_HT(:,3)>186);
    train_HT(rm_ind,:) = [];
%     disp(strcat('remove:  ', num2str(length(rm_ind)), ' segments longer than:  ', num2str(meandevthres),' time points'));
    disp(strcat('remove:  ', num2str(length(rm_ind)), ' segments longer than 186 time points'));
    
    
    
    
    X = [];    
    if size(train_HT,1)<61
        train_mfig = figure();
        for c = 1:size(train_HT, 1)
            figure(train_mfig);
            subplot(ceil(size(train_HT, 1)/10), 10, c);
            plot( train_sig_cell{1}(train_HT(c,1):train_HT(c,2)) );
        end
        
    elseif size(train_HT,1)<121
        
        train_mfig = figure();
        for c = 1:60
            figure(train_mfig);
            subplot(6, 10, c);
            plot( train_sig_cell{1}(train_HT(c,1):train_HT(c,2)) );
        end
        train_mfig2 = figure();
        for c = 61:size(train_HT, 1)
            figure(train_mfig2);
            subplot(6, 10, c-60);
            plot( train_sig_cell{1}(train_HT(c,1):train_HT(c,2)) );
        end
    elseif size(train_HT,1)<181
        
        train_mfig = figure();
        for c = 1:60
            figure(train_mfig);
            subplot(6, 10, c);
            plot( train_sig_cell{1}(train_HT(c,1):train_HT(c,2)) );
        end
        train_mfig2 = figure();
        for c = 61:120
            figure(train_mfig2);
            subplot(6, 10, c-60);
            plot( train_sig_cell{1}(train_HT(c,1):train_HT(c,2)) );
        end
        train_mfig3 = figure();
        for c = 121:size(train_HT, 1)
            figure(train_mfig3);
            subplot(6, 10, c-120);
            plot( train_sig_cell{1}(train_HT(c,1):train_HT(c,2)) );
        end
    end
    
       
    for i = 1:size(train_HT,1)
        maxlen = max(train_HT(:,3));
        x{i} =  train_sig_cell{1}(train_HT(i,1):train_HT(i,2));
        X = [X;zeros(1,maxlen - train_HT(i,3)),x{i}'];
    end
        
    [cluster_num, cent] = ksc_toy(X, nn);
    cent(all(cent==0,2),:)=[];
        
        
    mcell_ind = 1;
    
    for i = 1:size(cent,1)
            
        c_i = cent(i,:); % c_i means center_i
        % remove NaN, otherwise error msg when dealing with Shibo
        ind = find(isnan(c_i));
        c_i(ind)=0;    
        c_i_wo0 = c_i(find(any(c_i,1),1,'first'):end);
        
        motif_SAX_cell{mcell_ind} = timeseries2symbol(c_i_wo0, length(c_i_wo0), floor(length(c_i_wo0)/2), dict_size,1);
            
        mcell_ind = mcell_ind + 1;
        
    end
    
    mcell_ind = mcell_ind - 1;
        
    ind_tmp = 1;
        
    for i = 1:mcell_ind
        if all(motif_SAX_cell{i}==0,2)==0            
            ptns_cell_tmp{ind_tmp} = motif_SAX_cell{i};
            ind_tmp = ind_tmp + 1;
        end
    end
    
    motif_SAX_cell = ptns_cell_tmp;
    
end
