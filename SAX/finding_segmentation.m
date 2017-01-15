%% energy_acc_xyz is test data
%  ptns_cell is predefined data saved in mat file


function [test_pred_ht, num_pred, pw] = finding_segmentation(motif_cell, test_sig, std_thres, dist_thres, dict_size, predfilepath, pred_reduce_filepath,dist_folder)
    % convert training set to symbolic representation and save in cell
    pred_list = [];
    num_pred = 0;
    % for this test, set remove longer patterns, keep valid patterns
    motif_valid_list = [];
    
    test_SAX_ind = 1;
    test_SAX_for_motifscell= [];
    
    for i = 1:size(motif_cell, 2)
        % symbolic pattern length
        n = size(motif_cell{i}, 2);
        
        if length(test_sig) > 2*n
            [test_SAX, ~, std_dev] = timeseries2symbol(test_sig, n*2, n, dict_size,1);
            test_SAX_for_motifscell{test_SAX_ind} = test_SAX;  std_cell{test_SAX_ind} = std_dev;
            test_SAX_ind = test_SAX_ind + 1;
            motif_valid_list = [motif_valid_list,i];
        end  
    end
    
    %% find similar patterns to predefined patterns 
    %  pw means pointwise, in contrary to head-tail representation
    pw = zeros([1,length(test_sig)]);
    
    for j = 1:length(motif_valid_list)
        
        % raw signal pattern length
        n = size(motif_cell{motif_valid_list(j)},2);
        % symbolic pattern length
        N = n*2;
        dists = [];
        test_SAX = test_SAX_for_motifscell{j};
        std_dev = std_cell{j};
    %     noise_fg indicate this substring is noise if 1
        noise_fg = std_dev*0;
        noise_fg(find(std_dev<std_thres)) = 1;

        
        % calculate similarity with predefined pattern
        for i = 1: size(test_SAX,1)
            
            sax1 = test_SAX(i,:);
            sax2 = motif_cell{motif_valid_list(j)}(1,:);
            
%             
%           correct bug in SAX core function
%           when all signal are the same value, return illegal: all 0,
%           which is forbidden by SAX core func itself
% 
%             
            for n_sax = 1:size(sax1,2)
                ifsax1posint = isposint(sax1(n_sax));
                if ifsax1posint == 0
                    interflg = 1;
                    sax1 = sax1+floor(dict_size/2);
                end
            end
            
            for n_sax = 1:size(sax1,2)
                ifsax1posint = isposint(sax1(n_sax));
                if ifsax1posint == 0
                    interflg = 1;
                end
            end
            disttmp = min_dist(sax1, sax2, dict_size,1);
            dists = [dists;disttmp];
        end
        
        % convert distance of substrings with tiny std_dev, that is noise, to
        % inf
        % from [0,1,0,0,1] to [NaN, inf, NaN, NaN, inf]
        dist_filepath = strcat(dist_folder, ['pred_all_dist_motif', num2str(j), '.csv']);
        csvwrite(dist_filepath, dists);
        
        ptn_filepath = strcat(dist_folder, ['pred_motif', num2str(j), '.csv']);        
        csvwrite(ptn_filepath, motif_cell{motif_valid_list(j)}(1,:));
        
        factor = noise_fg*inf; 
        factor(isnan(factor)) = 1;
        dists_wo_noise = dists.*factor;
        [sorted_dists, test_SAX_ind] = sort(dists_wo_noise);

        %------------------------------------------------------------------
        % this parameter decides how many segemntations will be selected
        %------------------------------------------------------------------
%         sym_candidate_ind = sym_ind(find(sorted_dists < dist_thres));
        sym_candidate_ind = find(dists<dist_thres);

        for tmp_i = 1:length(sym_candidate_ind)
            tmp = sym_candidate_ind(tmp_i);
            pred_list = [pred_list, sym_candidate_ind(tmp_i),sym_candidate_ind(tmp_i)+N-1,N,dists(tmp)];
        end
        pw = pw + head2pointwise(sym_candidate_ind, N, length(test_sig));
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save each activity’s predicted segments in separate csv
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    headtaillendist = save_headtaildist(pred_list, predfilepath);
    %????
    headtaillendist = sortrows(headtaillendist);
    headtaillendist = unique(headtaillendist,'rows');
    
    if size(headtaillendist,1) ~= 0
        
%         temporary comment in this grouping function
        
        headtaillendist = grouping(headtaillendist);
        

    % remove the first row of all zeros
        headtaillendist(all(headtaillendist==0,2),:)=[];
    end

    num_pred = size(headtaillendist,1);
    
    csvwrite(pred_reduce_filepath, headtaillendist);
    test_pred_ht = headtaillendist(:,1:3);
    
%     pw_unit = sign(pw);
    
end