function [ptns_cell, engy_test_cell, test_gt_ht_cell] = FG_KcentriodSC_27MforAllAct_setsplit_us(subj, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!')
    end
        
    ptns_cell = [];
    test_gt_ht_cell = [];
    
    X = [];
    gt_headtail = [];

    energyfolder = [folder, subj,'(8Hz)/feature/energy/'];
        
    
        %% read energy file
        %% check the column number
        energyfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_labeled.csv'];
        data = csvread(strcat(energyfolder,energyfile),1);
        energy_acc_xyz = data(:,2);
        fClass = data(:,8);
        
        act_rootfolder = [energyfolder, 'segmentation/motif_activity/'];
        if ~exist(act_rootfolder, 'dir')    mkdir(act_rootfolder),   end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % save all activities' ground truth in one file
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        gt_headtail = pointwise2headtail(fClass);
        save_headtail(gt_headtail, strcat(act_rootfolder,'gt_feeding_headtail.csv'));
        gt_headtail = csvread(strcat(act_rootfolder,'gt_feeding_headtail.csv'));% it has (n,3) shape now

        % expend gt_headtail from 3 columns to 4 columns
        append_row = gt_headtail(2:end,1)-1;
        gt_headtail = [gt_headtail,[append_row;length(energy_acc_xyz)]];
        
        % randomly split to train test set
        rand_gt = gt_headtail(randperm(size(gt_headtail,1)),:); % a random permutation of your data
        train_HT = rand_gt(1:floor(size(gt_headtail,1)*0.7),:);
        test_HT = rand_gt(floor(size(gt_headtail,1)*0.7)+1:end,:);
       
        for i = 1:size(train_HT,1)
            maxlen = max(train_HT(:,3));

            a = energy_acc_xyz(train_HT(i,1):train_HT(i,2))';
            X = [X;zeros(1,maxlen - train_HT(i,3)), a];
        end
        
        nn=27;
        [~, cent] = ksc_toy(X, nn);
        
        cent(all(cent==0,2),:)=[];
        
        ptcell_ind = 1;
        for i = 1:size(cent,1)
            A = cent(i,:);
            A_nz = A(find(any(A,1),1,'first'):end);
            ptns_cell{ptcell_ind} = timeseries2symbol(A_nz, length(A_nz), floor(length(A_nz)/2), dict_size,1);
            ptcell_ind = ptcell_ind + 1;
        end
        ptcell_ind = ptcell_ind - 1;
        ind_tmp = 1;
        for i = 1:ptcell_ind
            if all(ptns_cell{i}==0,2)==0
                ptns_cell_tmp{ind_tmp} = ptns_cell{i};
                ind_tmp = ind_tmp + 1;
            end
        end
        ptns_cell = ptns_cell_tmp;
        
        engy_test_cell = [];
        for i = 1: size(test_HT,1)
            engy_test_cell{i} = energy_acc_xyz(test_HT(i,1): test_HT(i,4));
            test_gt_ht_cell{i} = test_HT(i,:);
        end
    
end
