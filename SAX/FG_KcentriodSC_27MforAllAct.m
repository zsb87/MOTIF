function FG_KcentriodSC_27MforAllAct(subj, config_file)
    %% Evaluate global configuration file
    try
        eval(config_file);
    catch
        disp('config file failed!')
    end
        
    ptns_cell = [];
    ptcell_ind = 1;
    
    %----------------------------------------------------------------------------------
    %  pick up the min-med-max motifs from all activities
    %----------------------------------------------------------------------------------
    X = [];
    
    for act_ind = 1:size(feeding,2)
        
        energyfolder = [folder, feeding{act_ind},'/',subj,'(8Hz)/'];
        
        %% read energy file
        %% check the column number
        energyfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_10gestures.csv'];
        labelfile = ['engy_ori_win', num2str(win), '_str', num2str(stride),'_10gestures_label.csv'];
        data = csvread(strcat(energyfolder,energyfile),1);
        energy_acc_xyz = data(:,2);
        label = csvread(strcat(energyfolder,labelfile),1);
        fClass = label(:,1);
        
        act_rootfolder = [energyfolder, 'segmentation/motif_activity/'];
        gt_headtail = csvread(strcat(act_rootfolder,'gt_feeding_headtail.csv'));
        
        maxlen = max(gt_headtail(:,3));
        
        
    
        nn=3;
        [ksc cent] = ksc_toy(X, nn);
        figure;
        for i=1:nn
          subplot(1,nn,i);
          plot(cent(i,:));
          title('ksc');
          if max(cent(i,:)) == 0
              axis([0 maxlen 0 1]);
          else
          axis([0 maxlen 0 1.2 * max(cent(i,:))]);
          end
        end
    end
    
    for i = 1:length(gt_headtail)
        a = energy_acc_xyz(gt_headtail(i,1):gt_headtail(i,2))';
        X = [X;zeros(1,maxlen - gt_headtail(i,3)), a];
    end
end
