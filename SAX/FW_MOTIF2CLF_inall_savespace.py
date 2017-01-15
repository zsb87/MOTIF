import os
import re
import csv
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import random
import glob
from sklearn import preprocessing
from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from scipy import *
from scipy.stats import *            
from scipy.signal import *
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import _pickle as cPickle
sys.path.append('C:/Users/szh702/Documents/FoodWatch/inlabStr/')
from stru_utils import *
import shutil   
import matlab.engine

i_subj = int(sys.argv[1])
run = int(sys.argv[2])
dist = float(sys.argv[3])
n_motif = float(sys.argv[4])
config_file = str(sys.argv[5])

protocol = 'inlabStr'
print(protocol)

# %  for US, qualified subjs: Dzung Shibo Rawan JC Jiapeng Matt
# %  for US, finished subjs:  Dzung Shibo Rawan  7     6     9

#           0       1       2      3      4       5        6      7     8   9(no HS)    10
subjs = ['Eric','Dzung','Gleb','Will','Shibo','Rawan','Jiapeng','JC','Cao','Matt', 'MattSmall']
subj = subjs[i_subj]


# 
# # segmentation based on motif
# 

eng = matlab.engine.start_matlab()
# i_subj = 1, run = 5, dist_thres = 1, n_motif =10 # ['Eric']=1 
ans = eng.FG_main_engy_newversion(subj, run, dist, n_motif, config_file)

print(ans)



# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("participant_name")
# args = parser.parse_args()
# print ("stru_engySeg2feats2clf_inall.py")
# print (args.participant_name)
# subj = args.participant_name
# print ("subj: %s" % str(sys.argv[1]))
# print ("run: %s" % str(sys.argv[2]))


save_flg = 1

# ,'Cao', 'JC', 'Eric', 'Jiapeng','Rawan'
# subjs = ['Dzung']#['Dzung','Cao', 'Shibo', 'Rawan', 'JC', 'Eric', ]# 'Matt',

# for run in [5]:
if 1:
    # for subj in subjs:
    if 1:
        # generate feature files both for test and train set of this subj
        split_subjs = ['train'+subj,'test'+subj]#

        for split_subj in split_subjs:

            subjfolder = split_subj + '/'
            folder = '../../'+protocol+'/subject/'
            featFolder = folder+subjfolder+"feature/all_features/"

            datafile =  folder+subjfolder+"testdata_labeled.csv"
            segfolder = folder+subjfolder+"segmentation/"
               
            # 
            #   generate prediction segment features
            #   1. generate raw data files from Head tail 
            # 
            if not os.path.exists(featFolder):
                os.makedirs(featFolder)

            # read in the raw data file
            # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
            #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  
            r_df = pd.read_csv(datafile)

            predActFilePath = segfolder+'engy_run'+str(run)+'_pred/pred_headtail_reduced_1.csv'
            gt_headtail = pd.read_csv(predActFilePath, names = ['Start','End','EnergyDur','dist'])

            # if os.path.exists(segfolder+'accx_pred_data/'):
            #     shutil.rmtree(segfolder+'accx_pred_data/')


            if not os.path.exists(segfolder+'engy_run'+str(run)+'_pred_data/'):
                os.makedirs(segfolder+'engy_run'+str(run)+'_pred_data/')

            for f in glob.glob(segfolder+'engy_run'+str(run)+'_pred_data/*'):
                os.remove(f)

            for i in range(len(gt_headtail)):
                saveFilePath = segfolder+'engy_run'+str(run)+'_pred_data/' + 'engy_pred_gesture_' + str(i) + '.csv'

                dataStart = int(gt_headtail['Start'].iloc[i])*2 - 2
                dataEnd = int(gt_headtail['End'].iloc[i])*2 + 1

                r_df_gesture = r_df.iloc[dataStart:dataEnd]
                # if dataEnd < len(r_df):
                if 1:
                    r_df_gesture.to_csv(saveFilePath)



            # 
            #   generate prediction segment features
            #   2. from raw data of segments generate features
            # 

            predFolder = segfolder+'engy_run'+str(run)+'_pred_data/'

            allfeatDF = pd.DataFrame()
            for _, dirnames, filenames in os.walk(predFolder):
                n_gest = len(filenames)

                if n_gest == 0:
                    print("no prediction file for current subject")
                    continue
                else:
                    for i in range(n_gest):

                        # read in the raw data file
                        # dataFrame:     Time  Angular_Velocity_x  Angular_Velocity_y  Angular_Velocity_z  
                        #                 Linear_Accel_x  Linear_Accel_y  Linear_Accel_z  unixtime synctime  

                        r_df = pd.read_csv(predFolder + 'engy_pred_gesture_' + str(i) + '.csv')

                        # pass raw data into filter
                        r_df = r_df[['Angular_Velocity_x', 'Angular_Velocity_y', 'Angular_Velocity_z', 'Linear_Accel_x','Linear_Accel_y','Linear_Accel_z']]

                        r_df = df_iter_flt(r_df)
                        if i % 200 == 0:
                            print(i)
                        r_df = add_pitch_roll(r_df)
                        # generate the features
                        feat = gen_feat(r_df)

                        featDF = pd.DataFrame(feat[1:] , columns=feat[0])
                        allfeatDF = pd.concat([allfeatDF,featDF])


            outfile = featFolder + "engy_run"+ str(run) +"_pred_features.csv"
            allfeatDF.to_csv(outfile, index =None)







        #
        # import features and model, do classification
        #         
        columns = ['Prec(pos)','F1(pos)','TPR','FPR','Specificity','MCC','CKappa','w-acc']
        crossValRes = pd.DataFrame(columns = columns, index = range(5))
        active_p_cnt = 0     


        outfolder = '../../'+protocol+'/result/seg_clf/engy_IS2ISseg_personalized/'+subj+'/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)

        for threshold_str in ['0.5']:#,'0.6','0.7','0.8','0.9'

            testsubj ='test'+subj
            trainsubj = 'train'+subj

            trainsubjF = trainsubj + '/'
            testsubjF = testsubj + '/'
            folder = '../../'+protocol+'/subject/'
            trainfeatFoler = folder+trainsubjF+"feature/all_features/"
            trainsegfolder = folder+trainsubjF+"segmentation/"


            df = pd.read_csv(trainfeatFoler+ "engy_run"+ str(run) +"_pred_features.csv" )
            labelDf = pd.read_csv(trainsegfolder+'engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv',names = ['label'])

            # 
            # notice:   duration should not be included in features 
            #           as in detection period this distinguishable feature will be in different distribution
            # 
            X = df.iloc[:,:-1].as_matrix()
            Y = labelDf['label'].iloc[:].as_matrix()

            classifier = RandomForestClassifier(n_estimators=185)
            classifier.fit(X, Y)


            # save the classifier
            mdlFolder = folder+trainsubjF+"model/"
            if not os.path.exists(mdlFolder):
                os.makedirs(mdlFolder)

            with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+threshold_str+'_run'+str(run)+'.pkl', 'wb') as fid:
                cPickle.dump(classifier, fid)    



            folder = '../../'+protocol+'/subject/'


            testfeatFolder = '../../'+protocol+'/subject/'+testsubj + '/feature/all_features/'
            testfeatFile = testfeatFolder+ "engy_run"+ str(run) +'_pred_features.csv'
            df_all = pd.read_csv(testfeatFile)
            print(len(df_all))

            labelFile = folder+testsubj + '/segmentation/engy_run'+str(run)+'_pred_label_thre'+threshold_str+'/seg_labels.csv'
            labelDf = pd.read_csv(labelFile, names = ['label'])
            print(len(labelDf))


            mdlFolder = folder+trainsubjF+"model/"
            # save the classifier
            with open(mdlFolder+'RF_185_trainset_motif_segs_thre'+threshold_str+'_run'+str(run)+'.pkl', 'rb') as fid:
                classifier = cPickle.load(fid)

            # 
            # notice:   duration should not be included in features 
            #           as in detection period this distinguishable feature will be in different distribution
            # 
            X = df_all.iloc[:,:-1].as_matrix()
            Y = labelDf['label'].as_matrix()

            

            prec_pos, f1_pos, TPR, FPR, Specificity, MCC, CKappa, w_acc, cm, y_pred = clf_cm_pickle(classifier, X, Y)
            ts = time.time()
            current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')

            print(current_time)
            np.savetxt(outfolder+'RF_185_motif_dist'+str(dist)+'_multi-thre_'+subj+'_run'+str(run)+'(109)_cm'+str(current_time)+'.csv', cm, delimiter=",")

            crossValRes['Prec(pos)'][active_p_cnt] = prec_pos
            crossValRes['F1(pos)'][active_p_cnt] = f1_pos
            crossValRes['TPR'][active_p_cnt] = TPR
            crossValRes['FPR'][active_p_cnt] = FPR
            crossValRes['Specificity'][active_p_cnt] = Specificity
            crossValRes['MCC'][active_p_cnt] = MCC
            crossValRes['CKappa'][active_p_cnt] = CKappa
            crossValRes['w-acc'][active_p_cnt] = w_acc
            active_p_cnt = active_p_cnt+1

        ts = time.time()
        current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
        print(current_time)

        crossValRes.to_csv( outfolder+'RF_185_motif_dist'+str(dist)+'_multi-thre_'+subj+'_run'+str(run)+'(109)'+str(current_time)+'.csv', index = None)