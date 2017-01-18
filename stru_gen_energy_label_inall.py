import os
import re
import csv
import matplotlib
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import scipy.io as sio
from collections import Counter
from sklearn import preprocessing
from scipy import stats
from scipy import *
from scipy.stats import *
from scipy.signal import *
from sklearn.metrics import matthews_corrcoef
import numpy.polynomial.polynomial as poly
import plotly 
from stru_utils import *



def read_r_df_test(subj, file, birthtime, deadtime):
    r_df = pd.read_csv(file)
    r_df = r_df[["Time","Angular_Velocity_x","Angular_Velocity_y","Angular_Velocity_z","Linear_Accel_x","Linear_Accel_y","Linear_Accel_z"]]
    r_df["unixtime"] = r_df["Time"]
    r_df["synctime"] = r_df["unixtime"]
    r_df['Time'] = pd.to_datetime(r_df['Time'],unit='ms',utc=True)
    r_df = r_df.set_index(['Time'])

    # to video absolute time
    r_df.index = r_df.index.tz_localize('UTC').tz_convert('US/Central')

    # cut and select the test part
    mask = ((r_df.index > birthtime) & (r_df.index < deadtime))
    r_df_test = r_df.loc[mask]
    
    return r_df_test

def read_r_df_test_st(subj, file, birthtime):
    r_df = pd.read_csv(file)
    r_df = r_df[["Time","Angular_Velocity_x","Angular_Velocity_y","Angular_Velocity_z","Linear_Accel_x","Linear_Accel_y","Linear_Accel_z"]]
    r_df["unixtime"] = r_df["Time"]
    r_df["synctime"] = r_df["unixtime"]
    r_df['Time'] = pd.to_datetime(r_df['Time'],unit='ms',utc=True)
    r_df = r_df.set_index(['Time'])

    # to video absolute time
    r_df.index = r_df.index.tz_localize('UTC').tz_convert('US/Central')

    # cut and select the test part
    mask = (r_df.index > birthtime)
    r_df_test = r_df.loc[mask]
    
    return r_df_test


def rm_black_out(r_df_test, annotDf):
    annot_blac = annotDf.loc[(annotDf["Annotation"]=="WeirdTimeJump")&(annotDf["MainCategory"]=="Confusing")]

    StartTime_list = list(annot_blac.StartTime.tolist())
    EndTime_list = list(annot_blac.EndTime.tolist())

    for n in range(len(StartTime_list)):
        r_df_test = r_df_test[(r_df_test.index < str(StartTime_list[n]))|(r_df_test.index > str(EndTime_list[n]))]

    return r_df_test


def importAnnoFile(annot_file):
    annotDf = pd.read_csv(annot_file, encoding = "ISO-8859-1")
    # print(annotDf['StartTime'])
    annotDf['StartTime'] = pd.to_datetime(annotDf['StartTime'],utc=True)
    annotDf['EndTime'] = pd.to_datetime(annotDf['EndTime'],utc=True)
    return annotDf

def processAnnot(annotDf):
    annotDf['drop'] = 0
    annotDf['duration'] = annotDf['EndTime'] - annotDf['StartTime']
    annotDf['drop'].loc[(annotDf['duration']<'00:00:01.5')&(annotDf["Annotation"]=="WeirdTimeJump")&(annotDf["MainCategory"]=="Confusing")]= 1
    annotDf = annotDf.loc[annotDf['drop'] == 0]

    annotDf  = annotDf[['StartTime','EndTime','Annotation','MainCategory']]
    return annotDf

def gen_energy_file(r_df_test, winsize, stride, freq, featsfile):
    i = 0
    allfeatDF = pd.DataFrame()

    while(i+winsize < r_df_test.shape[0]):    
        feat = gen_energy(r_df_test[i:i+winsize], freq)
        featDF = pd.DataFrame(feat[1:] , columns=feat[0])
        i += stride
        if i%100 == 0:
            print(i)
        allfeatDF = pd.concat([allfeatDF,featDF])
        
    if save_flg:
        print("saving energy to csv")
        allfeatDF.to_csv(featsfile)

    return allfeatDF


def checkHandUpDown(annot_HU, annot_HD):
    for i in range(1,max(len(annot_HD),len(annot_HU))):
        if not (annot_HU.StartTime.iloc[i]>annot_HD.StartTime.iloc[i-1]) & (annot_HU.StartTime.iloc[i]<annot_HD.StartTime.iloc[i]):
            print("subj: "+subj)
            print("trouble line: "+str(i))
            print("annot_HandDown.StartTime of "+str(i-1)+" is "+str(annot_HD.StartTime.iloc[i-1]))
            print("annot_HandUp.StartTime of "+str(i)+" is "+str(annot_HU.StartTime.iloc[i]))
            print("annot_HandDown.StartTime of "+str(i)+" is "+str(annot_HD.StartTime.iloc[i]))
            exit()


def genWinHandUpHoldingDownLabels(annotDf, r_df_test, activities):
    annot_HU = annotDf.loc[annotDf["MainCategory"]=="HandUp"]
    annot_HD = annotDf.loc[annotDf["MainCategory"]=="HandDown"]
    annot_HD.to_csv("../inlabStr/subject/"+subjfolder+"/tmp_HD.csv")
    annot_HU.to_csv("../inlabStr/subject/"+subjfolder+"/tmp_HU.csv")

    checkHandUpDown(annot_HU, annot_HD)

    if len(annot_HU) != len(annot_HD):
        print("hand up and hand down not pairs")
        exit()


    HU_St_list = list(annot_HU.StartTime.tolist())
    HD_Et_list = list(annot_HD.EndTime.tolist())

    UD_dur = []

    for n in range(len(HU_St_list)):
        UD_dur.append([HU_St_list[n],HD_Et_list[n]])

    r_df_test_label = markClassPeriod( r_df_test,'nonfeedingClass' , UD_dur )

    for i,activity in enumerate(activities):

        annot_act = annotDf.loc[(annotDf["Annotation"]==activity)]#&((annotDf["MainCategory"]=="Drinking")|(annotDf["MainCategory"]=="Eating")|(annotDf["MainCategory"]=="Other"))
        act_St_list = list(annot_act.StartTime.tolist())
        act_Et_list = list(annot_act.EndTime.tolist())
        act_dur = []

        for n in  range(len(act_St_list)):
            act_dur.append([act_St_list[n],act_Et_list[n]])

        r_df_test_label = markExistingClassPeriod( r_df_test_label , 'activity', i+1, act_dur )

        print(act_dur)

    return r_df_test_label

def genWinFrameLabels(annotDf, r_df_test, activities):

    # 'feedingClass' = 1 means it is feeding gesture including drinking
    # 'feedingClass' = 0 means it is not feeeding

    # 'activity' implies the activity of the whole period

    annot_feeding = annotDf.loc[annotDf["MainCategory"]=="FeedingGesture"]
    annot_drinking = annotDf.loc[annotDf["MainCategory"]=="Drinking"]

    Feeding_St_list = list(annot_feeding.StartTime.tolist())
    Feeding_Et_list = list(annot_feeding.EndTime.tolist())

    drinking_St_list = list(annot_drinking.StartTime.tolist())
    drinking_Et_list = list(annot_drinking.EndTime.tolist())

    feeding_dur = []
    drinking_dur = []

    for n in range(len(Feeding_St_list)):
        feeding_dur.append([Feeding_St_list[n],Feeding_Et_list[n]])

    for n in range(len(drinking_St_list)):
        drinking_dur.append([drinking_St_list[n],drinking_Et_list[n]])

    r_df_test_label = markClassPeriod( r_df_test,'feedingClass' , feeding_dur )
    r_df_test_label = markClassPeriod( r_df_test_label,'drinkingClass' , drinking_dur )

    for i,activity in enumerate(activities):

        annot_act = annotDf.loc[(annotDf["Annotation"]==activity)]#&((annotDf["MainCategory"]=="Drinking")|(annotDf["MainCategory"]=="Eating")|(annotDf["MainCategory"]=="Other"))
        act_St_list = list(annot_act.StartTime.tolist())
        act_Et_list = list(annot_act.EndTime.tolist())
        act_dur = []

        for n in  range(len(act_St_list)):
            act_dur.append([act_St_list[n],act_Et_list[n]])

        r_df_test_label = markExistingClassPeriod( r_df_test_label , 'activity', i+1, act_dur )
        print(act_dur)

    return r_df_test_label



def mergeFeatsLabels(allfeatDF, r_df_test_label, activities, lfeatfile):
    
    allfeatDF.index = range(allfeatDF.shape[0])
    allfeatDF['feedingClass'] = 0
    allfeatDF['drinkingClass'] = 0
    allfeatDF['activity'] = 0
    allfeatDF['nonfeedingClass'] = 0

    i = 0
    idx = 0
    # while(i+winsize < r_df_test.shape[0]):    
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.feedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"feedingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0

    while(idx < allfeatDF.shape[0]):    

    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.drinkingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"drinkingClass"] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.activity[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"activity"] = i_act+1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
    # while(i+winsize < r_df_test.shape[0]):    
        class_arr = r_df_test_label.nonfeedingClass[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"nonfeedingClass"] = i_act+1
        i += stride
        idx += 1

    allfeatDF = allfeatDF[['energy_acc_xyz','orientation_acc_xyz', 'energy_orientation', 'energy_acc_xxyyzz', 'energy_ang_xyz',"energy_ang_xyz_regularized", 'feedingClass', 'drinkingClass', 'activity','nonfeedingClass'
    ]];

    if save_flg:
        allfeatDF.to_csv(lfeatfile)

    return allfeatDF



def genVideoSyncFile(featsfile, birthtime, r_df_test, save_flg, syncfile):

    featDF = pd.read_csv(featsfile)
    featDF.index = range(featDF.shape[0])

    r_df_test['Time'] = r_df_test.index

    birthtime_s = birthtime[:19]

    import time
    
    base_unixtime = time.mktime(datetime.datetime.strptime(birthtime_s,"%Y-%m-%d %H:%M:%S").timetuple())
    base_unixtime = base_unixtime*1000 + video_sensor_bias_ms

    r_df_test["synctime"] = (r_df_test["synctime"] - base_unixtime)/1000
    extr_idx = list(range(0,len(r_df_test)-winsize,stride))
    r_df_tDsample = r_df_test.iloc[extr_idx]

    r_df_tDsample.index = range(len(r_df_tDsample))

    raw_energy = pd.concat([featDF, r_df_tDsample], axis=1)

    if save_flg:
        raw_energy = raw_energy[['Time','unixtime','synctime','energy_acc_xyz','orientation_acc_xyz','energy_orientation',"energy_acc_xxyyzz",'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','Class']]
        raw_energy.to_csv(syncfile)

        # enrg_class_activity = raw_energy[["energy_acc_xyz","Class","activity"]];
        # print(enrg_class_activity)
        # enrg_class_activity.dropna() 
        # enrg_class_activity.to_csv(featsfile)






# ------------------------------------------------------------------------------
# 
# import raw sensor data
# 
# ------------------------------------------------------------------------------
save_flg = 1

save_flg = 1

i_subj = int(sys.argv[1])

#   for US, qualified subjs: Dzung Shibo Rawan JC Jiapeng Matt
#   for US, finished subjs:  Dzung Shibo Rawan  7     6     9


#           0       1       2       3        4    5    6(no HS)  7       8     9      10
subjs = ['Rawan','Shibo','Dzung', 'Will', 'Gleb','JC','Matt','Jiapeng','Cao','Eric', 'MattSmall']
subj = subjs[i_subj]

if 1:

    subjfolder = subj #+ '(8Hz)'
    file = "../inlabStr/subject/"+subjfolder+"/right/data.csv"
    birthfile = "../inlabStr/subject/"+subjfolder+'/birth.txt'
    if subj == 'Matt':
        birthfile = "../inlabStr/subject/"+subjfolder+'/birth-20min.txt'

    birthtime = open(birthfile, 'r').read()
    print(birthtime)
    
    if subj == 'Shibo':
        endfile = "../inlabStr/subject/"+subjfolder+'/end.txt'
        endtime = open(endfile, 'r').read()
        print(endtime)
        r_df_test = read_r_df_test(subj, file, birthtime, endtime)
    else:
        r_df_test = read_r_df_test_st(subj, file, birthtime)


    

    video_sensor_bias_ms = 0
    
    # 

    # ------------------------------------------------------------------------------
    # 
    #  smoothing data
    # 
    # ------------------------------------------------------------------------------
    r_df_test = df_iter_flt(r_df_test)


    # ------------------------------------------------------------------------------
    # 
    # import annotation file
    # 
    # adjust annotation dataframe
    # 
    # ------------------------------------------------------------------------------
    annot_file = "../inlabStr/subject/" + subjfolder + "/annotation/annotations-edited.csv"
    annotDf = importAnnoFile(annot_file)

    def getTimeError(x):
        return {
            'Dzung': 0.24,
            'JC': 16.75,
            'Matt': -613,
            'Jiapeng': 14.5,
            'Eric': -63,
            'Will': 0,  #???
            'Shibo': 0,
            'Rawan': 0,
            'Gleb': 5
        }.get(x, 0)

    time_error = getTimeError(subj)

    annotDf.StartTime = annotDf.StartTime + pd.Timedelta(seconds=time_error)
    annotDf.EndTime = annotDf.EndTime + pd.Timedelta(seconds=time_error)

    annotDf = processAnnot(annotDf)
    if save_flg:
        annotDf.to_csv("../inlabStr/subject/" + subjfolder + "/annotation/annotations-edited-processed.csv")


    # ------------------------------------------------------------------------------
    # 
    # remove black out(time jumping) parts
    # 
    # save in testdata.csv
    # 
    # ------------------------------------------------------------------------------
    r_df_test = rm_black_out(r_df_test, annotDf)

    if save_flg:
        r_df_test.to_csv("../inlabStr/subject/"+subjfolder+"/testdata.csv")


    # ------------------------------------------------------------------------------
    # 
    # generate labels for raw data
    # 
    # save in testdata_labeled.csv
    # 
    # ------------------------------------------------------------------------------

    if subj == 'Rawan':
        activities = [ 
            'Spoon',            #1
            'HandBread',        #2
            'Chopstick',        #3
            'KnifeFork',        #4
            'SaladFork',        #5
            'HandChips',        #6
            'Cup',              #7
            'Straw',            #8
            'Phone',            #9
            'SmokeMiddle',      #10
            'SmokeThumb',       #11
            'Bottle',           #12
            'Nose',             #13
            'ChinRest',         #14
            'Scratches',        #15
            'Mirror',           #16
            'Teeth',            #17
        ]

    else:
        activities = [
            'Spoon',            #1
            'HandBread',        #2
            'Cup',              #3
            'Chopstick',        #4
            'KnifeFork',        #5
            'Bottle',           #6
            'SaladFork',        #7
            'HandChips',        #8
            'Straw',            #9
            'SmokeMiddle',      #10
            'SmokeThumb',       #11
            'ChinRest',         #12
            'Phone',            #13
            'Mirror',           #14
            'Scratches',        #15
            'Nose',             #16
            'Teeth',            #17
        ]

    r_df_test_label = genWinFrameLabels(annotDf, r_df_test, activities)
    r_df_test_label = genWinHandUpHoldingDownLabels(annotDf, r_df_test_label, activities)
    r_df_test_label.to_csv("../inlabStr/subject/" + subjfolder + "/testdata_labeled.csv")


    # ------------------------------------------------------------------------------
    # 
    # generate feature file wo labels
    # 
    # save in engy_ori_win4_str2.csv
    # 
    # ------------------------------------------------------------------------------
    freq = 31
    winsize = 4
    stride = 2

    featfolder = "../inlabStr/subject/"+subjfolder+"/feature/"
    energyfolder = "../inlabStr/subject/"+subjfolder+"/feature/energy/"

    if not os.path.exists(featfolder):
        os.makedirs(featfolder)
    if not os.path.exists(energyfolder):
        os.makedirs(energyfolder)

    featsfile = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+".csv"
    allfeatDF = gen_energy_file(r_df_test, winsize, stride, freq, featsfile)



    # ------------------------------------------------------------------------------
    # 
    # merge label file and feature file
    # 
    # ------------------------------------------------------------------------------
    file = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+".csv"
    allfeatDF = pd.read_csv(file)

    lfeatfile = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
    allfeatDF = mergeFeatsLabels(allfeatDF, r_df_test_label, activities, lfeatfile)



    # ------------------------------------------------------------------------------
    # 
    # import feature data
    # 
    # ------------------------------------------------------------------------------

    # featsfile = featfolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
    # syncfile = featfolder + "raw_engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled_time.csv"

    # # featsfile = featfolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
    # # syncfile = featfolder + "raw_engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled_time.csv"

    # genVideoSyncFile(featsfile, birthtime, r_df_test, save_flg, syncfile)
