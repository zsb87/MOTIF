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
import sys
sys.path.append('C:/Users/szh702/Documents/FoodWatch/inlabStr/')
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
# 
# remove WeirdTimeJump and Confusing data
# 
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
        if i%500 == 0:
            print(i)
        allfeatDF = pd.concat([allfeatDF,featDF])
        
    if save_flg:
        print("saving energy to csv")
        allfeatDF.to_csv(featsfile)

    return allfeatDF

def gen_energy_file_timestamp(r_df_test, winsize, stride, freq, featsfile):
    i = 0
    allfeatDF = pd.DataFrame()

    while(i+winsize < r_df_test.shape[0]):    
        feat = gen_energy(r_df_test[i:i+winsize], freq)
        data = feat[1:]
        col_name = feat[0]

        featDF = pd.DataFrame(data = data, columns=col_name)
        featDF['unixtime'] = r_df_test['unixtime'].iloc[i+int(winsize/2)]
        featDF['Time'] = r_df_test.index[i+int(winsize/2)]
        featDF = featDF.set_index('Time')

        i += stride
        if i%500 == 0:
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
            return -1
    return 0



def unstrGenFeedingLabels(annotDf, r_df_test, activities):
    # firstly, extract the eating and drinking activites for 'Annotation' column
    # secondly, find HandUp and HandDown for 'MainCatetory' column from the dataframe by step 1


    # step 1:
    annot_f = pd.DataFrame()
    act_dur = []

    for i,activity in enumerate(activities):
        annot_f_tmp = annotDf.loc[(annotDf["Annotation"]==activity)]
        annot_f = pd.concat([annot_f, annot_f_tmp])

    annot_f = annot_f.sort_values(by='StartTime')

    # step 2:
    annot_HU = annot_f.loc[(annot_f["MainCategory"]=="HandUp")]
    annot_HD = annot_f.loc[(annot_f["MainCategory"]=="HandDown")]
    annot_HU = annot_HU.drop_duplicates()
    annot_HD = annot_HD.drop_duplicates()
    annot_HU.to_csv("../inlabUnstr/subject/"+subjfolder+"/f_HU.csv")
    annot_HD.to_csv("../inlabUnstr/subject/"+subjfolder+"/f_HD.csv")


    if len(annot_HU) != len(annot_HD):
        print("feeding gesture hand up and hand down in pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    if(checkHandUpDown(annot_HU, annot_HD)):
        print("feeding gesture error")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    feeding_St_list = list(annot_HU.StartTime.tolist())
    feeding_Et_list = list(annot_HD.EndTime.tolist())
    dur = []
    for n in range(len(feeding_St_list)):
        dur.append([feeding_St_list[n],feeding_Et_list[n]])


    # step 3: label test data
    # mark( df , col, label, intervals ):
    r_df_test = mark(r_df_test, 'feedingClass', 1, dur)
    # print(act_dur)

    return r_df_test



def unstrGenNonfeedingLabels_tmp(annotDf, r_df_test, activities):
    # firstly, extract the eating and drinking activites for 'Annotation' column
    # secondly, find HandUp and HandDown for 'MainCatetory' column from the dataframe by step 1


    # step 1:
    annot_f = pd.DataFrame()
    act_dur = []

    for i,activity in enumerate(activities):
        annot_f_tmp = annotDf.loc[(annotDf["Annotation"]==activity)]
        annot_f = pd.concat([annot_f, annot_f_tmp])
    
    annot_f = annot_f.sort_values(by='StartTime')


    # step 2:
    annot_HU = annot_f.loc[(annot_f["MainCategory"]=="HandUp")]
    annot_HD = annot_f.loc[(annot_f["MainCategory"]=="HandDown")]
    annot_HU = annot_HU.drop_duplicates()
    annot_HD = annot_HD.drop_duplicates()

    annot_HU.to_csv("../inlabUnstr/subject/"+subjfolder+"/nf_HU.csv")
    annot_HD.to_csv("../inlabUnstr/subject/"+subjfolder+"/nf_HD.csv")


    if(checkHandUpDown(annot_HU, annot_HD)):
        print("non-feeding gesture error")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    if len(annot_HU) != len(annot_HD):
        print("feeding gesture hand up and hand down not in pairs")
        print(len(annot_HU))
        print(len(annot_HD))
        exit()

    feeding_St_list = list(annot_HU.StartTime.tolist())
    feeding_Et_list = list(annot_HD.EndTime.tolist())
    dur = []
    for n in range(len(feeding_St_list)):
        dur.append([feeding_St_list[n],feeding_Et_list[n]])


    # step 3: label test data
    # mark( df , col, label, intervals ):
    r_df_test = mark(r_df_test, 'nonfeedingClass', 1, dur)
    # print(act_dur)

    return r_df_test



def genWinHandUpHoldingDownLabels(annotDf, r_df_test, activities):
    annot_HU = annotDf.loc[(annotDf["MainCategory"]=="HandUp")]
    annot_HD = annotDf.loc[(annotDf["MainCategory"]=="HandDown")]

    # for act in activities:
    #     annot_HU = annotDf.loc[(annotDf["MainCategory"]=="HandUp")&(annotDf["Annotation"]==act)]
    #     annot_HD = annotDf.loc[(annotDf["MainCategory"]=="HandDown")&(annotDf["Annotation"]==act)]
    annot_HD.to_csv("../inlabUnstr/subject/"+subjfolder+"/tmp_HD.csv")
    annot_HU.to_csv("../inlabUnstr/subject/"+subjfolder+"/tmp_HU.csv")

    if(checkHandUpDown(annot_HU, annot_HD)):
        print("non-feeding gesture error")
        exit()
    if len(annot_HU) != len(annot_HD):
        print("hand up and hand down not pairs")
        print(len(annot_HU))
        print(len(annot_HD))
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

        r_df_test_label = mark( r_df_test_label , 'activity', i+1, act_dur )

        print(act_dur)

    return r_df_test_label


def genFeedingGesture_DrinkingLabels(annotDf, r_df_test, activities):

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

        r_df_test_label = mark( r_df_test_label , 'activity', i+1, act_dur )
        print(act_dur)

    return r_df_test_label



def mergeFeatsLabels(allfeatDF, r_df_test_label, activities, lfeatfile):
    
    # allfeatDF.index = range(allfeatDF.shape[0])
    allfeatDF['feedingClass'] = 0
    allfeatDF['drinkingClass'] = 0
    allfeatDF['activity'] = 0
    allfeatDF['nonfeedingClass'] = 0

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.feedingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"feedingClass"] = 1
            # allfeatDF["feedingClass"].iloc[idx] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0

    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.drinkingClass[i:i+winsize].as_matrix()
        if Counter(class_arr)[1] > int(winsize/2):
            allfeatDF.ix[idx,"drinkingClass"] = 1
            # allfeatDF["drinkingClass"].iloc[idx] = 1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.activity[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"activity"] = i_act+1
                # allfeatDF["activity"].iloc[idx] = i_act+1
        i += stride
        idx += 1

    i = 0
    idx = 0
    while(idx < allfeatDF.shape[0]):    
        class_arr = r_df_test_label.nonfeedingClass[i:i+winsize].as_matrix()
        for i_act ,activity in enumerate(activities):

            if Counter(class_arr)[i_act+1] > int(winsize/2):
                allfeatDF.ix[idx,"nonfeedingClass"] = i_act+1
                # allfeatDF["nonfeedingClass"].iloc[idx] = i_act+1
        i += stride
        idx += 1

    allfeatDF = allfeatDF[['energy_acc_xyz','orientation_acc_xyz', 'energy_orientation', 'energy_acc_xxyyzz', 'energy_ang_xyz',"energy_ang_xyz_regularized", 'feedingClass', 'drinkingClass', 'activity','nonfeedingClass','unixtime'
    ]];

    if save_flg:
        allfeatDF.to_csv(lfeatfile)

    return allfeatDF



def genVideoSyncFile(featsfile, birthtime, r_df_test, save_flg, syncfile):
    video_sensor_bias_ms = 0
    featDF = pd.read_csv(featsfile)
    # featDF.index = range(featDF.shape[0])

    # r_df_test['Time'] = r_df_test.index

    birthtime_s = birthtime[:19]

    import time
    
    base_unixtime = time.mktime(datetime.datetime.strptime(birthtime_s,"%Y-%m-%d %H:%M:%S").timetuple())
    base_unixtime = base_unixtime*1000 + video_sensor_bias_ms

    # r_df_test["synctime"] = (r_df_test["synctime"] - base_unixtime)/1000
    r_df_test["synctime"] = (r_df_test["unixtime"] - base_unixtime)/1000
    extr_idx = list(range(0,len(r_df_test)-winsize,stride))
    r_df_tDsample = r_df_test.iloc[extr_idx]

    r_df_tDsample.index = range(len(r_df_tDsample))
    r_df_tDsample = r_df_tDsample[['synctime','Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','feedingClass']]

    raw_energy = pd.concat([featDF, r_df_tDsample], axis=1)

    if save_flg:
        raw_energy = raw_energy[['Time','unixtime','synctime','energy_acc_xyz','orientation_acc_xyz','energy_orientation',"energy_acc_xxyyzz",'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','feedingClass']]
        raw_energy.to_csv(syncfile)



# getTimeError: Video/Annot leading sensor
def getTimeError(x, pos):
    if pos == 'tripod':
        return {
                # 10 subjs
            'Rawan': 12,
            'Dzung': 18.1,# 10.5,
            'Shibo': 22.77, #14,
            # 'JC': -24,
            # 'Matt': -2,
            # 'Jiapeng': 41,
            # 'Eric': 18,
            # 'Will': 0, 
            # 'Cao': 44,
            # 'Gleb': -4
        }.get(x, 0)
    if pos == 'desk':
        return {
        # 7 subjs
            'Dzung': 2,
            # 'Shibo':
            # 'JC': 0,
            # 'Matt': 3,
            # 'Jiapeng': 0,
            # 'Eric': 0,
            # 'Will': 0,
            # 'Gleb': 2,
            
        }.get(x, 0)

def getFeedingGestures(x):
    return {
        # 10 subjs
        'Dzung': ['Spoon', 'Straw','HandFries','HandChips'],
        'JC': ['Spoon', 'SaladFork','HandSourPatch', 'HandBread', 'HandChips'],
        'Matt': ['HandChips', 'HandBread','HandChips', 'HandBread','Spoon', 'SaladFork', 'Cup', 'Bottle'],
        'Jiapeng': ['HandCracker', 'Popcorn', 'HandChips','Cup', 'LicksFingers', 'Spoon','Cup','SaladFork', 'HandBread', 'SaladSpoon'],
        'Eric': ['HandChips','Cup','Spoon'],
        'Will':  ['popcorn', 'RedFish', 'swedishFish', 'chips', 'EatsFromBag', 'bottle', 'Bottle'], 
        'Shibo': ['Straw', 'HandFries','HandBurger', 'Spoon'],
        'Rawan': ['HandChips'],
        'Cao': ['Cup'],
        'Gleb': ['Bottle','Cup','KnifeFork','Spoon','HandBread','Straw']
        }.get(x, 0)

def getNonfeedingGestures(x):
    return {
        # 10 subjs
        'Dzung': [ 'MovesGlasses', 'Scratches', 'ChinRest', 'Napkin', 'PhoneText','MovesAccessory'],
        'JC': ['Wave', 'Scratches', 'PhoneText', 'ChinRest',   'RaisesBag', 'OutOfSeat', 'MovesAccessory'],
        'Matt': ['AdjustAccessory', 'CombHair', 'ChinRest'],
        'Jiapeng': ['Wave', 'Scratches', 'PhoneText', 'ChinRest', 'Nose', 'SyncSignal', 'PhoneText', 'Phone', 'ChinRest'],
        'Eric': ['Wave', 'AdjustAccessory', 'Scratches', 'Nose', 'nose', 'SyncSignal', 'PhoneText','TextPhone', 'CombHair', 'ChinRest'],
        'Will':  ['MovesGlasses', 'LicksLips',  'WipesMouth'], 
        'Shibo': ['Scratches', 'PhoneText','Nose'],
        'Rawan': ['AdjustAccessory','Scratches','Nose','Wave','PhoneText','HandText','Phone','CombHair','ChinRest'],
        'Cao': ['AdjustAccessory'],
        'Gleb': ['AdjustAccessory','Scratches','Nose','Wave','PhoneText','HandText','Phone','CombHair','ChinRest']
        }.get(x, 0)

# ------------------------------------------------------------------------------
# 
# import raw sensor data
# 
# ------------------------------------------------------------------------------
save_flg = 1

subjs = ['Shibo']#,'Shibo','Dzung','Will', 'Gleb', 'JC','Matt','Jiapeng', 'Cao', 'Eric'
protocol = 'inlabUnstr'

for active_participant_counter, subj in enumerate(subjs):
    if (not (active_participant_counter == 1)):

        subjfolder = subj
        file = "../inlabUnstr/subject/"+subjfolder+"/right/data.csv"
        birthfileT = "../inlabUnstr/subject/"+subjfolder+'/tripod video birth.txt'
        birthfileD = "../inlabUnstr/subject/"+subjfolder+'/desk video birth.txt'

        if (os.path.isfile(birthfileT)) and (os.path.isfile(birthfileD)):
            annotation_state =3
        elif (os.path.isfile(birthfileT)) :
            annotation_state =1
            position = 'tripod'
        elif (os.path.isfile(birthfileD)) :
            annotation_state =2
            position = 'desk'



        if annotation_state == 3:
            birthfile = birthfileD

            annotDf_D = importAnnoFile("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-desk.csv")
            annotDf_T = importAnnoFile("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-tripod.csv")
            
            annotDf_D.StartTime = annotDf_D.StartTime + pd.Timedelta(seconds=getTimeError(subj,'desk'))
            annotDf_D.EndTime = annotDf_D.EndTime + pd.Timedelta(seconds=getTimeError(subj,'desk'))
            annotDf_T.StartTime = annotDf_T.StartTime + pd.Timedelta(seconds=getTimeError(subj,'tripod'))
            annotDf_T.EndTime = annotDf_T.EndTime + pd.Timedelta(seconds=getTimeError(subj,'tripod'))

            # remove WeirdTimeJump and Confusing data
            annotDf_D = processAnnot(annotDf_D)
            annotDf_T = processAnnot(annotDf_T)
            annotDf = pd.concat([annotDf_D,annotDf_T])

            annotDf = annotDf.drop_duplicates()

            if save_flg:
                annotDf_D.to_csv("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-desk-sensortime.csv")
                annotDf_T.to_csv("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-tripod-sensortime.csv")
                annotDf.to_csv("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-united-sensortime.csv")
            


        if annotation_state ==2:
            birthfile = birthfileD
            annotDf = importAnnoFile("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-"+position+"-sensortime.csv")
            annotDf.StartTime = annotDf.StartTime + pd.Timedelta(seconds=getTimeError(subj,position))
            annotDf.EndTime = annotDf.EndTime + pd.Timedelta(seconds=getTimeError(subj,position))
            # remove WeirdTimeJump and Confusing data
            annotDf = processAnnot(annotDf)
            if save_flg:
                annotDf.to_csv("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-united-sensortime.csv")   



        if annotation_state ==1:
            birthfile = birthfileT
            annotDf = importAnnoFile("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-"+position+"-sensortime.csv")
            annotDf.StartTime = annotDf.StartTime + pd.Timedelta(seconds=getTimeError(subj,position))
            annotDf.EndTime = annotDf.EndTime + pd.Timedelta(seconds=getTimeError(subj,position))
            # remove WeirdTimeJump and Confusing data
            annotDf = processAnnot(annotDf)
            if save_flg:
                annotDf.to_csv("../inlabUnstr/subject/" + subjfolder + "/annotation/annotations-edited-united-processed-sensortime.csv")   



        birthtime = open(birthfile, 'r').read()
        r_df_test = read_r_df_test_st(subj, file, birthtime)
        r_df_test = df_iter_flt(r_df_test)

        # ------------------------------------------------------------------------------
        # 
        # import annotation file
        # 
        # adjust annotation dataframe
        # 
        # ------------------------------------------------------------------------------



        # ------------------------------------------------------------------------------
        # 
        # remove black out(time jumping) parts
        # 
        # save in testdata.csv
        # 
        # ------------------------------------------------------------------------------

        r_df_test = rm_black_out(r_df_test, annotDf)

        if save_flg:
            r_df_test.to_csv("../inlabUnstr/subject/"+subjfolder+"/testdata.csv")


        # ------------------------------------------------------------------------------
        # 
        # generate labels for raw data
        # 
        # save in testdata_labeled.csv
        # 
        # ------------------------------------------------------------------------------

        if subj == 'Rawan':
            activities = [ 
                'HandChips'
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

        fg = getFeedingGestures(subj)
        nfg = getNonfeedingGestures(subj)
        r_df_test_label = genFeedingGesture_DrinkingLabels(annotDf, r_df_test, activities)
        r_df_test_label.to_csv('dbgtmp_genFeedingGesture_DrinkingLabels.csv')

        r_df_test_label = unstrGenFeedingLabels(annotDf, r_df_test_label, fg)
        r_df_test_label.to_csv('dbgtmp_unstrGenFeedingLabels.csv')

        r_df_test_label = unstrGenNonfeedingLabels_tmp(annotDf, r_df_test_label, nfg)

        r_df_test_label.to_csv("../inlabUnstr/subject/" + subjfolder + "/testdata_labeled.csv")


        # ------------------------------------------------------------------------------
        # 
        # generate feature file wo labels
        # 
        # save in engy_ori_win4_str2.csv
        # 
        # ------------------------------------------------------------------------------

        freq = 31
        winsize = 4 # for fft energy calculation
        stride = 2

        featfolder = "../inlabUnstr/subject/"+subjfolder+"/feature/"
        energyfolder = "../inlabUnstr/subject/"+subjfolder+"/feature/energy/"

        if not os.path.exists(featfolder):
            os.makedirs(featfolder)
        if not os.path.exists(energyfolder):
            os.makedirs(energyfolder)

        featsfile = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+".csv"
        allfeatDF = gen_energy_file_timestamp(r_df_test, winsize, stride, freq, featsfile)


        # ------------------------------------------------------------------------------
        # 
        # merge label file and feature file
        # 
        # ------------------------------------------------------------------------------

        lfeatfile = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
        allfeatDF = mergeFeatsLabels(allfeatDF, r_df_test_label, activities, lfeatfile)


        # ------------------------------------------------------------------------------
        # 
        # import feature data
        # 
        # ------------------------------------------------------------------------------
        featsfile = "../"+protocol+"/subject/"+subjfolder+"/feature/energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"


        if annotation_state ==1:
            birthfile = birthfileT
            birthtime = open(birthfile, 'r').read()
            syncfile =  "../"+protocol+"/subject/"+subjfolder+"/feature/energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_sync_tripod.csv"
            genVideoSyncFile(featsfile, birthtime, r_df_test_label, save_flg, syncfile)


        if annotation_state ==2:
            birthfile = birthfileD
            birthtime = open(birthfile, 'r').read()
            syncfile =  "../"+protocol+"/subject/"+subjfolder+"/feature/energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_sync_desk.csv"
            genVideoSyncFile(featsfile, birthtime, r_df_test_label, save_flg, syncfile)


        if annotation_state == 3:
            birthfile = birthfileD
            birthtime = open(birthfile, 'r').read()
            syncfile =  "../"+protocol+"/subject/"+subjfolder+"/feature/energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_sync_desk.csv"
            genVideoSyncFile(featsfile, birthtime, r_df_test_label, save_flg, syncfile)

            birthfile = birthfileT
            birthtime = open(birthfile, 'r').read()
            syncfile =  "../"+protocol+"/subject/"+subjfolder+"/feature/energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_sync_tripod.csv"
            genVideoSyncFile(featsfile, birthtime, r_df_test_label, save_flg, syncfile)