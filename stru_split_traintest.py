import os
import sys
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


def pointwise2headtail(pointwise_rpr):
    pw = pointwise_rpr
    diff = np.concatenate((pw[:-1],np.array([0]))) - np.concatenate((np.array([0]),pw[:-1]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1

    headtail_rpr = np.vstack((ind_head, ind_tail)).T;

    return headtail_rpr


winsize = 4
stride = 2

save_flg = 1

i_subj = int(sys.argv[1])

#   for US, qualified subjs: Dzung Shibo Rawan JC Jiapeng Matt
#   for US, finished subjs:  Dzung Shibo Rawan  7     6     9


#           0       1       2       3        4    5    6(no HS)  7       8     9      10
subjs = ['Rawan','Shibo','Dzung', 'Will', 'Gleb','JC','Matt','Jiapeng','Cao','Eric', 'MattSmall']
subj = subjs[i_subj]

if 1:

    subjfolder = subj + '/'
    folder = '../inlabStr/subject/'
    allfeatDF = pd.DataFrame()

    featFolder = folder+subjfolder+"feature/"
    segFolder = folder+subjfolder+"segmentation/"
    rawDataFile =  folder+subjfolder+"testdata.csv"
    testDataFolder = segFolder+'test_data/'
    energyfolder = folder+subjfolder+"feature/energy/"
    allFeatFolder = folder+subjfolder+"feature/all_features/"    
    lengyfolder = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"

    trainFolder = folder+'train'+subjfolder+"feature/energy/"
    testFolder = folder+'test'+subjfolder+"feature/energy/"
    trainDataFolder = folder+'train'+subjfolder
    testDataFolder = folder+'test'+subjfolder

    for folder in [trainFolder, testFolder,trainDataFolder,testDataFolder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    energyDf = pd.read_csv(lengyfolder)
    dataDf = pd.read_csv(rawDataFile)



    energyDf['ges']  = energyDf['feedingClass'] + energyDf['nonfeedingClass']
    # print(energyDf['ges'])
    engy_arr = energyDf['ges'].as_matrix()

    headtail_rpr = pointwise2headtail(engy_arr)

    # split_in_middle:

    i=0
    headtail_rpr[i,0] = 0
    while i <(shape(headtail_rpr)[0])-1:
    	headtail_rpr[i,1] =int(headtail_rpr[i,1]+headtail_rpr[i+1,0])/2
    	headtail_rpr[i+1,0] = headtail_rpr[i,1]
    	i = i + 1
    headtail_rpr[i,1] = len(energyDf)


    [train_set, test_set]= tt_split(headtail_rpr, 0.7)

    train_set= train_set[train_set[:,0].argsort()]
    test_set= test_set[test_set[:,0].argsort()]




    trainDf = pd.DataFrame()
    testDf = pd.DataFrame()

    for i in range(shape(train_set)[0]):
        trainDf = pd.concat([trainDf, energyDf.iloc[train_set[i,0]:train_set[i,1]]])

    for i in range(shape(test_set)[0]):
        testDf = pd.concat([testDf, energyDf.iloc[test_set[i,0]:test_set[i,1]]])

    # trainDf = trainDf.drop('Unnamed: 0',1)
    # testDf = testDf.drop('Unnamed: 0',1)

    trainDf.to_csv(trainFolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv")
    testDf.to_csv(testFolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv")




    trainDf = pd.DataFrame()
    testDf = pd.DataFrame()

    for i in range(shape(train_set)[0]):
        trainDf = pd.concat([trainDf, dataDf.iloc[2*train_set[i,0]:2*train_set[i,1]]])

    for i in range(shape(test_set)[0]):
        testDf = pd.concat([testDf, dataDf.iloc[2*test_set[i,0]:2*test_set[i,1]]])


    trainDf.to_csv(trainDataFolder+"testdata.csv", index = None)
    testDf.to_csv(testDataFolder+"testdata.csv", index = None)
