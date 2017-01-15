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
sys.path.append('C:/Users/szh702/Documents/FoodWatch/inlabStr/')
from stru_utils import *




winsize = 4
stride = 2

subjs = [ 'Dzung']# 
# subjects wo problem: 'Cao', 'Dzung', 'Eric', 'Jiapeng', 'JC', 'Rawan', 'Shibo'
# problem subj:  'Gleb', 'Will','Matt',
# File b'./subject/Will/feature/energy/engy_ori_win4_str2_labeled.csv' does not exist


for subj in subjs:

    subjfolder = subj + '/'
    folder = './subject/'
    allfeatDF = pd.DataFrame()

    featFolder = folder+subjfolder+"feature/"
    segFolder = folder+subjfolder+"segmentation/"
    rawDataFile =  folder+subjfolder+"testdata.csv"
    testDataFolder = segFolder+'test_data/'
    energyfolder = folder+subjfolder+"feature/energy/"
    allFeatFolder = folder+subjfolder+"feature/all_features/"    
    lengyfolder = energyfolder + "engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"

    trainFolder = './subject/train'+subjfolder+"feature/energy/"
    testFolder = './subject/test'+subjfolder+"feature/energy/"
    trainDataFolder = './subject/train'+subjfolder
    testDataFolder = './subject/test'+subjfolder

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
    print(train_set)
    print(test_set)

    train_set= train_set[train_set[:,0].argsort()]
    test_set= test_set[test_set[:,0].argsort()]




    trainDf = pd.DataFrame()
    testDf = pd.DataFrame()
    len_train = 0
    len_test = 0

    for i in range(shape(train_set)[0]):
        trainDf = pd.concat([trainDf, energyDf.iloc[train_set[i,0]:train_set[i,1]]])
        len_train = len_train + (train_set[i,1] - train_set[i,0])*2

    for i in range(shape(test_set)[0]):
        testDf = pd.concat([testDf, energyDf.iloc[test_set[i,0]:test_set[i,1]]])
        len_test = len_test + (test_set[i,1] - test_set[i,0])*2

    trainDf = trainDf.drop('Unnamed: 0',1)
    testDf = testDf.drop('Unnamed: 0',1)

    trainDf.to_csv(trainFolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv")
    testDf.to_csv(testFolder+"engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv")


    print(len_train)
    print(len_test)

    trainDf = pd.DataFrame()
    testDf = pd.DataFrame()

    for i in range(shape(train_set)[0]):
        trainDf = pd.concat([trainDf, dataDf.iloc[2*train_set[i,0]:2*train_set[i,1]]])
    for i in range(shape(test_set)[0]):
        testDf = pd.concat([testDf, dataDf.iloc[2*test_set[i,0]:2*test_set[i,1]]])

    trainDf.to_csv(trainDataFolder+"testdata.csv", index = None)
    testDf.to_csv(testDataFolder+"testdata.csv", index = None)