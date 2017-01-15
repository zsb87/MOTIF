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
# from stru_utils import *


def genVideoSyncFile(featsfile, birthtime, video_sensor_bias_ms, r_df_test, save_flg, winsize, stride, syncfile):

    featDF = pd.read_csv(featsfile)
    featDF.index = range(featDF.shape[0])

    # r_df_test['Time'] = r_df_test.index

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
        raw_energy = raw_energy[['Time','unixtime','synctime','energy_acc_xyz','orientation_acc_xyz','energy_orientation',"energy_acc_xxyyzz",'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','feedingClass']]
        print("saving")
        print(syncfile)
        raw_energy.to_csv(syncfile)





save_flg = 1

# ------------------------------------------------------------------------------
# 
# import feature data
# 
# ------------------------------------------------------------------------------

# protocol = "inlabStr" 
protocol = "inlabUnstr" 
subj = 'Dzung'
subjfolder = subj

freq = 31
winsize = 4
stride = 2

featfolder = "../"+protocol+"/subject/"+subjfolder+"/feature/"
featsfile = featfolder+"energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled.csv"
birthfile = "../"+protocol+"/subject/"+subjfolder+'/desk video birth.txt'

birthtime = open(birthfile, 'r').read()   
print(birthtime)

rawfile = "../"+protocol+"/subject/"+subjfolder + "/testdata_labeled.csv"
r_df_test_label = pd.read_csv(rawfile)

syncfile =  "../"+protocol+"/subject/"+subjfolder+"/feature/energy/engy_ori_win"+str(winsize)+"_str"+str(stride)+"_sync1.csv"

video_sensor_bias_ms = 0

genVideoSyncFile(featsfile, birthtime, video_sensor_bias_ms, r_df_test_label, 1, winsize, stride, syncfile)
	


# featDF = pd.read_csv(file)
# featDF.index = range(featDF.shape[0])

# r_df_test['Time'] = r_df_test.index

# birthtime_s = birthtime[:19]
# import time
# base_unixtime = time.mktime(datetime.datetime.strptime(birthtime_s,"%Y-%m-%d %H:%M:%S").timetuple())
# base_unixtime = base_unixtime*1000 + video_sensor_bias_ms

# r_df_test["synctime"] = (r_df_test["synctime"] - base_unixtime + 0)/1000
# extr_idx = list(range(0,len(r_df_test)-winsize,stride))
# r_df_tDsample = r_df_test.iloc[extr_idx]

# r_df_tDsample.index = range(len(r_df_tDsample))

# raw_energy = pd.concat([featDF, r_df_tDsample], axis=1)

# if save_flg:
#     # raw_energy = raw_energy[['Time','unixtime','synctime','energy_acc_xyz','orientation_acc_xyz','energy_orientation',"energy_acc_xxyyzz",'Angular_Velocity_x','Angular_Velocity_y','Angular_Velocity_z','Linear_Accel_x','Linear_Accel_y','Linear_Accel_z','Class']]
#     raw_energy.to_csv(prepfolder + "raw_engy_ori_win"+str(winsize)+"_str"+str(stride)+"_labeled_time.csv")
#     enrg_class_activity = raw_energy[["energy_acc_xyz","Class","activity"]];
#     print(enrg_class_activity)
#     enrg_class_activity.dropna() 
#     enrg_class_activity.to_csv(prepfolder + "enrg_class_activity_win"+str(winsize)+"_str"+str(stride)+".csv")
