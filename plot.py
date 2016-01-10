# this is bad example do not use repeat 
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scipy import optimize
from pylab import *
# import pandas as pd
import scipy as sp

START_POINT = 70000
NUMBER = 1000
significant_figure = 5
PULSE_MIN_FIGURE = 690
ECG_MAX_FIGURE = 695
DIR_PATH = "/Volumes/WINDOWHDD/Windows/連続血圧計測/データ整理"

def calculate_min(x_f, y_f, x_min_f, y_min_f):
    for i in range(len(y_f) / PULSE_MIN_FIGURE):
        y_min_f_agent = y_f[i * PULSE_MIN_FIGURE:(i + 1) * PULSE_MIN_FIGURE - 1].min()
        x_min_f_agent = x_f[i * PULSE_MIN_FIGURE + y_f[i * PULSE_MIN_FIGURE:(i + 1) * PULSE_MIN_FIGURE - 1].argmin()]
        if y_min_f_agent != y_f[i * PULSE_MIN_FIGURE] and y_min_f_agent != y_f[(i + 1) * PULSE_MIN_FIGURE - 1]:
            x_min_f.append(x_min_f_agent)
            y_min_f.append(y_min_f_agent)

def calculate_max(x_f, y_f, x_max_f, y_max_f):
    for i in range(len(y_f) / ECG_MAX_FIGURE):
        y_max_f_agent = y_f[i * ECG_MAX_FIGURE:(i + 1) * ECG_MAX_FIGURE - 1].max()
        x_max_f_agent = x_f[i * ECG_MAX_FIGURE + y_f[i * ECG_MAX_FIGURE:(i + 1) * ECG_MAX_FIGURE - 1].argmax()]
        if y_max_f_agent != y_f[i * ECG_MAX_FIGURE] and y_max_f_agent != y_f[(i + 1) * ECG_MAX_FIGURE - 1]:
            x_max_f.append(x_max_f_agent)
            y_max_f.append(y_max_f_agent)

def calculate_and_save_figure(participant_name):
    print "started to calculate"
    pulse_data = np.genfromtxt(DIR_PATH + "/ecg pls data/" + "pulse_" + participant_name + ".csv", delimiter=",")
    pulse_data = pulse_data[START_POINT:(START_POINT + NUMBER), :]
    print "got pulse data"
    
    transposed_f = np.transpose(pulse_data)
    x_f = transposed_f[0]
    y_f = transposed_f[1]
    
    
    x_min_f = []
    y_min_f = []
    calculate_min(x_f, y_f, x_min_f, y_min_f)
    
    x_dash_f = []
    y_dash_f = []
    for i in range(len(y_f)):
        y_dash_f_agent = (y_f[i-1] - y_f[i]) / (x_f[i-1] - x_f[i])  
        x_dash_f_agent = x_f[i]
        x_dash_f.append(x_dash_f_agent)
        y_dash_f.append(y_dash_f_agent)
    x_dash_f = np.array(x_dash_f)
    y_dash_f = np.array(y_dash_f)
    x_dash_max_f = []
    y_dash_max_f = []
    x_dash_min_f = []
    y_dash_min_f = []
    calculate_max(x_dash_f, y_dash_f, x_dash_max_f, y_dash_max_f)
    calculate_min(x_dash_f, y_dash_f, x_dash_min_f, y_dash_min_f)
    
    
    x_dash_dash_f = []
    y_dash_dash_f = []
    for i in range(len(y_f) - 1):
        y_dash_dash_f_agent = (y_dash_f[i-1] - y_dash_f[i]) / (x_f[i-1] - x_f[i])
        x_dash_dash_f_agent = x_f[i]
        x_dash_dash_f.append(x_dash_dash_f_agent)
        y_dash_dash_f.append(y_dash_dash_f_agent)
    
    SMOOTH_N = 10 
    x_smooth = []
    y_smooth = []
    for i in range(len(y_dash_dash_f) - SMOOTH_N):
        y_smooth_agent = 0
        for j in range(SMOOTH_N):
            y_smooth_agent += y_dash_dash_f[i + j]
        x_smooth.append(x_f[i])
        y_smooth.append(y_smooth_agent / SMOOTH_N)
    
    x_smooth = np.array(x_smooth)
    y_smooth = np.array(y_smooth)
    x_smooth_max_f = []
    y_smooth_max_f = []
    x_smooth_min_f = []
    y_smooth_min_f = []
    calculate_max(x_smooth, y_smooth, x_smooth_max_f, y_smooth_max_f)
    calculate_min(x_smooth, y_smooth, x_smooth_min_f, y_smooth_min_f)
    
    
    plt.figure(figsize=(10, 9))
    plt.suptitle("Participant Name: " + participant_name + ", START_POINT: " + str(START_POINT))
    
    plt.subplot(5, 1, 2)
    plt.ylabel("Pulse wave")
    plt.plot(x_f, y_f)
    plt.scatter(x_min_f, y_min_f)
    
    
    plt.subplot(5, 1, 3)
    plt.ylabel("Pulse wave velocity")
    plt.plot(x_dash_f, y_dash_f)
    plt.scatter(x_dash_max_f, y_dash_max_f)
    plt.scatter(x_dash_min_f, y_dash_min_f)
    
    
    plt.subplot(5, 1, 4)
    plt.ylabel("Pulse wave acceleration")
    plt.plot(x_smooth, y_smooth)
    plt.scatter(x_smooth_max_f, y_smooth_max_f)
    plt.scatter(x_smooth_min_f, y_smooth_min_f)
    
    plt.xlabel("Time[s]")
    print "calculated pulse data"
    
    
    ecg_data = np.genfromtxt(DIR_PATH + "/ecg pls data/" + "ecg_" + participant_name + ".csv", delimiter=",")
    ecg_data = ecg_data[START_POINT:(START_POINT + NUMBER), :]
    print "got ecg data"

    transposed_f = np.transpose(ecg_data)
    x_ecg = transposed_f[0]
    y_ecg = transposed_f[1]
    x_ecg_max_f = []
    y_ecg_max_f = []
    calculate_max(x_ecg, y_ecg, x_ecg_max_f, y_ecg_max_f)
    x_ecg_max_diff = []
    
    x_new = []
    y_new = []
    for i in range(len(x_ecg_max_f) - 1):
        x_ecg_max_diff_agent = x_ecg_max_f[i + 1] - x_ecg_max_f[i]
        x_ecg_max_diff.append(x_ecg_max_diff_agent)
        if i == 0:
            x_new.append(x_ecg_max_f[i])
        else:
            for j in range(len(x_ecg)):
                if x_ecg[j] < x_ecg_max_f[i]:
                    x_new.append(x_ecg[j])
    print len(x_new)
    
    
    # for i in :
    #     for j in :
    # 
    #         x_new[i + 1] = x_new[i] + 
    #         y_new[i + 1] = y_new[i] + 
    
        
        
    
    
    
    plt.subplot(5, 1, 1)
    plt.ylabel("Electrocardiogram")
    plt.plot(x_ecg, y_ecg)
    plt.scatter(x_ecg_max_f, y_ecg_max_f)
    print "calculated ecg data"
    
    plt.savefig(DIR_PATH + "/figures4/" + participant_name + ".png")
    plt.close()
    print "saved figure"
    
    
    # plt.ion()

participant_names = ["E02", "E03", "E04", "E05", "E06", "E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15", "E19", "E20-4", "E24-2", "E27-2", "E28", "E29", "E31", "E32", "E33", "higashi", "kurosu", "maya", "moriya", "qian", "shomura", "suda", "tamura", "watanabe", "xiao", "yasumi"]

for number in range(len(participant_names)):
    calculate_and_save_figure(participant_names[number])
    
