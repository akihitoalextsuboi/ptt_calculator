# import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
# import csv 
# 
# data = csv.reader(open('pulse_kurosu.csv', 'rb'), delimiter=',')
# data.plot()
# plt.show()

from pylab import *
import pandas as pd
import scipy

# import scipy.fftpack

# f = open("pulse_kurosu.csv", "r")
# f = f.head(10)
# f = pd.read_csv("pulse_kurosu.csv")
# f = f.head(10)
# xlist = list()
# ylist = list()
# for line in f:
#     s = line.split(",")
#     xx = s[0]
#     yy = s[1][:-1]
#     xlist.append(xx)
#     ylist.append(yy)
# f.close()
# 
# lines = plot(xlist, ylist, 'k')
# setp(lines, color='r', linewidth=2.0)
# xlabel('x')
# ylabel('y')
# show()
NUMBER = 5000
significant_figure = 5
MIN_FIGURE = 300
ECG_MAX_FIGURE = 600


def calculate_min(x_f, y_f, x_min_f, y_min_f):
    for i in range(len(y_f) / MIN_FIGURE):
        y_min_f_agent = y_f[(i - 1) * MIN_FIGURE:(i * MIN_FIGURE - 1)].min()
        x_min_f_agent = x_f[(i - 1) * MIN_FIGURE + y_f[(i - 1) * MIN_FIGURE:(i * MIN_FIGURE - 1)].argmin()]
        if y_min_f_agent != y_f[(i - 1) * MIN_FIGURE] and y_min_f_agent != y_f[i * MIN_FIGURE - 1]:
            x_min_f.append(x_min_f_agent)
            y_min_f.append(y_min_f_agent)

def calculate_max(x_f, y_f, x_max_f, y_max_f):
    for i in range(len(y_f) / ECG_MAX_FIGURE):
        y_max_f_agent = y_f[(i - 1) * ECG_MAX_FIGURE:(i * ECG_MAX_FIGURE - 1)].max()
        x_max_f_agent = x_f[(i - 1) * ECG_MAX_FIGURE + y_f[(i - 1) * ECG_MAX_FIGURE:(i * ECG_MAX_FIGURE - 1)].argmax()]
        if y_max_f_agent != y_f[(i - 1) * ECG_MAX_FIGURE] and y_max_f_agent != y_f[i * ECG_MAX_FIGURE - 1]:
            x_max_f.append(x_max_f_agent)
            y_max_f.append(y_max_f_agent)

f = pd.read_csv("data/pulse_xiao.csv")
f = f.head(NUMBER)
f = np.array(f)
transposed_f = np.transpose(f)
x_f = transposed_f[0]
y_f = transposed_f[1]
# y_min = optimize.fmin_bfgs(y_f, 0)

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


x_dash_dash_f = []
y_dash_dash_f = []
for i in range(len(y_f) - 1):
    y_dash_dash_f_agent = (y_dash_f[i-1] - y_dash_f[i]) / (x_f[i-1] - x_f[i])
    x_dash_dash_f_agent = x_f[i]
    x_dash_dash_f.append(x_dash_dash_f_agent)
    y_dash_dash_f.append(y_dash_dash_f_agent)

SMOOTH_N = 5 
x_smooth = []
y_smooth = []
for i in range(len(y_dash_dash_f) - SMOOTH_N):
    y_smooth_agent = 0
    for j in range(SMOOTH_N):
        y_smooth_agent += y_dash_dash_f[i + j]
    x_smooth.append(x_f[i])
    y_smooth.append(y_smooth_agent / SMOOTH_N)

x_smooth_max_f = []
y_smooth_max_f = []
# calculate_max(x_smooth, y_smooth, x_smooth_max_f, y_smooth_max_f)

y_dash_dash_f_fft = np.hstack((x_f, y_dash_dash_f))
y_dash_dash_f_fft = np.fft.fft(y_dash_dash_f_fft)
# y_dash_dash_f_fft = abs(y_dash_dash_f_fft)

plt.figure(figsize=(10, 9))
plt.title("Pulse wave")

plt.subplot(5, 1, 2)
plt.xlabel("Time[s]")
plt.ylabel("Pulse wave")
plt.plot(x_f, y_f)
plt.scatter(x_min_f, y_min_f)



# fft = np.fft.fft(f)
# fft = abs(fft)
# plt.plot(fft)
plt.subplot(5, 1, 3)
plt.xlabel("Time[s]")
plt.ylabel("Pulse wave velocity")
plt.plot(x_dash_f, y_dash_f)
plt.xlim(-1, 6)


plt.subplot(5, 1, 4)
plt.xlabel("Time[s]")
plt.ylabel("Pulse wave acceleration")
plt.plot(x_smooth, y_smooth)
plt.autoscale(tight=True)
plt.xlim(-1, 6)
plt.scatter(x_smooth_max_f, y_smooth_max_f)


f = pd.read_csv("data/ecg_xiao.csv")
f = f.head(NUMBER)
f = np.array(f)
transposed_f = np.transpose(f)
x_ecg = transposed_f[0]
y_ecg = transposed_f[1]
x_ecg_max_f = []
y_ecg_max_f = []
calculate_max(x_ecg, y_ecg, x_ecg_max_f, y_ecg_max_f)
x_ecg_max_diff = []
for i in range(len(x_ecg_max_f)):
    x_ecg_max_diff = x_ecg_max_f[i - 1] - x_ecg_max_f[i]
print x_ecg_max_diff


x_new = []
y_new = []
# for i in :
#     for j in :
# 
#         x_new[i + 1] = x_new[i] + 
#         y_new[i + 1] = y_new[i] + 

    
    



plt.subplot(5, 1, 1)
plt.ylabel("Electrocardiogram")
plt.plot(x_ecg, y_ecg)
plt.scatter(x_ecg_max_f, y_ecg_max_f)

plt.subplot(5, 1, 5)
plt.xlabel("Time[s]")

plt.ion()
# plt.plot(y_dash_dash_f_fft)
# plt.ylim(-50, 50)
# plt.show()
