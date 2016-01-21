# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
import scipy.optimize as optimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
import sys
import os
import re

# Get file names
DATA_DIR = './csvs'
DATA_DIR_2 = './csvs_concat'
DATA_DIR_3 = './csvs_cut_by_time'
DATA_DIR_4 = './csvs_wo_noise'
DATA_DIR_5 = './csvs_normalized'
DATA_DIR_6 = './csvs_ecg_max_pls_min_ptt_pulse_wave_features'
DATA_DIR_7 = './csvs_of_ptt'

DATA_DIR_8 = './csvs_ketsuatsu_ptt'
DATA_DIR_9 = './csvs_b1_b2_rmse'
DATA_DIR_10 = './csvs_b1_b2_rmse_pwf'
DATA_DIR_11 = './csvs_bp_pwf'
# KETSUATSU_PTT_DIR = './ketsuatsu_ptt'
KETSUATSU_DIR = './ketsuatsu'
KETSUATSU_ORIGINAL_DIR = './ketsuatsu_backup_original'

FILES = []

for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        FILES.append(file)

FILE_COUNT = len(FILES)
PERSON_COUNT = len(FILES) / 2
logging.debug( FILES )
logging.debug( 'FILE_COUNT {0}'.format(FILE_COUNT) )

FILE_NUM = 0
# ECG_FILE = FILES[ FILE_NUM ]
# PLS_FILE = FILES[ PERSON_COUNT / 2 + FILE_NUM ]
ECG_FILE = None
PLS_FILE = None

# concatenated file
CONCAT_FILES = []
for file in os.listdir(DATA_DIR_2):
    if file.endswith('.csv'):
        CONCAT_FILES.append(file)

# cut by time
CUT_BY_TIME_FILES = []
for file in os.listdir(DATA_DIR_3):
    if file.endswith('.csv'):
        CUT_BY_TIME_FILES.append(file)

# no noise file 2
# WO_NOISE_FILES_2 = []
# for file in os.listdir(DATA_DIR_4):
#     if file.endswith('.csv'):
#         WO_NOISE_FILES_2.append(file)

# no noise file
WO_NOISE_FILES = []
for file in os.listdir(DATA_DIR_4):
    if file.endswith('.csv'):
        WO_NOISE_FILES.append(file)

NORMALIZED_FILES = []
for file in os.listdir(DATA_DIR_5):
    if file.endswith('.csv'):
        NORMALIZED_FILES.append(file)

FILES_6 = []
for file in os.listdir(DATA_DIR_6):
    if file.endswith('.csv'):
        FILES_6.append(file)

FILES_7 = []
for file in os.listdir(DATA_DIR_7):
    if file.endswith('.csv'):
        FILES_7.append(file)

KETSUATSU_FILES = []
for file in os.listdir(KETSUATSU_DIR):
    if file.endswith('.csv'):
        KETSUATSU_FILES.append(file)

FILES_8 = []
for file in os.listdir(DATA_DIR_8):
    if file.endswith('.csv'):
        FILES_8.append(file)
    
# Constants
TIME = 'Time[s]'
PLS = 'PLS'
ECG = 'ECG'
PLS_WITHOUT_NOISE = 'PLS(w/o)'
ECG_WITHOUT_NOISE = 'ECG(w/o)'
PLS_WITHOUT_NOISE_2 = 'PLS(w/o)2'
ECG_WITHOUT_NOISE_2 = 'ECG(w/o)2'
PLS_NORMALIZED = 'PLS(norm)'
ECG_NORMALIZED = 'ECG(norm)'
ECG_DIFF = 'ECG(Diff)'
ECG_MAX = 'ECG(Max)'
PLS_MIN = 'PLS(Min)'
PLS_MAX = 'PLS(Max)'
PTT = 'PTT'
BLOOD_PRESSURE = 'Blood Pressure'
BLOOD_PRESSURE_LOW = 'BP_L'
DATETIME = 'Datetime'
TIME_RANGE = 2000 #[ms]
PAUSE_TIME = 0.1 #[s]
DATA_COUNT = 20000 #[ms]

# CutByTimeControllerの定数
OFFSET_OF_MESUREMENT_TIME = 10 #[s]

# NoiseCancelerの定数
FFT_WINDOW_NUM = 100000 # TODO in fftでは2*n例えば1024や1024*2を用いると早い
ECG_CUTOFF_FREQUENCY = 40
PLS_CUTOFF_FREQUENCY = 10
PLS_SUPER_CUTOFF_FREQUENCY = 3.5
SAMPLING_FREQUENCY = 1000

# PulseFeature
PWA_B = 'PWA(b/a)'
PWA_C = 'PWA(c/a)'
PWA_D = 'PWA(d/a)'
PWA_E = 'PWA(e/a)'
TIME_B = 'TIME(b/a)'
TIME_C = 'TIME(c/a)'
TIME_D = 'TIME(d/a)'
TIME_E = 'TIME(e/a)'
K_VALUE = 'K-Value'
AGING_INDEX = 'AgingIndex'
AUGMENTATION_INDEX = 'AI'
# Logger setting
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class CSVReader:
    def __init__(self, data_dir=DATA_DIR, pls_file=PLS_FILE, ecg_file=ECG_FILE):
        self.data_dir = data_dir
        self.pls_file = pls_file
        self.ecg_file = ecg_file
        self.pls = None
        self.ecg = None
        self.start_datetime = None

    def read_pls(self):
        logging.debug('read pls csv')
        pls_raw = pd.read_csv(self.data_dir + '/' + self.pls_file, skiprows=np.arange(0, 11, 1), names=[TIME, PLS], index_col=TIME)
        logging.debug( pls_raw.head(n=10) )
        logging.debug( 'pls data num =  {0} rows'.format( pls_raw.shape[0] - 11) )
        logging.debug( 'equal about {0} minutes'.format( (pls_raw.shape[0] - 11) / 1000 / 60 ) )
        self.pls = pls_raw
        return pls_raw
    
    def read_ecg(self):
        logging.debug('read ecg csv')
        ecg_raw = pd.read_csv(self.data_dir + '/' + self.ecg_file, skiprows=np.arange(0, 11, 1), names=[TIME, ECG], index_col=TIME)
        logging.debug( ecg_raw.head(n=10) )
        logging.debug( 'ecg data num = {0} rows'.format( ecg_raw.shape[0] - 11) )
        logging.debug( 'equal about {0} minutes'.format( (ecg_raw.shape[0] - 11) / 1000 / 60 ) )
        self.ecg = ecg_raw
        return ecg_raw
    
    def get_start_datetime(self):
        pls_raw = pd.read_csv(self.data_dir + '/' + self.pls_file)
        start_datetime = pd.to_datetime(pls_raw.iloc[1][1])
        self.start_datetime = start_datetime
        return start_datetime

    # def get_total_sample_num(self):
    #     pls_raw = pd.read_csv(self.data_dir + '/' + self.pls_file)
    #     # total_sample_num = pls_raw.iloc[[2][1])
    #     total_sample_num = self.pls_raw.shape[0]
    #     self.total_sample_num = total_sample_num
    #     return total_sample_num

class ConcatenatedDataGetter:
    def __init__(self, filename, data_dir=DATA_DIR_3):
        self.data_dir = data_dir
        self.filename = filename
        self.data = None

    def read(self):
        data  = pd.read_csv(self.data_dir + '/' + self.filename, index_col=0)
        self.data = data
        return data

class DataReader:
    def __init__(self, data_dir, filename):
        self.data_dir = data_dir
        self.filename = filename
        self.data = pd.read_csv(data_dir + '/' + filename, index_col=0)

class DataFrameController:
    def __init__(self, csv_reader_object):
        self.csv_reader_object = csv_reader_object
        self.dataframe = None
        self.filename = re.sub(r'_p0|_p1|ECG_|PLS_', '', csv_reader_object.pls_file, count=2)

    def concat_pls_and_ecg(self):
        logging.debug('concat data on position')
        df = pd.concat([self.csv_reader_object.pls, self.csv_reader_object.ecg], axis=1)
        self.dataframe = df
        return df

    def create_datetime(self):
        logging.debug('create datetime columns')
        date_range = pd.date_range(self.csv_reader_object.start_datetime, periods=self.dataframe.shape[0], freq='ms')
        # date_range_series = pd.Series(date_range, name=DATETIME)
        # df = pd.concat([self.dataframe, date_range_series], axis=1, ignore_index=True)
        # self.dataframe = df
        self.dataframe.index = date_range
        return self.dataframe

class DataFrameController2:
    def __init__(self, csv_reader_object_0, csv_reader_object_1):
        self.csv_reader_object_0 = csv_reader_object_0
        self.csv_reader_object_1 = csv_reader_object_1
        self.dataframe_0 = None
        self.dataframe_1 = None
        self.big_dataframe = None
        self.filename = re.sub(r'_p0|_p1|ECG_|PLS_', '', csv_reader_object_0.pls_file, count=2)

    def concat_pls_and_ecg(self):
        logging.debug('concat data on position')
        dataframe_0 = pd.concat([self.csv_reader_object_0.pls, self.csv_reader_object_0.ecg], axis=1)
        dataframe_1 = pd.concat([self.csv_reader_object_1.pls, self.csv_reader_object_1.ecg], axis=1)
        self.dataframe_0 = dataframe_0
        self.dataframe_1 = dataframe_1
        return

    def concat_p0_and_p1(self):
        big_dataframe= pd.concat([self.dataframe_0, self.dataframe_1])
        self.big_dataframe = big_dataframe
        return big_dataframe

    def create_datetime(self):
        logging.debug('create datetime columns')
        logging.debug(self.csv_reader_object_0.start_datetime)
        date_range = pd.date_range(self.csv_reader_object_0.start_datetime, periods=self.big_dataframe.shape[0], freq='ms')
        # date_range_series = pd.Series(date_range, name=DATETIME)
        # df = pd.concat([self.dataframe, date_range_series], axis=1, ignore_index=True)
        # self.dataframe = df
        self.big_dataframe.index = date_range
        return self.big_dataframe

    def create_csv(self):
        self.big_dataframe.to_csv(DATA_DIR_2 + '/' + self.filename)
        return

class CutByTimeController:
    def __init__(self, data):
        self.data = data

    def cut_by_time(self, filename):
        date_string = pd.to_datetime(self.data.index[0]).strftime('%Y-%m-%d')
        ketsuatsu = pd.read_csv(KETSUATSU_DIR + '/' + filename, header=None)
        ketsuatsu.columns = [TIME, BLOOD_PRESSURE, BLOOD_PRESSURE_LOW]
        ketsuatsu[TIME] = date_string + ' ' + ketsuatsu[TIME]
        start_time_index = ketsuatsu[TIME].iloc[0]
        end_time_index = ketsuatsu[TIME].iloc[-1]
        # ketsuatsu.index = pd.DatetimeIndex(ketsuatsu[TIME])
        logging.debug('index: {0}'.format(start_time_index))
        logging.debug('index: {0}'.format(end_time_index))
        start_time = pd.to_datetime(start_time_index) - pd.tseries.offsets.Second(OFFSET_OF_MESUREMENT_TIME)
        end_time = pd.to_datetime(end_time_index) + pd.tseries.offsets.Second(OFFSET_OF_MESUREMENT_TIME)
        logging.debug('time: {0}'.format(start_time))
        logging.debug('time: {0}'.format(end_time))
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data[start_time:end_time].copy()
        logging.debug('data: {0}'.format(self.data.head(1)))
        logging.debug('data: {0}'.format(self.data.tail(1)))
        return self.data

    def create_csv(self, filename='default.csv'):
        self.data.to_csv(DATA_DIR_3 + '/' + filename)
        return


class FigureWriter: 
    def __init__(self, data):
        self.data = data

    def save_figure(self, filename='default.png'):
        self.data.plot()
        plt.savefig(filename)
        return True

class DataPlotter:
    def __init__(self, data):
        self.data = data

    def plot_now(self):
        logging.debug('plot start')
        plt.close()
        fig, ax = plt.subplots()
        self.data.iloc[300000:400000].plot(ax=ax)
        plt.draw()
        plt.pause(PAUSE_TIME)
        return


class RealTimeDrawer:
    def __init__(self, data, filename='NONE'):
        self.data = data
        self.filename = filename
        self.title = re.sub(r'ECG|PLS', '', filename)

    def draw_now(self):
        count = self.data.shape[0] / TIME_RANGE
        logging.debug(count)

        plt.close()
        fig, ax = plt.subplots()
        for i in np.arange(count):
            logging.debug(i)
            ax.clear()
            ax.set_title( self.title )
            self.data.iloc[ i*TIME_RANGE:(i+1)*TIME_RANGE].plot(ax=ax, legend=False)
            if ECG_MAX in self.data.columns:
                self.data[ECG_MAX].iloc[ i*TIME_RANGE:(i+1)*TIME_RANGE].plot(ax=ax, marker='v')
            if PLS_MIN in self.data.columns:
                self.data[PLS_MIN].iloc[ i*TIME_RANGE:(i+1)*TIME_RANGE].plot(ax=ax, marker='o')

            plt.draw()
            # if i == 1:
            #     plt.savefig(filename= DATA_DIR + 'sample.png')
            plt.pause(PAUSE_TIME)
        return

class NoiseCanceler:
    def __init__(self, data):
        self.data = data

    def cancel_noise(self, ecg_cutoff_frequency=ECG_CUTOFF_FREQUENCY, pls_cutoff_frequency=PLS_CUTOFF_FREQUENCY):
        # only low pass filter
        length = self.data.shape[0]
        num = length / FFT_WINDOW_NUM + 1
        ecg_without_noise = []
        pls_without_noise = []

        for i in np.arange(num):
            if i == num - 1:
                fft_frequency = np.fft.fftfreq(length - FFT_WINDOW_NUM * (num - 1), 1.0 / SAMPLING_FREQUENCY)
                ecg_fft = np.fft.fft(self.data[ECG].values[FFT_WINDOW_NUM * i:])
                pls_fft = np.fft.fft(self.data[PLS].values[FFT_WINDOW_NUM * i:])
            else:
                fft_frequency = np.fft.fftfreq(FFT_WINDOW_NUM, 1.0 / SAMPLING_FREQUENCY)
                ecg_fft = np.fft.fft(self.data[ECG].values[FFT_WINDOW_NUM * i:FFT_WINDOW_NUM * (i + 1)])
                pls_fft = np.fft.fft(self.data[PLS].values[FFT_WINDOW_NUM * i:FFT_WINDOW_NUM * (i + 1)])

            ecg_fft[ (fft_frequency > ecg_cutoff_frequency) | (fft_frequency < -ecg_cutoff_frequency) ] = 0
            pls_fft[ (fft_frequency > pls_cutoff_frequency) | (fft_frequency < -pls_cutoff_frequency) ] = 0
            ecg_without_noise.extend( np.real(np.fft.ifft(ecg_fft)).tolist() )
            pls_without_noise.extend( np.real(np.fft.ifft(pls_fft)).tolist() )
        self.data[ECG_WITHOUT_NOISE_2] = ecg_without_noise
        self.data[PLS_WITHOUT_NOISE_2] = pls_without_noise
        return 

    def cancel_noise_3(self):
        # by butterworth filter, but not that good to me
        SAMPLING_FREQUENCY = 1000
        NYQUIST_FREQUENCY = float(0.5 * SAMPLING_FREQUENCY)
        PASS_FREQUENCY = 0.5
        ECG_STOP_FREQUENCY = ECG_CUTOFF_FREQUENCY
        PLS_STOP_FREQUENCY = PLS_CUTOFF_FREQUENCY
        gpass = 3
        gstop = 40
        # normalize
        wp = PASS_FREQUENCY / NYQUIST_FREQUENCY
        wp_ecg = ECG_STOP_FREQUENCY / NYQUIST_FREQUENCY
        wp_pls = PLS_STOP_FREQUENCY / NYQUIST_FREQUENCY
        ws = 0 
        ws_ecg = (ECG_STOP_FREQUENCY + 1) / NYQUIST_FREQUENCY
        ws_pls = (PLS_STOP_FREQUENCY + 1) / NYQUIST_FREQUENCY
        n1, wn1 = signal.buttord([wp, wp_ecg], [ws, ws_ecg], gpass, gstop)
        # b1, a1 = signal.butter(n1, wn1, 'bandpass')
        b1, a1 = signal.butter(1, [wp, wp_ecg], btype='bandpass')
        n2, wn2 = signal.buttord([wp, wp_pls], [ws, ws_pls], gpass, gstop)
        # b2, a2 = signal.butter(n2, wn2, 'bandpass')
        b2, a2 = signal.butter(1, [wp, wp_pls], btype='bandpass')
        data1 = self.data[ECG].values
        data2 = self.data[PLS].values

        ecg_filtered = signal.lfilter(b1, a1, self.data[ECG].values)
        pls_filtered = signal.lfilter(b2, a2, self.data[PLS].values)
        self.data['ECG_FILTERD'] = ecg_filtered
        self.data['PLS_FILTERD'] = pls_filtered
        return self.data

    def cancel_noise_2(self):
        # with highpass and lowpass filter
        # 0.5Hzでハイパス, ECG: 40Hz, PLS: 10Hz, でローパス
        FFT_WINDOW_NUM = 100000 # TODO in fftでは2*n例えば1024や1024*2を用いると早い
        # FFT_WINDOW_NUMの値が悪いのでここで時間がかかることあり
        # TODO 窓関数buttord filter等を使う
        ECG_LOW_CUTOFF_FREQUENCY = 0.5
        PLS_LOW_CUTOFF_FREQUENCY = 0.5
        SAMPLING_FREQUENCY = 1000
        length = self.data.shape[0]
        num = length / FFT_WINDOW_NUM + 1
        ecg_without_noise = []
        pls_without_noise = []

        for i in np.arange(num):
            if i == num - 1:
                fft_frequency = np.fft.fftfreq(length - FFT_WINDOW_NUM * (num - 1), 1.0 / SAMPLING_FREQUENCY)
                ecg_fft = np.fft.fft(self.data[ECG].values[FFT_WINDOW_NUM * i:])
                pls_fft = np.fft.fft(self.data[PLS].values[FFT_WINDOW_NUM * i:])
            else:
                fft_frequency = np.fft.fftfreq(FFT_WINDOW_NUM, 1.0 / SAMPLING_FREQUENCY)
                ecg_fft = np.fft.fft(self.data[ECG].values[FFT_WINDOW_NUM * i:FFT_WINDOW_NUM * (i + 1)])
                pls_fft = np.fft.fft(self.data[PLS].values[FFT_WINDOW_NUM * i:FFT_WINDOW_NUM * (i + 1)])

            ecg_fft[ (fft_frequency > 0) & (fft_frequency < ECG_LOW_CUTOFF_FREQUENCY) ] = 0
            ecg_fft[ (fft_frequency < 0) & (fft_frequency > -ECG_LOW_CUTOFF_FREQUENCY) ] = 0
            pls_fft[ (fft_frequency > 0) & (fft_frequency < PLS_LOW_CUTOFF_FREQUENCY) ] = 0
            pls_fft[ (fft_frequency < 0) & (fft_frequency > -PLS_LOW_CUTOFF_FREQUENCY) ] = 0

            ecg_fft[ (fft_frequency > ECG_CUTOFF_FREQUENCY) | (fft_frequency < -ECG_CUTOFF_FREQUENCY) ] = 0
            pls_fft[ (fft_frequency > PLS_CUTOFF_FREQUENCY) | (fft_frequency < -PLS_CUTOFF_FREQUENCY) ] = 0
            ecg_without_noise.extend( np.real(np.fft.ifft(ecg_fft)).tolist() )
            pls_without_noise.extend( np.real(np.fft.ifft(pls_fft)).tolist() )
        self.data[ECG_WITHOUT_NOISE] = ecg_without_noise
        self.data[PLS_WITHOUT_NOISE] = pls_without_noise
        return 

    def create_csv(self, filename='default.csv'):
        self.data.to_csv(DATA_DIR_4 + '/' + filename)
        return

class DataNormalizer:
    def __init__(self, data):
        self.data = data

    def get_normalized(self):
        ecg_mean = self.data[ECG_WITHOUT_NOISE].mean()
        pls_mean = self.data[PLS_WITHOUT_NOISE].mean()
        ecg_std = self.data[ECG_WITHOUT_NOISE].std()
        pls_std = self.data[PLS_WITHOUT_NOISE].std()
        self.data[ECG_NORMALIZED] = ( self.data[ECG_WITHOUT_NOISE] - ecg_mean ) / ecg_std
        self.data[PLS_NORMALIZED] = ( self.data[PLS_WITHOUT_NOISE] - pls_mean ) / pls_std
        return self.data

    def create_csv(self, filename='default.csv'):
        self.data[[ECG_NORMALIZED, PLS_NORMALIZED]].to_csv(DATA_DIR_5 + '/' + filename)
        return

class PTTCalculator:
    def __init__(self, data):
        self.data = data

    def get_ecg_max_index(self):
        # ECGの逆位相は気にしない
        self.data[ECG_DIFF] = self.data[ECG_NORMALIZED].diff()

        # しきい値は上から97%の所 または 3%以下
        ecg_threshold_max = self.data[ECG_NORMALIZED].quantile(.98)
        ecg_threshold_min = self.data[ECG_NORMALIZED].quantile(.02)

        # 最大値の変曲点でかつ閾値以上 または 最小値の変曲点でかつ閾値以下
        if np.abs(ecg_threshold_max) > np.abs(ecg_threshold_min):
            mask_max = signal.argrelmax(self.data[ECG_NORMALIZED].values)[0]
            data_max = self.data[ECG_NORMALIZED][mask_max] 
            data_max = data_max[data_max > ecg_threshold_max]
            self.data[ECG_MAX] = data_max
        else: 
            mask_min = signal.argrelmin(self.data[ECG_NORMALIZED].values)[0]
            data_min = self.data[ECG_NORMALIZED][mask_min]
            data_min = data_min[data_min < ecg_threshold_min]
            self.data[ECG_MAX] = data_min

        logging.debug(self.data[ECG_MAX])
        return self.data

    def get_pls_min_index(self, pls_cutoff_frequency=PLS_SUPER_CUTOFF_FREQUENCY):
        # 3.5Hzでスムージングした曲線の最小値の+-200ms以内に最小値が存在する
        MIN_POINTS_WINDOW = 100 #+-100[ms]以内
        length = self.data.shape[0]
        num = length / FFT_WINDOW_NUM + 1
        pls_smooth = []
        logging.debug(num)
        for i in np.arange(num):
            logging.debug(i)
            if i == num - 1:
                pls_fftfreq = np.fft.fftfreq(length - FFT_WINDOW_NUM * (num - 1), 1.0 / SAMPLING_FREQUENCY)
                pls_fft = np.fft.fft( self.data[PLS_NORMALIZED].values[FFT_WINDOW_NUM * i:])
            else:
                pls_fftfreq = np.fft.fftfreq( FFT_WINDOW_NUM, 1.0/SAMPLING_FREQUENCY)
                pls_fft = np.fft.fft( self.data[PLS_NORMALIZED].values[FFT_WINDOW_NUM * i:FFT_WINDOW_NUM * (i + 1) ] )

            pls_fft[ (pls_fftfreq > pls_cutoff_frequency) | (pls_fftfreq < -pls_cutoff_frequency) ] = 0
            pls_smooth.extend( np.real(np.fft.ifft(pls_fft)) )
        self.data['PLS(smooth)'] = pls_smooth
        min_point_indexes_of_smooth = signal.argrelmin(self.data['PLS(smooth)'].values)[0]
        max_point_indexes_of_smooth = signal.argrelmax(self.data['PLS(smooth)'].values)[0]

        min_indexes = []
        max_indexes = []
        for index in min_point_indexes_of_smooth:
            if index > MIN_POINTS_WINDOW:
                min_index = self.data[PLS_NORMALIZED].iloc[index - MIN_POINTS_WINDOW:index + MIN_POINTS_WINDOW].argmin()
            else:
                min_index = self.data[PLS_NORMALIZED].iloc[0:index + MIN_POINTS_WINDOW].argmin()
            min_indexes.append(min_index)
        for index_2 in max_point_indexes_of_smooth:
            if index_2 > MIN_POINTS_WINDOW:
                max_index = self.data[PLS_NORMALIZED].iloc[index_2 - MIN_POINTS_WINDOW:index_2 + MIN_POINTS_WINDOW].argmax()
            else:
                max_index = self.data[PLS_NORMALIZED].iloc[0:index_2 + MIN_POINTS_WINDOW].argmax()
            max_indexes.append(max_index)
                

        min_indexes = pd.unique(min_indexes)
        max_indexes = pd.unique(max_indexes)
        self.data[PLS_MIN] = self.data[PLS_NORMALIZED][min_indexes]
        self.data[PLS_MAX] = self.data[PLS_NORMALIZED][max_indexes]
        logging.debug(self.data[PLS_MIN])
        logging.debug(self.data[PLS_MAX])
        return self.data

    def get_ptt(self):
        # TODO pd.DataFrame.instance.set_indexを使うともっと簡潔に書ける
        self.data['PTT(calculater)'] = np.nan
        self.data.ix[self.data[PLS_MIN].notnull(), 'PTT(calculater)'] = 1
        self.data.ix[self.data[ECG_MAX].notnull(), 'PTT(calculater)'] = 0
        ptt_calculator = pd.DataFrame(self.data['PTT(calculater)'].dropna())
        logging.debug(ptt_calculator)
        ptt_calculator_len = len(ptt_calculator)
        ptt_calculator['PTT(TIME)'] = pd.to_datetime(ptt_calculator.index)
        ptt_calculator['PTT(calculater-diff)'] = ptt_calculator['PTT(calculater)'].diff()

        ptt_calculator.index = np.arange(ptt_calculator_len)
        # ptt_calculator_diff = ptt_calculator['PTT(calculater)'][1:].values - ptt_calculator['PTT(calculater)'][:-1].values
        # logging.debug(len(ptt_calculator_diff))
        # logging.debug(len(ptt_calculator[1:]))
        # ptt_calculator['PTT(calculater)'][1:] = ptt_calculator_diff

        ptt_calculator_eq_1_index = ptt_calculator[ptt_calculator['PTT(calculater-diff)'] == 1].index.values
        ptt_calculator_eq_0_index = ptt_calculator_eq_1_index - 1
        ptt_calculator[PTT] = np.nan
        ptt_calculator[PTT][ptt_calculator_eq_1_index] = ptt_calculator['PTT(TIME)'][ptt_calculator_eq_1_index].values - ptt_calculator['PTT(TIME)'][ptt_calculator_eq_0_index].values
        # ptt_calculator.index = pd.Index(ptt_calculator['PTT(TIME)'])
        ptt_calculator.index = ptt_calculator['PTT(TIME)']
        not_null_ptt = ptt_calculator[PTT][ptt_calculator[PTT].notnull()]
        logging.debug('------------------\n')
        logging.debug(not_null_ptt)
        logging.debug('------------------\n')
        ptt_sec = not_null_ptt.dt.components.milliseconds / 1000

        logging.debug('------------------\n')
        logging.debug(ptt_sec)
        logging.debug('------------------\n')
        # ptt_milli = ptt_calculator[PTT].dt.components.milliseconds / 1000
        # ptt_calculator[PTT] = ptt_sec
        # self.data[PTT] = ptt_calculator[PTT]
        logging.debug('------------------\n')
        logging.debug(type(self.data.index))
        logging.debug(type(ptt_sec.index))
        logging.debug('------------------\n')
        self.data[PTT] = np.nan
        self.data.index = pd.to_datetime(self.data.index)
        logging.debug('------------------\n')
        logging.debug(type(self.data.index))
        logging.debug(type(ptt_sec.index))
        logging.debug('------------------\n')
        self.data[PTT] = ptt_sec

        # ptt_calculator['PTT(TIME)'] = pd.to_datetime(ptt_calculator.index)
        # ptt_calculator[PTT] = np.NaN
        # ptt_calculator[PTT][1:] = ptt_calculator['PTT(TIME)'][1:].values - ptt_calculator['PTT(TIME)'][:-1].values
        # self.data[PTT] = ptt_calculator[PTT]
        # x = self.data[PTT]
        # xxx = x[x.notnull()]
        # self.data[PTT] = xxx.dt.components.milliseconds / 1001
        # 0.01 < PTT < 0.2 のものは理論上不可能なので
        self.data[PTT] = self.data[PTT][self.data[PTT] < 0.2]
        self.data[PTT] = self.data[PTT][self.data[PTT] > 0.01]
        return self.data 

    def create_csv(self, filename='default.csv'):
        self.data[PTT].to_csv(DATA_DIR_7 + '/' + filename )

    def draw_now(self):
        num = 1200000
        rtd = RealTimeDrawer(self.data.iloc[num:(num+DATA_COUNT)], header=None)
        rtd.draw_now()

class CuffDataCleaner:
    def __init__(self):
        dict = {}
        length = len(KETSUATSU_FILES)
        for i in np.arange(length):
            ketsuatsu = pd.read_csv(KETSUATSU_ORIGINAL_DIR + '/' + KETSUATSU_FILES[i], header=None)
            dict[i] = ketsuatsu[1]
        self.data = pd.DataFrame(dict)

    def clean(self):
        self.data[((self.data - self.data[0:2].min()) > 50) & ((np.abs(self.data.diff()) > 50) | (np.abs(self.data.diff(2)) > 50))] = np.nan
        self.data[((self.data - self.data[0:2].min()) < -10) & (np.abs(self.data.diff()) > 10)] = np.nan
        # TODO 松浦のスチューデント化残差を用いた外れ値処理が素晴らしいので実装したい

    def create_csv(self, filename='default.csv'):
        length = len(KETSUATSU_FILES)
        for i in np.arange(length):
            ketsuatsu = pd.read_csv(KETSUATSU_DIR + '/' + KETSUATSU_FILES[i], header=None)
            ketsuatsu[1] = self.data[i]
            ketsuatsu.to_csv(KETSUATSU_DIR + '/' + KETSUATSU_FILES[i], header=None, index=False)

class PulseWaveFeatureGetter:
    def __init__(self, data, ketsuatsu):
        self.data = data
        self.ketsuatsu = ketsuatsu

    def get_pulse_wave_accelation(self):
        PULSE_WAVE_VELOCITY = 'pwv'
        PULSE_WAVE_ACCELATION = 'pwa'
        PWA_B = 'PWA(b/a)'
        PWA_C = 'PWA(c/a)'
        PWA_D = 'PWA(d/a)'
        PWA_E = 'PWA(e/a)'
        TIME_B = 'TIME(b/a)'
        TIME_C = 'TIME(c/a)'
        TIME_D = 'TIME(d/a)'
        TIME_E = 'TIME(e/a)'
        K_VALUE = 'K-Value'
        AGING_INDEX = 'AgingIndex'
        AUGMENTATION_INDEX = 'AI'
        
        self.data[PULSE_WAVE_VELOCITY] = self.data[PLS_NORMALIZED].diff()
        self.data[PULSE_WAVE_ACCELATION] = self.data[PULSE_WAVE_VELOCITY].diff()
        # self.data = self.data.set_index([np.arange(len(self.data)), pd.to_datetime(self.data.index)])
        self.data.index = pd.to_datetime(self.data.index)
        self.data['ID'] = np.arange(len(self.data))
        # self.data.index.names = ['position', 'date']
        argrelmax_points_index = signal.argrelmax(self.data[PULSE_WAVE_ACCELATION].values)
        argrelmin_points_index = signal.argrelmin(self.data[PULSE_WAVE_ACCELATION].values)

        ptt_calculator = PTTCalculator(self.data)
        ptt_calculator.get_ecg_max_index()
        ptt_calculator.get_pls_min_index()
        # ptt_calculator.get_ptt()
        self.data[ECG_MAX] = ptt_calculator.data[ECG_MAX]
        self.data[PLS_MIN] = ptt_calculator.data[PLS_MIN]

        # pls_min_points_index = self.data[PLS_MIN].dropna().index.get_level_values('position')
        # logging.debug(pls_min_points_index)
        # self.data[PTT] = ptt_calculator.data[PTT]

        # get ketsuatsu files
        self.ketsuatsu.index = pd.to_datetime(self.ketsuatsu.index)

        self.ketsuatsu[PWA_B] = np.nan
        self.ketsuatsu[PWA_C] = np.nan
        self.ketsuatsu[PWA_D] = np.nan
        self.ketsuatsu[PWA_E] = np.nan
        self.ketsuatsu[TIME_B] = np.nan
        self.ketsuatsu[TIME_C] = np.nan
        self.ketsuatsu[TIME_D] = np.nan
        self.ketsuatsu[TIME_E] = np.nan
        self.ketsuatsu[K_VALUE] = np.nan
        self.ketsuatsu[AGING_INDEX] = np.nan
        self.ketsuatsu[AUGMENTATION_INDEX] = np.nan

        logging.debug(self.data[ECG_MAX])
        for datetime_index in self.ketsuatsu.index:
            logging.debug('start for bun---------------------------')

            logging.debug('start {0}'.format(datetime_index))
            sec = pd.tseries.offsets.Second(30)
            ecg_maxes_of_ketsuatsu_window = self.data[(datetime_index - sec):datetime_index][self.data[ECG_MAX].notnull()]
            
            logging.debug(ecg_maxes_of_ketsuatsu_window)
            b_a_list = []
            b_a_point_list = []
            c_a_list = []
            c_a_point_list = []
            d_a_list = []
            d_a_point_list = []
            e_a_list = []
            e_a_point_list = []

            k_value_list = []
            augmentation_index_list = []

            logging.debug('start for bun---------------------------')

            for ecg_max_position in ecg_maxes_of_ketsuatsu_window['ID']:
                logging.debug('ecg_max_position {0}'.format(ecg_max_position))

                # get pulse wave accelation height ratio, time ratio
                logging.debug('x--------')
                ecg_max_window = np.arange(int(ecg_max_position), int(ecg_max_position) + 1000)
                argrel_max_points = np.intersect1d(ecg_max_window, argrelmax_points_index)
                argrel_min_points = np.intersect1d(ecg_max_window, argrelmin_points_index)
                logging.debug('y----------')

                logging.debug(argrel_max_points)
                logging.debug(argrel_min_points)
                logging.debug('a = {0}'.format(self.data.ix[argrel_max_points[0]]))
                logging.debug('c = {0}'.format(self.data.ix[argrel_max_points[1]]))
                logging.debug('e = {0}'.format(self.data.ix[argrel_max_points[2]]))
                logging.debug(argrel_min_points)
                logging.debug('b = {0}'.format(self.data.ix[argrel_min_points[0]]))
                logging.debug('d = {0}'.format(self.data.ix[argrel_min_points[1]]))

                a = self.data.ix[argrel_max_points[0]]
                a_point = argrel_max_points[0]
                c = self.data.ix[argrel_max_points[1]]
                c_point = argrel_max_points[1]
                e = self.data.ix[argrel_max_points[2]]
                e_point = argrel_max_points[2]
                # 経験的に1と2
                b = self.data.ix[argrel_min_points[1]]
                b_point = argrel_min_points[1]
                d = self.data.ix[argrel_min_points[2]]
                d_point = argrel_min_points[2]

                logging.debug(datetime_index)
                logging.debug(a_point)
                logging.debug(b_point)
                logging.debug(c_point)
                logging.debug(d_point)
                logging.debug(e_point)


                b_a_list.append( b['pwa'] / a['pwa'] )
                c_a_list.append( c['pwa'] / a['pwa'] )
                d_a_list.append( d['pwa'] / a['pwa'] )
                e_a_list.append( e['pwa'] / a['pwa'] )

                b_a_point_list.append( ( b_point - a_point ) / 1000.0 ) 
                c_a_point_list.append( ( c_point - a_point ) / 1000.0 )
                d_a_point_list.append( ( d_point - a_point ) / 1000.0 )
                e_a_point_list.append( ( e_point - a_point ) / 1000.0 )


                logging.debug('for k-value min_point {0}'.format(self.data[PLS_NORMALIZED][a_point]))
                logging.debug('same {0}'.format(self.data[PLS_MIN][a_point]))
                logging.debug('for k-value and max point {0}'.format(self.data[PLS_NORMALIZED][d_point]))
                logging.debug('for ai second point {0}'.format(self.data[PLS_NORMALIZED][e_point]))

                # 理論上は変曲点aがpulse waveの最小値
                # 変曲点dが最大値
                # 変曲点eがAIの変曲点
                k_value_list.append( self.data[PLS_NORMALIZED][a_point] / (-self.data[PLS_NORMALIZED][d_point] ))
                augmentation_index_list.append( self.data[PLS_NORMALIZED][e_point] / self.data[PLS_NORMALIZED][d_point] )
                logging.debug('z----------')

            b_a = pd.Series(b_a_list).median()
            c_a = pd.Series(c_a_list).median()
            d_a = pd.Series(d_a_list).median()
            self.ketsuatsu[PWA_B].loc[datetime_index] = b_a
            self.ketsuatsu[PWA_C].loc[datetime_index] = c_a
            self.ketsuatsu[PWA_D].loc[datetime_index] = d_a
            self.ketsuatsu[PWA_E].loc[datetime_index] = pd.Series(e_a_list).median()

            self.ketsuatsu[TIME_B].loc[datetime_index] = pd.Series(b_a_point_list).median()
            self.ketsuatsu[TIME_C].loc[datetime_index] = pd.Series(c_a_point_list).median()
            self.ketsuatsu[TIME_D].loc[datetime_index] = pd.Series(d_a_point_list).median()
            self.ketsuatsu[TIME_E].loc[datetime_index] = pd.Series(e_a_point_list).median()

            self.ketsuatsu[K_VALUE].loc[datetime_index] = pd.Series(k_value_list).median()
            self.ketsuatsu[AUGMENTATION_INDEX].loc[datetime_index] = pd.Series(augmentation_index_list).median()

            self.ketsuatsu[AGING_INDEX] = (-b_a + c_a + d_a)


        return self.data


    def max_min_point(self):
        x

    def get_pulse_wave_accelation_wave_height_raio(self):
        x

    def get_pulse_wave_accelation_wave_time_ratio(self):
        x

    def get_inflection_point_of_pulse_wave(self):
        x

    def get_k_value(self):
        x

    def get_augmentation_index(self):
        # inflection point の 一つ目 b / aに等しいので省略
        x

    def get_aging_index(self):
        # 後から計算するので省略
        x

    def get_area_volume_of_pulse_wave(self):
        # 比率に変換する方法を思いつかないので省略
        x

    def create_csv(self, filename='default.csv'):
        length = len(NORMALIZED_FILES)
        # for i in np.arange(length):
        #     self.data.to_csv(DATA_DIR_6 + '/' + filename)
        self.ketsuatsu.to_csv(DATA_DIR_6 + '/' + filename)

class FeatureRegression:
    def __init__(self):
        self.data = pd.read_csv(DATA_DIR_9 + '/' + 'b1_b2_rmse.csv')

    def get_static_pulse_wave_feature(self):
        length = len(FILES_6)
        self.data[PWA_B] = np.nan
        self.data[PWA_C] = np.nan
        self.data[PWA_D] = np.nan
        self.data[PWA_E] = np.nan
        self.data[TIME_B] = np.nan
        self.data[TIME_C] = np.nan
        self.data[TIME_D] = np.nan
        self.data[TIME_E] = np.nan
        self.data[K_VALUE] = np.nan
        self.data[AGING_INDEX] = np.nan
        self.data[AUGMENTATION_INDEX] = np.nan
        for i in np.arange(length):
            static_pwfs = pd.read_csv(DATA_DIR_6 + '/' + FILES_6[i])
            static_pwf = static_pwfs.iloc[0:3].mean()
            self.data[PWA_B].iloc[i] = static_pwf[2]
            self.data[PWA_C].iloc[i] = static_pwf[3]
            self.data[PWA_D].iloc[i] = static_pwf[4]
            self.data[PWA_E].iloc[i] = static_pwf[5]
            self.data[TIME_B].iloc[i] = static_pwf[6]
            self.data[TIME_C].iloc[i] = static_pwf[7]
            self.data[TIME_D].iloc[i] = static_pwf[8]
            self.data[TIME_E].iloc[i] = static_pwf[9]
            self.data[K_VALUE].iloc[i] = static_pwf[10]
            self.data[AGING_INDEX].iloc[i] = static_pwf[11]
            self.data[AUGMENTATION_INDEX].iloc[i] = static_pwf[12]
        return self.data

    def create_csv(self, filename='default.csv'):
        self.data.to_csv(DATA_DIR_10 + '/' + 'b1_b2_pwf_rmse.csv')

class FeatureBloodPressureRegression:
    def __init__(self):
        x_data = pd.read_csv(DATA_DIR_6 + '/' + FILES_6[0])
        self.df = pd.DataFrame(index=x_data.columns)
        self.df = self.df.fillna(0)
        print 'here'

    def run(self):

        length = len(FILES_6)
        for i in np.arange(length):
            data = pd.read_csv(DATA_DIR_6 + '/' + FILES_6[i])
            data.corr().to_csv(DATA_DIR_11 + '/' + FILES_6[i])
            self.df[FILES_6[i]] = data.corr().iloc[1]
        self.df.to_csv(DATA_DIR_11 + '/' + 'correlation_of_sporting_pwf.csv')

    def get_ols(self):
        print 'here'



if __name__ == '__main__':
    logging.debug( 'PID {0}'.format( os.getpid() ) )
    def calculate_one_file(pls_file, ecg_file):
        logging.debug( 'calculation started')
        csv_reader = CSVReader(pls_file=pls_file, ecg_file=ecg_file)
        pls_raw = csv_reader.read_pls()
        ecg_raw = csv_reader.read_ecg()
        start_datetime = csv_reader.get_start_datetime()
    
        data_frame_controller = DataFrameController(csv_reader)
        data_frame_controller.concat_pls_and_ecg()
        data_frame_controller.create_datetime()
        df = data_frame_controller.dataframe
        logging.debug(df.head(20))
        logging.debug(df.describe())
    
        if DATA_COUNT and DATA_COUNT == 0:
            real_time_drawer = RealTimeDrawer(df)
        elif DATA_COUNT and DATA_COUNT > 0:
            logging.debug( 'DATA_COUNT: {0}'.format(DATA_COUNT) )
            real_time_drawer = RealTimeDrawer(df.head(DATA_COUNT), ecg_file)
        else:
            real_time_drawer = RealTimeDrawer(df)
        real_time_drawer.draw_now()
    
    
        logging.debug( 'calculation ended' )
        return df

    def concat_p0_and_p1_to_csv():
        for i in np.arange(0, PERSON_COUNT, 2):
            ecg_p0 = FILES[i]
            ecg_p1 = FILES[i + 1]
            pls_p0 = FILES[PERSON_COUNT + i]
            pls_p1 = FILES[PERSON_COUNT + i + 1]

            csv_reader_0 = CSVReader(pls_file=pls_p0, ecg_file=ecg_p0)
            csv_reader_0.read_ecg()
            csv_reader_0.read_pls()
            csv_reader_0.get_start_datetime()
            csv_reader_1 = CSVReader(pls_file=pls_p1, ecg_file=ecg_p1)
            csv_reader_1.read_ecg()
            csv_reader_1.read_pls()
            csv_reader_1.get_start_datetime()
            data_frame_controller_2 = DataFrameController2(csv_reader_0, csv_reader_1)
            data_frame_controller_2.concat_pls_and_ecg()
            data_frame_controller_2.concat_p0_and_p1()
            data_frame_controller_2.create_datetime()
            data_frame_controller_2.create_csv()
        return


    def simulate_one_file():
        calculate_one_file(pls_file=FILES[0], ecg_file=FILES[PERSON_COUNT])

    def simulate_raw_files():
        for i in np.arange(0, PERSON_COUNT, 2):
            logging.debug( 'calculation started' )
            df = calculate_one_file(ecg_file=FILES[i + 1], pls_file=FILES[PERSON_COUNT + i + 1])

    def concat_raw_files():
        concat_p0_and_p1_to_csv()

    def create_cut_by_time_files():
        for file in CONCAT_FILES: 
            logging.debug('start to create cut by time {0}'.format(file))
            dr = DataReader(DATA_DIR_2, file)
            cbtc = CutByTimeController(dr.data)
            cbtc.cut_by_time(file)
            cbtc.create_csv(file)


    def create_noise_removed_files():
        for file in CUT_BY_TIME_FILES:
            logging.debug('creating low- and high- passed {0}'.format(file))
            dr = DataReader(DATA_DIR_3, file)
            nc = NoiseCanceler(dr.data)
            nc.cancel_noise_2()
            nc.create_csv(file)

    # def create_noise_removed_files_2():
    #     for file in WO_NOISE_FILES:
    #         logging.debug('creating highpassed {0}'.format(file))
    #         dr = DataReader(DATA_DIR_4_2, file)
    #         nc = NoiseCanceler(dr.data)
    #         nc.cancel_noise()
    #         nc.cancel_noise_2()
    #         nc.create_csv(file)

    def normalize_files():
        for file in WO_NOISE_FILES:
            logging.debug('creating normalized {0}'.format(file))
            dr = DataReader(DATA_DIR_4, file)
            dn = DataNormalizer(dr.data)
            dn.get_normalized()
            dn.create_csv(file)

    def simulate_normalized_files():
        for file in NORMALIZED_FILES:
            logging.debug('simulating {0}'.format(file))
            dr = DataReader(DATA_DIR_5, file)
            num = 1200000
            rtd = RealTimeDrawer(dr.data.iloc[num:(num+DATA_COUNT)], file)
            rtd.draw_now()

    def simulate_normalized_files_2():
        for file in FILES_6:
            logging.debug('simulating {0}'.format(file))
            dr = DataReader(DATA_DIR_6, file)
            num = 1200000
            rtd = RealTimeDrawer(dr.data.iloc[num:(num+DATA_COUNT)], file)
            rtd.draw_now()

    def simulate_one_ptt_calculator():
        dr = DataReader(DATA_DIR_6, FILES_6[1])
        pc = PTTCalculator(dr.data)
        pc.get_ecg_max_index()
        pc.get_pls_min_index()
        pc.get_ptt()
        pc.draw_now()

    def simulate_ptt_calculator():
        for file in FILES_6:
            logging.debug('simulating {0}'.format(file))
            dr = DataReader(DATA_DIR_6, file)
            pc = PTTCalculator(dr.data)
            pc.get_pls_min_index()
            pc.get_ecg_max_index()
            pc.get_ptt()
            pc.create_csv(file)

    def create_ptt_files():
        for file in NORMALIZED_FILES:
            logging.debug('simulating {0}'.format(file))
            dr = DataReader(DATA_DIR_5, file)
            pc = PTTCalculator(dr.data)
            pc.get_pls_min_index()
            pc.get_ecg_max_index()
            pc.get_ptt()
            pc.create_csv(file)
            # pc.draw_now()

    def plot_ptt_with_time():
        file_count = len(FILES_7)
        logging.debug('plot start')
        fig =  plt.figure(figsize=(15, 10))
        big_ax = fig.add_subplot(111)
        big_ax.set_title('PTT vs Time with sport')
        big_ax.set_ylabel('PTT[s]')
        big_ax.set_xlabel('Time with sport about 30 minutes')
        big_ax.set_xticklabels('')
        big_ax.set_yticklabels('')
        big_ax.set_xticklabels('')
        big_ax.set_yticklabels('')
        for i in np.arange(file_count):
            logging.debug('start to plot {0}'.format(FILES_7[i]))
            dr = DataReader(DATA_DIR_7, FILES_7[i])
            row = i / 4
            column = i % 4
            max_row = file_count / 4 + 1
            ax = fig.add_subplot(max_row, 4, i+1)
            dr.data.dropna().plot(linestyle='None', marker='o', ax=ax, legend=True, sharey=True, ylim=(0, 0.25))
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels('')
            # name = FILES_7[i].replace('.csv', '')
            name = re.search('\d*', FILES_7[i]).group(0)
            logging.debug(name)
            ax.legend([name], loc='upper right')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig('ptt_with_time.png')
        return 

    def clean_cuff_data():
        cdc = CuffDataCleaner()
        cdc.clean()
        cdc.create_csv()

    def get_ketsuatsu_and_ptt():
        file_count = len(FILES_7)
        for i in np.arange(file_count):
            filename = FILES_7[i]
            logging.debug('start {0}'.format(filename))
            ptt = DataReader(DATA_DIR_7, filename)
            date_string = pd.to_datetime(ptt.data.index[0]).strftime('%Y-%m-%d')
            na_dropped_ptt = ptt.data.dropna()
            na_dropped_ptt.index = pd.DatetimeIndex(na_dropped_ptt.index)
            # resampled_ptt = na_dropped_ptt.resample('s', how='mean')
            ketsuatsu = pd.read_csv(KETSUATSU_DIR + '/' + filename, header=None)
            ketsuatsu.columns = [TIME, BLOOD_PRESSURE, BLOOD_PRESSURE_LOW]
            ketsuatsu[TIME] = date_string + ' ' + ketsuatsu[TIME]
            ketsuatsu.index = pd.DatetimeIndex(ketsuatsu[TIME])
            ptt_list = []
            for datetime_index in ketsuatsu.index:
                # sec = pd.tseries.offsets.Second(2)
                sec = pd.tseries.offsets.Second(30)
                # PTTは血圧測定時の-30secのmedian, 日高、松浦論文にもあるが中央値はノイズに対する
                # 堅牢性が高いため
                ptt = na_dropped_ptt[(datetime_index - sec):(datetime_index)].median()[0]
                logging.debug(ptt)
                ptt_list.append(ptt)
            # ketsuatsu_ptt = pd.concat([ketsuatsu, resampled_ptt], axis=1, join='inner')
            # ketsuatsu_ptt.columns = [TIME, BLOOD_PRESSURE, BLOOD_PRESSURE_LOW, PTT]
            ketsuatsu[PTT] = ptt_list
            logging.debug(ketsuatsu)
            ketsuatsu[[PTT, BLOOD_PRESSURE]].to_csv(DATA_DIR_8 + '/' + filename)
            # ketsuatsu_ptt.plot(kind='scatter', x=BLOOD_PRESSURE, y=PTT)


    def plot_ketsuatsu_and_ptt():
        file_count = len(FILES_8)
        logging.debug('plot start')
        fig =  plt.figure(figsize=(15, 10))
        big_ax = fig.add_subplot(111)
        big_ax.set_title('Blood Pressure vs PTT')
        big_ax.set_xlabel('PTT[s]', fontsize=9)
        big_ax.set_ylabel('Blood Pressure[mmHg]')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xticklabels('')
        big_ax.set_yticklabels('')

        # 単回帰によるフィッティングをプロット&出力
        rmse_list_of_linear_regression = []
        id_list = []

        for i in np.arange(file_count):
            logging.debug('start to plot {0}'.format(FILES_8[i]))
            dr = DataReader(DATA_DIR_8, FILES_8[i])
            # polyfit
            clean_data = dr.data.dropna()
            x = clean_data[PTT]
            y = clean_data[BLOOD_PRESSURE]
            fit = np.polyfit(x, y, 1)
            fitx = np.poly1d(fit)
            z = []
            for j in np.arange(len(x.values)):
                z.append(y.values[j] - fitx(x.values[j]))
            # sample N is small number, so amount of freedom is minus 2
            zzz = np.sqrt(( (np.array(z)**2).sum() / (len(z)-2) ))
            rmse_list_of_linear_regression.append(zzz)
            id_list.append(re.search('\d*', FILES_8[i]).group(0))

            logging.debug(zzz)
            logging.debug(x)
            logging.debug(y)

            row = i / 4
            column = i % 4
            max_row = file_count / 4 + 1
            ax = fig.add_subplot(max_row, 4, i+1)
            dr.data.dropna().plot(kind='scatter', x=PTT, y=BLOOD_PRESSURE, ax=ax, legend=True) # , sharex=True, sharey=True)
            # , xlim=(0.05, 0.18), ylim=(95, 270))
            ax.plot(x, fitx(x), 'vr-')
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_yticklabels('')
            # name = FILES_8[i].replace('.csv', '')
            name = re.search('\d*', FILES_8[i]).group(0)
            logging.debug(name)
            ax.legend([name], loc='upper right')
        linear_rmse_df = pd.DataFrame({'ID': id_list, 'RMSE': rmse_list_of_linear_regression})
        linear_rmse_df = linear_rmse_df.set_index(['ID'])
        linear_rmse_df.to_csv(DATA_DIR_9 + '/' + 'single_linear_regression.csv')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig('blood_pressure_with_ptt.png')
        return rmse_list_of_linear_regression

    def get_b1_and_b2():
        file_count = len(FILES_8)

        fig =  plt.figure(figsize=(15, 10))
        big_ax = fig.add_subplot(111)
        big_ax.set_title('Blood Pressure vs PTT')
        big_ax.set_xlabel('PTT[s]', fontsize=9)
        big_ax.set_ylabel('Blood Pressure[mmHg]')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xticklabels('')
        big_ax.set_yticklabels('')

        def function_to_fit(x, b1, b2):
            return (1.0 / (x ** 2)) * b1 + b2

        def function_to_fit_2(params, x, y):
            b1 = params[0]
            b2 = params[1]
            residual = y - (b1 / x**2 + b2)
            return residual

        params_init = np.array([0.0, 0.0])
        b1_list = []
        b2_list = []
        rmse_list = []
        id_list = []

        # 提案式への回帰をプロット&出力
        for i in np.arange(file_count):
            dr = DataReader(DATA_DIR_8, FILES_8[i]) 
            data = dr.data.dropna()
            x = data[PTT]
            y = data[BLOOD_PRESSURE]

            # params_optimized, covariance = optimize.curve_fit(function_to_fit, x, y, p0=params_init)
            params_optimized, covariance = optimize.leastsq(function_to_fit_2, params_init, args=(x, y))
            logging.debug('params {0}'.format(params_optimized))
            logging.debug('covariance {0}'.format(covariance))

            z = []
            for j in np.arange(len(x)):
                z.append(function_to_fit_2(params_optimized, x[j], y[j]))
            # sample N is small number, so amount of freedom is minus 2
            rmse = np.sqrt((np.array(z) ** 2).sum() / (len(z)- 2))
            logging.debug('RMSE {0}'.format(rmse))

            b1_list.append(params_optimized[0])
            b2_list.append(params_optimized[1])
            rmse_list.append(rmse)
            id_list.append(re.search('\d*', FILES_8[i]).group(0))

            # y_fitted = function_to_fit(x, params_optimized[0], params_optimized[1])
            y_fitted = params_optimized[0] / x.values**2 + params_optimized[1]
            logging.debug('start to plot {0}'.format(FILES_8[i]))

            row = i / 4
            column = i % 4
            max_row = file_count / 4 + 1
            ax = fig.add_subplot(max_row, 4, i+1)
            dr.data.dropna().plot(kind='scatter', x=PTT, y=BLOOD_PRESSURE, ax=ax, legend=True) # , sharex=True, sharey=True)
            data = pd.DataFrame({'x': x, 'y_fitted': y_fitted})
            data = data.set_index(['x']).sort_index()
            logging.debug(data)
            ax.plot(data.index, data['y_fitted'], 'vr-')
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_yticklabels('')
            name = re.search('\d*', FILES_8[i]).group(0)
            logging.debug(name)
            ax.legend([name], loc='upper right')
        logging.debug(rmse_list)
        logging.debug(np.array(rmse_list).mean())
        b1_b2_rmse_df = pd.DataFrame({'ID': id_list, 'B1': b1_list, 'B2': b2_list, 'RMSE': rmse_list})
        b1_b2_rmse_df = b1_b2_rmse_df.set_index(['ID'])
        b1_b2_rmse_df.to_csv(DATA_DIR_9 + '/' + 'b1_b2_rmse.csv')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig('ptt_with_blood_pressure_fitted_b1_b2.png')
        return b1_b2_rmse_df

    def get_b1_and_b2_by_log_linear():
        file_count = len(FILES_8)

        fig =  plt.figure(figsize=(15, 10))
        big_ax = fig.add_subplot(111)
        big_ax.set_title('Blood Pressure vs PTT')
        big_ax.set_xlabel('PTT[s]', fontsize=9)
        big_ax.set_ylabel('Blood Pressure[mmHg]')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.set_xticklabels('')
        big_ax.set_yticklabels('')

        def function_to_fit(x, b1, b2):
            return (1.0 / (x ** 2)) * b1 + b2

        def function_to_fit_2(params, x, y):
            b1 = params[0]
            b2 = params[1]
            alpha = params[2]
            residual = y - (b1 * (x**alpha) + b2)
            return residual

        params_init = np.array([0.0, 0.0, 0.0])
        b1_list = []
        b2_list = []
        alpha_list = []
        rmse_list = []
        id_list = []

        # 提案式への回帰をプロット&出力
        for i in np.arange(file_count):
            dr = DataReader(DATA_DIR_8, FILES_8[i]) 
            data = dr.data.dropna()
            x = data[PTT].values
            y = data[BLOOD_PRESSURE].values

            # params_optimized, covariance = optimize.curve_fit(function_to_fit, x, y, p0=params_init)
            params_optimized, covariance = optimize.leastsq(function_to_fit_2, params_init, args=(x, y))
            logging.debug('params {0}'.format(params_optimized))
            logging.debug('covariance {0}'.format(covariance))

            z = []
            for j in np.arange(len(x)):
                z.append(function_to_fit_2(params_optimized, x[j], y[j]))
            # sample N is small number, so amount of freedom is minus 2
            # this time 3 params so minus 3
            rmse = np.sqrt((np.array(z) ** 2).sum() / (len(z)- 3))
            logging.debug('RMSE {0}'.format(rmse))

            b1_list.append(params_optimized[0])
            b2_list.append(params_optimized[1])
            alpha_list.append(params_optimized[2])
            rmse_list.append(rmse)
            id_list.append(re.search('\d*', FILES_8[i]).group(0))

            # y_fitted = function_to_fit(x, params_optimized[0], params_optimized[1])
            y_fitted = params_optimized[0] * (x**params_optimized[2]) + params_optimized[1]
            logging.debug('start to plot {0}'.format(FILES_8[i]))

            row = i / 4
            column = i % 4
            max_row = file_count / 4 + 1
            ax = fig.add_subplot(max_row, 4, i+1)
            dr.data.dropna().plot(kind='scatter', x=PTT, y=BLOOD_PRESSURE, ax=ax, legend=True) # , sharex=True, sharey=True)
            data = pd.DataFrame({'x': x, 'y_fitted': y_fitted})
            data = data.set_index(['x']).sort_index()
            logging.debug(data)
            ax.plot(data.index, data['y_fitted'], 'vr-')
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.set_xticklabels('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_yticklabels('')
            name = re.search('\d*', FILES_8[i]).group(0)
            logging.debug(name)
            ax.legend([name], loc='upper right')
        logging.debug(rmse_list)
        logging.debug(np.array(rmse_list).mean())
        b1_b2_rmse_df = pd.DataFrame({'ID': id_list, 'B1': b1_list, 'B2': b2_list, 'alpha': alpha_list, 'RMSE': rmse_list})
        b1_b2_rmse_df = b1_b2_rmse_df.set_index(['ID'])
        b1_b2_rmse_df.to_csv(DATA_DIR_9 + '/' + 'b1_b2_rmse_log_linear.csv')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig('ptt_with_blood_pressure_fitted_b1_b2_with_log_linear.png')
        return b1_b2_rmse_df



    def get_wave_form_features():
        length = len(NORMALIZED_FILES)
        for i in np.arange(length):
            dr = DataReader(DATA_DIR_5, NORMALIZED_FILES[i])
            ketsuatsu = pd.read_csv(DATA_DIR_8 + '/' + NORMALIZED_FILES[i], index_col=0)
            pwfg = PulseWaveFeatureGetter(dr.data, ketsuatsu)
            pwfg.get_pulse_wave_accelation()
            pwfg.create_csv(NORMALIZED_FILES[i])
        return pwfg

    def get_wave_form_features_1():
        length = len(NORMALIZED_FILES)
        dr = DataReader(DATA_DIR_5, NORMALIZED_FILES[0])
        ketsuatsu = pd.read_csv(DATA_DIR_8 + '/' + NORMALIZED_FILES[0], header=None)
        pwfg = PulseWaveFeatureGetter(dr.data, ketsuatsu)
        x = pwfg.get_pulse_wave_accelation()
        # pwfg.create_csv(NORMALIZED_FILES[0])
        return x


    # # in sequence, it is supposed that p0 and p1 file exist which means 
    # # data over 31 minutes and under 60 minutes
    # # so this case one file under 30 minutes is eliminated
    # simulate_raw_files()
    # concat_p0_and_p1_to_csv()
    # create_cut_by_time_files()
    # create_noise_removed_files()
    # normalize_files()
    # create_ptt_files()
    # plot_ptt_with_time()
    # clean_cuff_data()
    # get_ketsuatsu_and_ptt()
    # y = plot_ketsuatsu_and_ptt()
    # z = get_b1_and_b2()
    # y = get_b1_and_b2_by_log_linear()

    # x = get_wave_form_features()
    # fr = FeatureRegression()
    # x = fr.get_static_pulse_wave_feature()
    # FeatureBloodPressureRegression().run()


    # simulate_normalized_files()
    # simulate_ptt_calculator()

    # plot_ptt_with_time()
    # simulate_one_ptt_calculator()
    # zzz = DataReader(KETSUATSU_PTT_DIR, KETSUATSU_PTT_FILES[0])
