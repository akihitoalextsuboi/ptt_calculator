# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
import logging
import sys
import os
import re

# Get file names
DATA_DIR = './csvs'
DATA_DIR_2 = './csvs_concat'
DATA_DIR_3 = './csvs_cut_by_time'
DATA_DIR_4 = './csvs_wo_noise_2'
DATA_DIR_4_2 = './csvs_wo_noise'
DATA_DIR_5 = './csvs_normalized'
DATA_DIR_6 = './csvs_selected_and_ecg_corrected'
DATA_DIR_7 = './csvs_with_ptt'

KETSUATSU_PTT_DIR = './ketsuatsu_ptt'
KETSUATSU_DIR = './ketsuatsu'

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
WO_NOISE_FILES_2 = []
for file in os.listdir(DATA_DIR_4):
    if file.endswith('.csv'):
        WO_NOISE_FILES_2.append(file)

# no noise file
WO_NOISE_FILES = []
for file in os.listdir(DATA_DIR_4_2):
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

KETSUATSU_PTT_FILES = []
for file in os.listdir(KETSUATSU_PTT_DIR):
    if file.endswith('.csv'):
        KETSUATSU_PTT_FILES.append(file)
    
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
PTT = 'PTT'
BLOOD_PRESSURE = 'Blood Pressure'
BLOOD_PRESSURE_LOW = 'BP_L'
DATETIME = 'Datetime'
TIME_RANGE = 2000 #[ms]
PAUSE_TIME = 0.1 #[s]
DATA_COUNT = 20000 #[ms]

# NoiseCancelerの定数
FFT_WINDOW_NUM = 100000 # TODO in fftでは2*n例えば1024や1024*2を用いると早い
ECG_CUTOFF_FREQUENCY = 40
PLS_CUTOFF_FREQUENCY = 10
PLS_SUPER_CUTOFF_FREQUENCY = 3.5
SAMPLING_FREQUENCY = 1000

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

    def create_new_csv(self):
        self.big_dataframe.to_csv(DATA_DIR_2 + '/' + self.filename)
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

    def cancel_noise_2(self):
        # 0.5Hzでハイパス, ECG: 40Hz, PLS: 10Hz, でローパス
        FFT_WINDOW_NUM = 100000 # TODO in fftでは2*n例えば1024や1024*2を用いると早い
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

class PTTCalculater:
    def __init__(self, data):
        self.data = data

    def get_ecg_max_index(self):
        # これを改良すればECGの逆位相は気にしなくて良くなる
        # 一次微分の最大値の右側300[ms]以内
        self.data[ECG_DIFF] = np.append(0, self.data[ECG_NORMALIZED].values[1:] - self.data[ECG_NORMALIZED].values[:-1])
        # diff_max_index = signal.argrelmax(self.data[ECG_DIFF].values)[0] )
        # np.array diff_max_index.
        # しきい値は上から97%の所
        ecg_threshold = self.data.quantile(.97)[0]
        # しきい値以上またはしきい値以下
        # self.data[ECG_SUPER] = self.data[ECG_NORMALIZED]

        # 最大値の変曲点でかつ閾値以上
        data = self.data[ECG_NORMALIZED][ signal.argrelmax(self.data[ECG_NORMALIZED].values)[0] ]
        data = data[data > ecg_threshold]
        self.data[ECG_MAX] = data
        return self.data

    def get_pls_min_index(self, pls_cutoff_frequency=PLS_SUPER_CUTOFF_FREQUENCY):
        # 3.5Hzでスムージングした曲線の最小値の+-200ms以内に最小値が存在する
        MIN_POINTS_WINDOW = 200 #+-200[ms]以内
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
        min_indexes = []
        for index in min_point_indexes_of_smooth:
            if index > 100:
                min_index = self.data[PLS_NORMALIZED].iloc[index - 100:index + 100].argmin()
            else:
                min_index = self.data[PLS_NORMALIZED].iloc[0:index + 100].argmin()
            min_indexes.append(min_index)
        min_indexes = pd.unique(min_indexes)
        self.data[PLS_MIN] = self.data[PLS_NORMALIZED][min_indexes]
        logging.debug(self.data[PLS_MIN])
        return self.data

    def get_ptt(self):
        # TODO pd.DataFrame.instance.set_indexを使うともっと簡潔に書ける
        self.data['PTT(calculater)'] = np.NaN
        self.data.ix[self.data[PLS_MIN].notnull(), 'PTT(calculater)'] = 1
        self.data.ix[self.data[ECG_MAX].notnull(), 'PTT(calculater)'] = 0
        ptt_calculater = pd.DataFrame(self.data['PTT(calculater)'].dropna())
        logging.debug(ptt_calculater)
        ptt_calculater_len = len(ptt_calculater)
        ptt_calculater['PTT(TIME)'] = pd.to_datetime(ptt_calculater.index)
        ptt_calculater.index = np.arange(ptt_calculater_len)
        ptt_calculater_diff = ptt_calculater['PTT(calculater)'][1:].values - ptt_calculater['PTT(calculater)'][:-1].values
        logging.debug(len(ptt_calculater_diff))
        logging.debug(len(ptt_calculater[1:]))
        ptt_calculater['PTT(calculater)'][1:] = ptt_calculater_diff
        ptt_calculater_eq_1_index = ptt_calculater[ptt_calculater['PTT(calculater)'] == 1].index.values
        ptt_calculater_eq_0_index = ptt_calculater_eq_1_index - 1
        ptt_calculater[PTT] = np.nan
        ptt_calculater[PTT][ptt_calculater_eq_1_index] = ptt_calculater['PTT(TIME)'][ptt_calculater_eq_1_index].values - ptt_calculater['PTT(TIME)'][ptt_calculater_eq_0_index].values
        ptt_calculater.index = pd.Index(ptt_calculater['PTT(TIME)'])
        not_null_ptt = ptt_calculater[PTT][ptt_calculater[PTT].notnull()]
        logging.debug('------------------\n')
        logging.debug(not_null_ptt)
        logging.debug('------------------\n')
        ptt_sec = not_null_ptt.dt.components.milliseconds / 1000

        logging.debug('------------------\n')
        logging.debug(ptt_sec)
        logging.debug('------------------\n')
        # ptt_milli = ptt_calculater[PTT].dt.components.milliseconds / 1000
        # ptt_calculater[PTT] = ptt_sec
        # self.data[PTT] = ptt_calculater[PTT]
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

        # ptt_calculater['PTT(TIME)'] = pd.to_datetime(ptt_calculater.index)
        # ptt_calculater[PTT] = np.NaN
        # ptt_calculater[PTT][1:] = ptt_calculater['PTT(TIME)'][1:].values - ptt_calculater['PTT(TIME)'][:-1].values
        # self.data[PTT] = ptt_calculater[PTT]
        # x = self.data[PTT]
        # xxx = x[x.notnull()]
        # self.data[PTT] = xxx.dt.components.milliseconds / 1001
        self.data[PTT] = self.data[PTT][self.data[PTT] < 0.25]
        return self.data 

    def create_csv(self, filename='default.csv'):
        self.data[PTT].to_csv(DATA_DIR_7 + '/' + filename )

    def draw_now(self):
        num = 1200000
        rtd = RealTimeDrawer(self.data.iloc[num:(num+DATA_COUNT)])
        rtd.draw_now()

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
            data_frame_controller_2.create_new_csv()
        return

    def simulate_one_file():
        calculate_one_file(pls_file=FILES[0], ecg_file=FILES[PERSON_COUNT])

    def simulate_raw_files():
        for i in np.arange(0, PERSON_COUNT, 2):
            logging.debug( 'calculation started' )
            df = calculate_one_file(ecg_file=FILES[i + 1], pls_file=FILES[PERSON_COUNT + i + 1])

    def concat_raw_files():
        concat_p0_and_p1_to_csv()

    def create_noise_removed_files():
        for file in CONCAT_FILES:
            dr = DataReader(DATA_DIR_2, file)
            nc = NoiseCanceler(dr.data)
            nc.cancel_noise()
            nc.create_csv(file)

    def create_noise_removed_files_2():
        for file in WO_NOISE_FILES:
            logging.debug('creating highpassed {0}'.format(file))
            dr = DataReader(DATA_DIR_4_2, file)
            nc = NoiseCanceler(dr.data)
            nc.cancel_noise()
            nc.cancel_noise_2()
            nc.create_csv(file)

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

    def simulate_one_ptt_calculater():
        dr = DataReader(DATA_DIR_6, FILES_6[1])
        pc = PTTCalculater(dr.data)
        pc.get_ecg_max_index()
        pc.get_pls_min_index()
        pc.get_ptt()
        pc.draw_now()

    def simulate_ptt_calculater():
        for file in FILES_6:
            logging.debug('simulating {0}'.format(file))
            dr = DataReader(DATA_DIR_6, file)
            pc = PTTCalculater(dr.data)
            pc.get_pls_min_index()
            pc.get_ecg_max_index()
            pc.get_ptt()
            pc.create_csv(file)
            # mmpg.draw_now()

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
            name=FILES_7[i].replace('.csv', '')
            logging.debug(name)
            ax.legend([name], loc='upper right')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig('ptt_with_time.png')
        return 

    def get_ketsuatsu_and_ptt():
        file_count = len(FILES_7)
        for i in np.arange(file_count):
            filename = FILES_7[i]
            logging.debug('start {0}'.format(filename))
            ptt = DataReader(DATA_DIR_7, filename)
            date_string = pd.to_datetime(ptt.data.index[0]).strftime('%Y-%m-%d')
            na_dropped_ptt = ptt.data.dropna()
            na_dropped_ptt.index = pd.DatetimeIndex(na_dropped_ptt.index)
            resampled_ptt = na_dropped_ptt.resample('s', how='mean')
            ketsuatsu = pd.read_csv(KETSUATSU_DIR + '/' + filename, header=None)
            ketsuatsu.columns = [TIME, BLOOD_PRESSURE, BLOOD_PRESSURE_LOW]
            ketsuatsu[TIME] = date_string + ' ' + ketsuatsu[TIME]
            ketsuatsu.index = pd.DatetimeIndex(ketsuatsu[TIME])
            ketsuatsu_ptt = pd.concat([ketsuatsu, resampled_ptt], axis=1, join='inner')
            ketsuatsu_ptt.columns = [TIME, BLOOD_PRESSURE, BLOOD_PRESSURE_LOW, PTT]
            logging.debug(ketsuatsu_ptt)
            ketsuatsu_ptt[[PTT, BLOOD_PRESSURE]].to_csv(KETSUATSU_PTT_DIR + '/' + filename)
            # ketsuatsu_ptt.plot(kind='scatter', x=BLOOD_PRESSURE, y=PTT)

    def plot_ketsuatsu_and_ptt():
        file_count = len(KETSUATSU_PTT_FILES)
        logging.debug('plot start')
        fig =  plt.figure(figsize=(15, 10))
        big_ax = fig.add_subplot(111)
        big_ax.set_title('PTT vs Blood Pressure')
        big_ax.set_ylabel('PTT[s]')
        big_ax.set_xlabel('Blood Pressure[mmHg]')
        big_ax.set_xticks([])
        big_ax.set_xticklabels('')
        for i in np.arange(file_count):
            logging.debug('start to plot {0}'.format(KETSUATSU_PTT_FILES[i]))
            dr = DataReader(KETSUATSU_PTT_DIR, KETSUATSU_PTT_FILES[i])
            row = i / 4
            column = i % 4
            max_row = file_count / 4 + 1
            ax = fig.add_subplot(max_row, 4, i+1)


            if dr.data[PTT].any():
                # polyfit
                clean_data = dr.data[dr.data[BLOOD_PRESSURE].notnull() & dr.data[PTT].notnull()]
                x = clean_data[BLOOD_PRESSURE]
                y = clean_data[PTT]
                fit = np.polyfit(x, y, 2)
                fitx = np.poly1d(fit)
                logging.debug(x)
                logging.debug(y)
                dr.data.plot(kind='scatter', x=BLOOD_PRESSURE, y=PTT, ax=ax, legend=True, sharex=True, sharey=True)
                ax.plot(x, fitx(x), 'b-')
            ax.set_xlabel('')
            # ax.set_xticks([])
            # ax.set_xticklabels('')
            ax.set_ylabel('')
            # ax.set_yticks([])
            # ax.set_yticklabels('')
            name=FILES_7[i].replace('.csv', '')
            logging.debug(name)
            ax.legend([name], loc='upper right')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig('ptt_with_blood_pressure.png')


    # simulate_normalized_files_2()
    # simulate_max_min_point_getter()
    # simulate_ptt_calculater()

    # plot_ptt_with_time()
    # simulate_one_ptt_calculater()
    # get_ketsuatsu_and_ptt()
    # plot_ketsuatsu_and_ptt()
    zzz = DataReader(KETSUATSU_PTT_DIR, KETSUATSU_PTT_FILES[0])
