# Convert data from single battery dataset into three separate python dictionaries, for charging, discharging and impedance
# code is used from the following link: https://github.com/ABzry/battery_skunkworks/blob/master/nasa-battery-data-analysis.ipynb

import scipy.io as sio
from scipy.io import loadmat, whosmat
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import os


def process_file(mess):
    all_data = sio.loadmat(file_path)
    mat_data = {}
    for key in all_data.keys():
        if not key.startswith('__'):
            mat_data[key] = all_data[key]

    discharge, charge, impedance = {}, {}, {}

    xx = mat_data[key][0][0][0][0]

    for i, element in enumerate(xx):

        # checks if the step will be charge, discharge, impedance
        step = element[0][0]

        if step == 'discharge':
            discharge[str(i)] = {}
            discharge[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1) * 1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)

            discharge[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            discharge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
            discharge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
            discharge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
            discharge[str(i)]["current_load"] = data[0][0][3][0].tolist()
            discharge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
            discharge[str(i)]["time"] = data[0][0][5][0].tolist()
            discharge[str(i)]["capacity"] = data[0][0][6][0].tolist()


        if step == 'charge':
            charge[str(i)] = {}
            charge[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1) * 1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)

            charge[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            charge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
            charge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
            charge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
            charge[str(i)]["current_load"] = data[0][0][3][0].tolist()
            charge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
            charge[str(i)]["time"] = data[0][0][5][0].tolist()

        if step == 'impedance':
            impedance[str(i)] = {}
            impedance[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1) * 1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)

            impedance[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            impedance[str(i)]["sense_current"] = {}
            impedance[str(i)]["battery_current"] = {}
            impedance[str(i)]["current_ratio"] = {}
            impedance[str(i)]["battery_impedance"] = {}
            impedance[str(i)]["rectified_impedance"] = {}

            impedance[str(i)]["sense_current"]["real"] = np.real(data[0][0][0][0]).tolist()
            impedance[str(i)]["sense_current"]["imag"] = np.imag(data[0][0][0][0]).tolist()

            impedance[str(i)]["battery_current"]["real"] = np.real(data[0][0][1][0]).tolist()
            impedance[str(i)]["battery_current"]["imag"] = np.imag(data[0][0][1][0]).tolist()

            impedance[str(i)]["current_ratio"]["real"] = np.real(data[0][0][2][0]).tolist()
            impedance[str(i)]["current_ratio"]["imag"] = np.imag(data[0][0][2][0]).tolist()

            impedance[str(i)]["battery_impedance"]["real"] = np.real(data[0][0][3]).tolist()
            impedance[str(i)]["battery_impedance"]["imag"] = np.imag(data[0][0][3]).tolist()

            impedance[str(i)]["rectified_impedance"]["real"] = np.real(data[0][0][4]).tolist()
            impedance[str(i)]["rectified_impedance"]["imag"] = np.imag(data[0][0][4]).tolist()

            impedance[str(i)]["re"] = float(data[0][0][5][0][0])
            impedance[str(i)]["rct"] = float(data[0][0][6][0][0])

    # print(discharge, charge, impedance)

    return discharge, charge, impedance



# get data from folder
folder = '/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4'
filenames = [f for f in os.listdir(folder) if f.endswith('.mat')]


file_paths = [    '/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat',    '/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0006.mat',    '/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0007.mat', '/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0018.mat']

# empty dictionaries for all of data
all_discharge, all_charge, all_impedance = [], [], []
for file_path in file_paths:
    print(file_path)
    discharge, charge, impedance = process_file(file_path)
    all_discharge.append(discharge)
    all_charge.append(charge)
    all_impedance.append(impedance)


