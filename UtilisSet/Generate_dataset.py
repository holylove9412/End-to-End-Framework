
import pandas as pd
import numpy as np
import math
import pickle
import networkx as nx
from swmm_api import swmm5_run,read_inp_file
from sklearn.preprocessing import MinMaxScaler
import shutil
import zarr
import pickle
import os
import matplotlib.pyplot as plt
import dask.array as da
from swmm_api import read_inp_file
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import swmmtoolbox.swmmtoolbox as swm
import yaml
from yaml.loader import SafeLoader
from datetime import datetime, timedelta,date
##防止中文乱码##
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
#####
load_dotenv()
swmm_executable_path = 'D:/EPASWMM5.1.015/swmm5.exe'

def conv_time(time):
    time /= 60
    hours = int(time)
    minutes = (time * 60) % 60
    return "%d:%02d" % (hours, minutes)
def altblocks(idf, dur, dt, dur_padding=60, multiplier=1):

    value_padding_pre = np.zeros(int(5 / dt))
    value_padding = np.zeros(int(dur_padding / dt))

    aPad = np.arange(dt, 5 + dt, dt)
    aPostPad = np.arange(dur_padding + 5 + dt, 5 + dur_padding + dur + dt, dt)

    aDur = np.arange(dt, dur + dt, dt)  # in minutes
    aInt = idf["A"] / (
        aDur ** idf["n"] + idf["B"]
    )  # idf equation - in mm/h for a given return period
    aDeltaPmm = np.diff(
        np.append(0, np.multiply(aInt, aDur / 60.0))
    )  # Duration: min -> hours
    aOrd = np.append(
        np.arange(1, len(aDur) + 1, 2)[::-1], np.arange(2, len(aDur) + 1, 2)
    )

    prec = np.asarray([aDeltaPmm[x - 1] for x in aOrd])

    aDur = aDur + 5
    prec = np.concatenate((value_padding_pre, prec, value_padding)) * multiplier
    aDur = np.concatenate((aPad, aDur, aPostPad))

    prec_str = list(map(str, np.round(prec, 2)))

    aDur_time = list(map(conv_time, aDur))
    aDur_str = list(map(str, aDur_time))

    aAltBl = dict(zip(aDur_str, prec_str))

    return aAltBl
def chicago_rainstorm(idf, dur, dt, dur_padding=60):
    value_padding = np.zeros(int(dur_padding / dt))

    aPostPad = np.arange(dur+dt, dur_padding + dur + dt, dt)
    rT = idf['r'] * dur
    r,b,n = idf['r'],idf['b'],idf['n']
    A = idf['a']*(1+idf['C']*math.log10(idf['P']))
    Ht = A * dur / ((dur + idf['b']) ** idf['n'])

    # 生成雨型数据
    aDur = np.arange(0, dur + dt, dt)
    cumulative_rain = []
    rain_intensity = []

    H_old = 0
    for t in aDur:
        if t <= rT:
            H = Ht * (r - (r - t / dur) / ((1 - t / (r * (dur + b))) ** n))
        else:
            H = Ht * (r + (t / dur - r) / ((1 + (t - dur) / ((1 - r) * (dur + b))) ** n))
        cumulative_rain.append(H)
        if t == 0:
            rain_intensity.append(0)
        else:
            rain_intensity.append((H - H_old) / dt)
        H_old = H
    aDeltaPmm = np.diff(
        np.append(0, np.multiply(cumulative_rain,60/dt)))
    prec = np.concatenate((aDeltaPmm, value_padding))
    aDur = np.concatenate((aDur, aPostPad))

    prec_str = list(map(str, np.round(prec, 2)))

    aDur_time = list(map(conv_time, aDur))
    aDur_str = list(map(str, aDur_time))

    aAltBl = dict(zip(aDur_str, prec_str))

    return aAltBl

def get_multiple_alt_blocks_rainfalls(n_rainfalls, dt,params_list=None, dur_padding=60, multiplier=1):
    rainfalls = []

    for i in range(n_rainfalls):
        rand_a = np.random.uniform(params_list['params_A'][0],params_list['params_A'][1])
        rand_C = np.random.uniform(params_list['params_C'][0],params_list['params_C'][1])
        rand_b = np.random.uniform(params_list['params_B'][0],params_list['params_B'][1])
        rand_n = np.random.uniform(params_list['params_n'][0],params_list['params_n'][1])
        rand_r = np.random.uniform(params_list['params_r'][0],params_list['params_r'][1])
        rand_P = np.random.randint(params_list['params_P'][0], params_list['params_P'][1])
        rand_dur = np.random.randint(params_list['range_D'][0], params_list['range_D'][1])

        random_idf = {"a":  rand_a, "C": rand_C,"P":rand_P, "b":rand_b,"n": rand_n,"r":rand_r}
        # random_idf = {"a":  9.58, "C": 0.846,"P":rand_P, "b":7,"n": 0.656,"r":0.405}

        # specs_rain = ["Synt_" + str(i), random_idf, rand_dur]
        # one_random_rain = altblocks(
        #     random_idf,
        #     dur=rand_dur,
        #     dt=dt,
        #     dur_padding=dur_padding,
        #     multiplier=multiplier,
        # )
        one_random_rain = chicago_rainstorm(random_idf,rand_dur,dt)
        # rainfalls.append((specs_rain, one_random_rain))
        rainfalls.append(one_random_rain)

    return rainfalls


def SwmmRun(inp_folder,inp_name):

    subprocess.run(
        [
            swmm_executable_path,
            inp_folder / f"{inp_name}.inp",
            inp_folder / f"{inp_name}.rpt",
            inp_folder /f"{inp_name}.out",
        ]
    )
def tabulator(list_of_strings):
    new_list = []
    for i in list_of_strings:
        new_list.append(i)
        new_list.append("\t")
    new_list[-1] = "\n"
    return new_list
def new_rain_lines_dat(
    rainfall_dict, name_new_rain="name_new_rain", day="1", month="1", year="2019"
):

    new_lines_rainfall = []

    for key, value in rainfall_dict.items():
        hour, minute = key.split(":")
        new_lines_rainfall.append(
            tabulator(
                [
                    name_new_rain,
                    year,
                    month,
                    str(int(day) + int((int(hour) / 24) % 24)),
                    str(int(hour) % 24),
                    minute,
                    str(value),
                ]
            )
        )  # STA01  2004  6  12  00  00  0.12

    return new_lines_rainfall
def get_new_rain_lines_real(
    rainfall_dict, name_new_rain="name_new_rain"
):  # , day='1', month = '1', year = '2019'):

    new_lines_rainfall = []

    for key, value in rainfall_dict.items():
        year = key[:4]
        month = key[4:6]
        day = key[6:8]
        hour, minute = key[8:10], key[10:12]
        new_lines_rainfall.append(
            tabulator([name_new_rain, year, month, day, hour, minute, str(value)])
        )  # STA01  2004  6  12  00  00  0.12

    return new_lines_rainfall

# Formatter for rain text lines --------------------------------------------------------------------------------
def create_datfiles(rainfalls, rainfall_dats_directory, identifier, isReal, offset=0):
    for idx, single_event in enumerate(rainfalls):

        if isReal:
            string_rain = [
                "".join(i)
                for i in get_new_rain_lines_real(single_event, name_new_rain="R1")
            ]
        else:
            string_rain = [
                "".join(i) for i in new_rain_lines_dat(single_event, name_new_rain="R1")
            ]

        filename = identifier + "_" + str(idx + offset) + ".dat"
        with open(rainfall_dats_directory / filename, "w") as f:
            f.writelines(string_rain)
def load_pickle(path):
    with open(path, "rb") as handle:
        variable = pickle.load(handle)
    return variable
def get_max_from_raindict(dict_rain):
    list_of_values = list(dict_rain.values())
    return np.array(list_of_values).max()
def generate_rainfalls():
    global rainfall_dats_directories,simulation_directories

    type_datasets = ["training", "validation", "testing"]

    rainfall_dats_directories = [
        swmm_data_folder / yaml_data["network"] / "rainfall_dats" / type_dataset
        for type_dataset in type_datasets
    ]

    simulation_directories = [
        swmm_data_folder / yaml_data["network"] / "simulations" / type_dataset
        for type_dataset in type_datasets
    ]

    for dir in rainfall_dats_directories + simulation_directories:
        if not dir.exists():
            dir.mkdir(parents=True)

    # ## Synthetic rainfalls
    # Alternating blocks method
    n_synthetic_rainfalls = [
        20,
        15,
        0,
    ]  # The number of synthetic rainfalls for training, validation and testing
    params_list = {"params_A": [5, 15], "params_C": [0.6, 1.6], "params_r": [0.35, 0.6],
                   "params_B": [5, 30], "params_n": [0.5, 1], "params_P": [5, 100],
                   "range_D": [30, 300]}
    c = 0
    for i in range(3):
        alternating_random_rainfalls = get_multiple_alt_blocks_rainfalls(
            n_synthetic_rainfalls[i],
            dt=1, params_list=params_list)
        create_datfiles(
            alternating_random_rainfalls,
            rainfall_dats_directories[i],
            identifier="synt",
            isReal=False,
            offset=c,
        )
        c += n_synthetic_rainfalls[i]


    ## Real rainfalls
    n_real_rains = [80, 15, 30]
    real_rainfalls_directory = swmm_data_folder /yaml_data["network"]/"real_rainfalls"
    pixel2 = load_pickle(real_rainfalls_directory / "events_pixel2_2014_5h.pk")

    rains_larger_than_ten = []
    # There are some rains that have None values, these are catched with the try.
    c = 0
    for i in range(3):
        rains_of_type = []
        while len(rains_of_type) < n_real_rains[i] and c <= 200 * (i + 1):
            try:
                max_rain = get_max_from_raindict(pixel2[c])
            except Exception as e:
                max_rain = 0
                print("error with rain #" + str(c))
            if max_rain > 10:
                rains_of_type.append(pixel2[c])
            c += 1
        rains_larger_than_ten.append(rains_of_type)

    print("There are", len(rains_larger_than_ten[2]), "rains larger than 10 mm")

    c = 0
    for i, rain_dir in enumerate(rainfall_dats_directories):
        create_datfiles(
            rains_larger_than_ten[i], rain_dir, identifier="real", isReal=True, offset=c
        )
        c += n_real_rains[i]

    return rainfall_dats_directories
def get_lines_from_textfile(path):
    with open(path, "r") as fh:
        lines_from_file = fh.readlines()
    return lines_from_file
def generate_dataset_inp(rain_dats_folder,simulation_folder):
    list_of_rain_events = os.listdir(rain_dats_folder)
    for event in list_of_rain_events:
        rain_event_path = rain_dats_folder / event
        inp = read_inp_file(inp_file)
        dat = get_lines_from_textfile(rain_event_path)
        splitted_line_dat = dat[0].split("\t")
        st_date = "".join(
            [
                splitted_line_dat[2],  # Month
                "/",
                splitted_line_dat[3],  # Day
                "/",
                splitted_line_dat[1],  # Year
            ]
        )
        st_date = datetime.strptime(st_date, "%m/%d/%Y").date()
        inp.OPTIONS['START_DATE']= st_date
        splitted_line_dat_last = dat[-1].split("\t")
        et_date_time = "".join(
            [
                splitted_line_dat_last[2],
                "/",
                splitted_line_dat_last[3],
                "/",
                splitted_line_dat_last[1],
                " ",
                splitted_line_dat_last[4],
                ":", splitted_line_dat_last[5], ":", "00"
            ]
        )

        # Convert new_date_time into a datetime object
        et_time_obj = datetime.strptime(et_date_time, "%m/%d/%Y %H:%M:%S")

        #add duration time
        et_time_obj += timedelta(hours=12)
        inp.OPTIONS['END_DATE'],inp.OPTIONS['END_TIME'] = et_time_obj.date(),et_time_obj.time()
        inp.OPTIONS['REPORT_START_DATE'] = st_date
        inp.RAINGAGES['R1'].filename=rain_event_path

        save_swmm_simulation = simulation_folder/event.replace(".dat", "")

        save_swmm_simulation.mkdir(parents=True, exist_ok=True)

        inp.write_file(save_swmm_simulation/'model.inp')

        shutil.copy(rain_event_path, save_swmm_simulation)

        subprocess.run(
            [
                swmm_executable_path,
                save_swmm_simulation / "model.inp",
                save_swmm_simulation / "model.rpt",
                save_swmm_simulation / "model.out",
            ]
        )
def load_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            yaml_data = yaml.load(f, Loader=SafeLoader)
    else:
        raise InvalidYAMLPathException
    return yaml_data
class InvalidYAMLPathException(Exception):
    pass
def save_SWMM_results(simulations_path):
    list_of_simulations = os.listdir(simulations_path)
    for sim in list_of_simulations:
        print("Extracting simulation", sim)
        c_simulation_folder_path = Path(simulations_path) / sim
        working_out = c_simulation_folder_path / "model.out"
        if not os.path.exists(c_simulation_folder_path / "hydraulic_head.zarr"):
            head_out_timeseries = swm.extract(working_out, ("node","","Hydraulic_head"))
            runoff_timeseries = swm.extract(working_out, ("subcatchment","","Runoff_rate"))
            head_out_timeseries = head_out_timeseries.to_dict(orient='index')
            runoff_timeseries =runoff_timeseries.to_dict(orient='index')
            with open(c_simulation_folder_path / "hydraulic_head.pkl",'wb') as f:
                pickle.dump(head_out_timeseries,f)
            with open(c_simulation_folder_path / "runoff.pkl", 'wb') as f:
                pickle.dump(runoff_timeseries, f)

            # zarr.save(c_simulation_folder_path / "hydraulic_head.zarr", head_out_timeseries)
            # zarr.save(c_simulation_folder_path / "runoff.zarr", runoff_timeseries)
            # head_out_timeseries.to_csv(c_simulation_folder_path / "hydraulic_head.csv")
            # runoff_timeseries.to_csv(c_simulation_folder_path / "runoff.csv")
        else:
            print("Already extracted")


if __name__ == '__main__':
    current_file = Path(__file__)
    current_dir = current_file.parent.parent
    swmm_data_folder = current_dir/'data'/'SWMM_data'
    yaml_folder = current_dir/'configs'
    yaml_name = "test.yaml"
    yaml_path = yaml_folder / yaml_name
    yaml_data = load_yaml(yaml_path)
    inp_file = swmm_data_folder/yaml_data["network"]/'networks'/(yaml_data["network"]+'.inp')
    generate_rainfalls()
    for i in range(3):
        generate_dataset_inp(rainfall_dats_directories[i],simulation_directories[i])
    for i in range(3):
        save_SWMM_results(simulation_directories[i])





