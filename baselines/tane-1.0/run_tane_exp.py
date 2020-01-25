import os
import time
import numpy as np
import pandas as pd
import subprocess

dataset_arg = {
    'alarm': (),
    'asia': (),
    'australian': (),
    'cancer': (),
    'child': (),
    'earthquake': (),
    'hospital': (),
    'mam': (),
    'nypd': (),
    'thoraric': (),
    'ttt': ()
}

cmd = './bin/{} {} {} {} ./data/{}.dat {}'


def cmd_formater(cmd, dataset_name, func_name='taneg3', approx=0.1):
    args = dataset_arg[dataset_name]
    return cmd.format(func_name, args[0], args[1], args[2], dataset_name, approx)


def read_col_name(dataset_name):
    col_encode_name = {}
    with open('description/{}.atr'.format(dataset_name), 'r') as f_atr,\
            open('description/{}.atr.name'.format(dataset_name), 'r') as f_col_names:
        atr_line = f_atr.readline()
        col_line = f_col_names.readline()
        atr_line = atr_line.replace('\n', '')
        atr_line = atr_line.replace(' ', '')
        col_line = atr_line.replace('\n', '')
        col_line = atr_line.replace(' ', '')
        while (atr_line and col_line and len(col_line) >= 1 and len(atr_line) >= 1):
            col_encode_name[atr_line] = col_line
            atr_line = f_atr.readline()
            col_line = f_col_names.readline()
            atr_line = atr_line.replace('\n', '')
            atr_line = atr_line.replace(' ', '')
            col_line = atr_line.replace('\n', '')
            col_line = atr_line.replace(' ', '')
    return col_encode_name


def run_cmd(cmd):
    timeout_sec = 28800  # 8 hours
    try:
        start = time.time()
        result = subprocess.run(
            [cmd], stdout=subprocess.PIPE, timeout=timeout_sec)
        total_time = time.time() - start
        result = result.stdout.decode('utf-8')
    except Exception as e:
        print("[Error]: Error occured in running cmd: {}".format(cmd))
        print("[Error]: Error msg: ", e)
        return str(e), -1
    return result, total_time


def get_fd(cmd_result, dataset_name=None):
    # select the columns that contain fd
    cmd_res_list = cmd_result.split('\n')
    ret_fd_str = []
    for line in cmd_res_list:
        if '->' in line:
            ret_fd_str.append(line)
    return ret_fd_str


def translate_fd(fd_str_list, dataset_name):
    translated_fd_list = []
    col_dict = read_col_name(dataset_name)
    for fd_str in fd_str_list:
        if fd_str.startwith('->'):
            continue
        else:

    return translated_fd_str


if __name__ == '__main__':
    pass
