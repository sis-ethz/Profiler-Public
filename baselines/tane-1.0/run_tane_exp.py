import os
import time
import numpy as np
import pandas as pd
import subprocess
import sys
import re


dataset_arg = {
    'alarm': (37, 1000, 37),
    'asia': (8, 1000, 8),
    'australian': (15, 690, 15),
    'cancer': (5, 1000, 5),
    'child': (20, 1000, 20),
    'earthquake': (5, 1000, 5),
    'hospital': (19, 1000, 19),
    'mam': (6, 830, 6),
    'nypd': (17, 34382, 17),
    'thoraric': (17, 470, 17),
    'ttt': (10, 958, 10)
}

CMD = './bin/{} {} {} {} ./data/{}.dat {}'


def cmd_formater(cmd, dataset_name, func_name='taneg3', approx=0.3):
    args = dataset_arg[dataset_name]
    return cmd.format(func_name, args[0], args[1], args[2], dataset_name, approx)


def read_col_name(dataset_name):
    col_encode_name = {}
    cnt = 0
    with open('descriptions/{}.atr'.format(dataset_name), 'r') as f_atr,\
            open('descriptions/{}.atr.name'.format(dataset_name), 'r') as f_col_names:
        atr_line = f_atr.readline()
        cnt += 1
        col_line = f_col_names.readline()
        atr_line = atr_line.replace('\n', '')
        atr_line = atr_line.replace(' ', '')
        col_line = col_line.replace('\n', '')
        col_line = col_line.replace(' ', '')
        while (atr_line and col_line and len(col_line) >= 1 and len(atr_line) >= 1):
            col_encode_name[str(cnt)] = col_line
            cnt += 1
            atr_line = f_atr.readline()
            col_line = f_col_names.readline()
            atr_line = atr_line.replace('\n', '')
            atr_line = atr_line.replace(' ', '')
            col_line = col_line.replace('\n', '')
            col_line = col_line.replace(' ', '')
    return col_encode_name


def run_cmd(cmd):
    timeout_sec = 28800  # 8 hours
    try:
        start = time.time()
        result = subprocess.run(
            cmd.split(' '), stdout=subprocess.PIPE, timeout=timeout_sec)
        total_time = time.time() - start
        result = result.stdout.decode('utf-8')
    except Exception as e:
        print("[Error]: Error occured in running cmd: {}".format(cmd))
        print("[Error]: Error msg: ", e)
        return str(e), -1
    return result, total_time


def get_fd_list(cmd_result, dataset_name=None):
    # select the columns that contain fd
    cmd_res_list = cmd_result.split('\n')
    ret_fd_str_list = []
    for line in cmd_res_list:
        if '->' in line:
            line = re.sub("[\(\[].*", "", line).strip()
            ret_fd_str_list.append(line)
            print("[Info]: found raw fd str: \'%s\'" % (line))
    return ret_fd_str_list


def translate_fd_list(fd_str_list, dataset_name):
    translated_fd_list = []
    col_dict = read_col_name(dataset_name)
    for fd_str in fd_str_list:
        if str.startswith(fd_str, '->') or '->' not in fd_str:
            print('[Info]: ignore illegal fd string -- ', fd_str)
            continue
        else:
            determinants = fd_str.split('->')[0].strip().split(' ')
            dependents = fd_str.split('->')[1].strip().split(' ')
            for i in range(len(determinants)):
                determinants[i] = col_dict[determinants[i]]
            for j in range(len(dependents)):
                dependents[j] = col_dict[dependents[j]]
            translated_fd_list.append(
                '%s->%s' % (','.join(determinants), ','.join(dependents)))
    return translated_fd_list


def write_fd(fd_list, dataset_name, run_time=-1, dir='./results/'):
    with open(os.path.join(dir, dataset_name + '_fd.txt'), 'w') as g:
        for fd in fd_list:
            g.write(fd)
            g.write('\n')
        if run_time > 0:
            g.write('run time: %f' % run_time)
        g.flush()
    return


if __name__ == '__main__':
    dataset_names = ['asia', 'cancer', 'alarm', 'australian', 'child',
                     'earthquake', 'hospital', 'mam', 'nypd', 'thoraric', 'ttt']
    if len(sys.argv) > 1:
        dataset_names = [sys.argv[1]]

    for dataset in dataset_names:
        cmd = cmd_formater(CMD, dataset)
        result, run_time = run_cmd(cmd)
        print(result)
        result = get_fd_list(result)
        translated_fd_list = translate_fd_list(result, dataset)
        print("[Info]: FD from {}: {}".format(dataset, translated_fd_list))
        write_fd(translated_fd_list, dataset, run_time=run_time)
