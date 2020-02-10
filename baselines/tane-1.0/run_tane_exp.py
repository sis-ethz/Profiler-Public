import os
import time
import numpy as np
import pandas as pd
import subprocess
import sys
import re
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

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
    'ttt': (10, 958, 10),
    'base_tlarge_rlarge_dlarge_n0_0_nlow': (65, 100000, 65),
    'base_tlarge_rlarge_dlarge_n0_1_nlow': (60, 100000, 60),
    'base_tlarge_rlarge_dlarge_n0_2_nlow': (58, 100000, 58),
    'base_tlarge_rlarge_dlarge_n0_3_nlow': (62, 100000, 62),
    'base_tlarge_rlarge_dlarge_n0_4_nlow': (60, 100000, 60),
    'base_tlarge_rlarge_dsmall_n0_0_nlow': (58, 100000, 58),
    'base_tlarge_rlarge_dsmall_n0_1_nlow': (63, 100000, 63),
    'base_tlarge_rlarge_dsmall_n0_2_nlow': (59, 100000, 59),
    'base_tlarge_rlarge_dsmall_n0_3_nlow': (54, 100000, 54),
    'base_tlarge_rlarge_dsmall_n0_4_nlow': (62, 100000, 62),
    'base_tlarge_rsmall_dlarge_n0_0_nlow': (10, 100000, 10),
    'base_tlarge_rsmall_dlarge_n0_1_nlow': (9, 100000, 9),
    'base_tlarge_rsmall_dlarge_n0_2_nlow': (10, 100000, 10),
    'base_tlarge_rsmall_dlarge_n0_3_nlow': (13, 100000, 13),
    'base_tlarge_rsmall_dlarge_n0_4_nlow': (11, 100000, 11),
    'base_tlarge_rsmall_dsmall_n0_0_nlow': (8, 100000, 8),
    'base_tlarge_rsmall_dsmall_n0_1_nlow': (11, 100000, 11),
    'base_tlarge_rsmall_dsmall_n0_2_nlow': (10, 100000, 10),
    'base_tlarge_rsmall_dsmall_n0_3_nlow': (13, 100000, 13),
    'base_tlarge_rsmall_dsmall_n0_4_nlow': (14, 100000, 14),
    'base_tsmall_rlarge_dlarge_n0_0_nlow': (56, 1000, 56),
    'base_tsmall_rlarge_dlarge_n0_1_nlow': (57, 1000, 57),
    'base_tsmall_rlarge_dlarge_n0_2_nlow': (60, 1000, 60),
    'base_tsmall_rlarge_dlarge_n0_3_nlow': (59, 1000, 59),
    'base_tsmall_rlarge_dlarge_n0_4_nlow': (61, 1000, 61),
    'base_tsmall_rlarge_dsmall_n0_0_nlow': (63, 1000, 63),
    'base_tsmall_rlarge_dsmall_n0_1_nlow': (66, 1000, 66),
    'base_tsmall_rlarge_dsmall_n0_2_nlow': (60, 1000, 60),
    'base_tsmall_rlarge_dsmall_n0_3_nlow': (60, 1000, 60),
    'base_tsmall_rlarge_dsmall_n0_4_nlow': (59, 1000, 59),
    'base_tsmall_rsmall_dlarge_n0_0_nlow': (12, 1000, 12),
    'base_tsmall_rsmall_dlarge_n0_1_nlow': (11, 1000, 11),
    'base_tsmall_rsmall_dlarge_n0_2_nlow': (14, 1000, 14),
    'base_tsmall_rsmall_dlarge_n0_3_nlow': (9, 1000, 9),
    'base_tsmall_rsmall_dlarge_n0_4_nlow': (9, 1000, 9),
    'base_tsmall_rsmall_dsmall_n0_0_nlow': (11, 1000, 11),
    'base_tsmall_rsmall_dsmall_n0_1_nlow': (10, 1000, 10),
    'base_tsmall_rsmall_dsmall_n0_2_nlow': (11, 1000, 11),
    'base_tsmall_rsmall_dsmall_n0_3_nlow': (10, 1000, 10),
    'base_tsmall_rsmall_dsmall_n0_4_nlow': (13, 1000, 13),
    'base_tlarge_rlarge_dlarge_n0_0_nhigh': (65, 100000, 65),
    'base_tlarge_rlarge_dlarge_n0_1_nhigh': (60, 100000, 60),
    'base_tlarge_rlarge_dlarge_n0_2_nhigh': (58, 100000, 58),
    'base_tlarge_rlarge_dlarge_n0_3_nhigh': (62, 100000, 62),
    'base_tlarge_rlarge_dlarge_n0_4_nhigh': (60, 100000, 60), 'base_tlarge_rlarge_dsmall_n0_0_nhigh': (58, 100000, 58), 'base_tlarge_rlarge_dsmall_n0_1_nhigh': (63, 100000, 63), 'base_tlarge_rlarge_dsmall_n0_2_nhigh': (59, 100000, 59), 'base_tlarge_rlarge_dsmall_n0_3_nhigh': (54, 100000, 54), 'base_tlarge_rlarge_dsmall_n0_4_nhigh': (62, 100000, 62), 'base_tlarge_rsmall_dlarge_n0_0_nhigh': (10, 100000, 10), 'base_tlarge_rsmall_dlarge_n0_1_nhigh': (9, 100000, 9), 'base_tlarge_rsmall_dlarge_n0_2_nhigh': (10, 100000, 10), 'base_tlarge_rsmall_dlarge_n0_3_nhigh': (13, 100000, 13), 'base_tlarge_rsmall_dlarge_n0_4_nhigh': (11, 100000, 11), 'base_tlarge_rsmall_dsmall_n0_0_nhigh': (8, 100000, 8), 'base_tlarge_rsmall_dsmall_n0_1_nhigh': (11, 100000, 11), 'base_tlarge_rsmall_dsmall_n0_2_nhigh': (10, 100000, 10), 'base_tlarge_rsmall_dsmall_n0_3_nhigh': (13, 100000, 13), 'base_tlarge_rsmall_dsmall_n0_4_nhigh': (14, 100000, 14), 'base_tsmall_rlarge_dlarge_n0_0_nhigh': (56, 1000, 56), 'base_tsmall_rlarge_dlarge_n0_1_nhigh': (57, 1000, 57), 'base_tsmall_rlarge_dlarge_n0_2_nhigh': (60, 1000, 60), 'base_tsmall_rlarge_dlarge_n0_3_nhigh': (59, 1000, 59), 'base_tsmall_rlarge_dlarge_n0_4_nhigh': (61, 1000, 61), 'base_tsmall_rlarge_dsmall_n0_0_nhigh': (63, 1000, 63), 'base_tsmall_rlarge_dsmall_n0_1_nhigh': (66, 1000, 66), 'base_tsmall_rlarge_dsmall_n0_2_nhigh': (60, 1000, 60), 'base_tsmall_rlarge_dsmall_n0_3_nhigh': (60, 1000, 60), 'base_tsmall_rlarge_dsmall_n0_4_nhigh': (59, 1000, 59), 'base_tsmall_rsmall_dlarge_n0_0_nhigh': (12, 1000, 12), 'base_tsmall_rsmall_dlarge_n0_1_nhigh': (11, 1000, 11), 'base_tsmall_rsmall_dlarge_n0_2_nhigh': (14, 1000, 14), 'base_tsmall_rsmall_dlarge_n0_3_nhigh': (9, 1000, 9), 'base_tsmall_rsmall_dlarge_n0_4_nhigh': (9, 1000, 9), 'base_tsmall_rsmall_dsmall_n0_0_nhigh': (11, 1000, 11), 'base_tsmall_rsmall_dsmall_n0_1_nhigh': (10, 1000, 10), 'base_tsmall_rsmall_dsmall_n0_2_nhigh': (11, 1000, 11), 'base_tsmall_rsmall_dsmall_n0_3_nhigh': (10, 1000, 10), 'base_tsmall_rsmall_dsmall_n0_4_nhigh': (13, 1000, 13)}

TANE_CMD = './bin/{} {} {} {} ./data/{}.dat {}'


def cmd_formater(cmd, dataset_name, func_name='taneg3', approx=0.3):
    args = dataset_arg[dataset_name]
    if approx < 0.3:
        print("Using approx {}".format(approx))
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
        print(result)
    except Exception as e:
        print("[Error]: Error occured in running cmd: {}".format(cmd))
        print("[Error]: Error msg: ", e)
        return str(e), -2 if 'kill' in e else -1
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
        if not os.path.exists(os.path.join('./data/', dataset_name + '.dat')):
            g.write("File not found error.\n")
        for fd in fd_list:
            g.write(fd)
            g.write('\n')
        if run_time > 0:
            g.write('run time: %f' % run_time)
        g.flush()
    return


def prepare_syn_tane_data(dataset_name, noise_level='nlow', csv_path='./original/'):

    print("get dataset", dataset_name)

    description = 'Umask = 007\nDataIn = ../original/{}.orig\nRemoveDuplicates = OFF\nAttributesOut = $BASENAME.atr\nStandardOut = ../data/$BASENAME.dat\
        \nSavnikFlachOut = ../data/$BASENAME.rel\nNOOFDUPLICATES = 1'

    csv_path = os.path.join(csv_path, noise_level)

    dataset_list = [
        dataset_name] if dataset_name != 'ALL' else os.listdir(csv_path)

    dataset_name_list = []

    for dataset in tqdm(dataset_list):

        if 'rsmall' not in dataset or 'n0_0' not in dataset:
            continue

        col_num = 0
        row_num = 0

        dataset_name = dataset.split('.')[0] + '_' + noise_level
        dataset_name_list.append(dataset_name)

        df = pd.read_csv(os.path.join(csv_path, dataset))
        col_num = len(df.columns.values)
        row_num = len(df.index.values)
        print('[Info]: getting {} rows and {} columns'.format(row_num, col_num))
        dataset_arg[dataset_name] = (col_num, row_num, col_num)

        if os.path.exists(os.path.join('./data/', dataset_name + '.dat')):
            continue

        # copy dataset to .orig files
        with open(os.path.join(csv_path, dataset), 'r') as f,\
                open(os.path.join(csv_path, '../', dataset_name + '.orig'), 'w') as g:
            whole_file = f.read()
            g.write('\n'.join(whole_file.split('\n')[1:]))
            f.close()
            g.close()

        # generate description file
        with open('./descriptions/{}.dsc'.format(dataset_name), 'w') as f:
            f.write(description.format(dataset_name))
            f.close()

        # generate col name file (.atr and .atr.name)
        with open('./descriptions/{}.atr'.format(dataset_name), 'w') as f, \
                open('./descriptions/{}.atr.name'.format(dataset_name), 'w') as g:
            f.write('\n'.join(df.columns.values))
            g.write('\n'.join(df.columns.values))
            f.close()
            g.close()

        # generate data using tane: select.perl
        os.system(
            'cd original && ../bin/select.perl ../descriptions/{}.dsc '.format(dataset_name))

    return dataset_name_list


if __name__ == '__main__':

    # print(len(dataset_names))

    # dataset_names = ['asia', 'cancer', 'alarm', 'australian', 'child',
    #                  'earthquake', 'hospital', 'mam', 'nypd', 'thoraric', 'ttt']

    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
        # for dataset in tqdm(dataset_names):
        if 'low' in dataset:
            approx = 0.01
        else:
            approx = 0.3
        cmd = cmd_formater(TANE_CMD, dataset, approx=approx)
        result, run_time = run_cmd(cmd)
        # print(result)
        result = get_fd_list(result)
        translated_fd_list = translate_fd_list(result, dataset)
        print("[Info]: FD from {}: {}".format(dataset, translated_fd_list))
        write_fd(translated_fd_list, dataset, run_time=run_time)
    else:
        # dataset_names = ['asia', 'cancer', 'alarm', 'australian', 'child',
        #                  'earthquake', 'hospital', 'mam', 'nypd', 'thoraric', 'ttt']
        dataset_names = prepare_syn_tane_data(
            'ALL', noise_level='nlow', csv_path='./original/')
        dataset_names += prepare_syn_tane_data(
            'ALL', noise_level='nhigh', csv_path='./original/')

        print(dataset_names)

        workers = 10
        pool = ThreadPool(workers)
        print("Created pool with %d workers" % workers)
        print("Start to run %d workers" % workers)
        results = pool.map(
            os.system, ['python3 {} {}'.format(sys.argv[0], ds) for ds in dataset_names])
        pool.close()
        pool.join()
        print("Finished !")
