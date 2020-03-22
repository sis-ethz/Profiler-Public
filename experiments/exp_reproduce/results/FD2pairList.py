import numpy as np

name = 'child'
# file_name = '../../../baselines/metanome-cli/results/BN/%s/%s_FDs.txt' % (name, name)
# out_name = '../../../baselines/metanome-cli/results/BN/%s/%s_FD_cnt_matrix.txt' % (name, name)

# file_name = '../../../baselines/fodiscovery/discoveringDependencies/results/BN/%s/FD_top-1-per-attr_a-0.3.txt'%(name)
# out_name = '../../../baselines/fodiscovery/discoveringDependencies/results/BN/%s/FD_top-1-per-attr_a-0.3_cnt_matrix.txt'%(name)

# file_name = './BN/%s/FDs_by_col.txt'%name
# out_name = './BN/%s/FDs_by_col_cnt_matrix.txt'%name

file_name = '../../../baselines/results/graphical-lasso/BN/%s/FDs.txt' % (name)
out_name = '../../../baselines/results/graphical-lasso/BN/%s/FDs_cnt_matrix.txt' % (name)

number_of_col = 20

pair_list = []

m = np.zeros([number_of_col, number_of_col])
col_index = {}
current_index = 0

with open(file_name) as f:
    for line in f:
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        if '(' in line:
            line = line[0 : line.find('(')]
        arr = line.split("->")
        frm = arr[0].split(",")
        to = arr[1]
        for deter in frm:
            # print("Found 1:")
            if deter in col_index:
                inx = col_index[deter]
            else:
                inx = current_index
                col_index[deter] = inx
                current_index += 1
            if to in col_index:
                iny = col_index[to]
            else:
                iny = current_index
                col_index[to] = iny
                current_index += 1
            m[inx,iny] += 1
            m[iny,inx] += 1
            if [deter, to] not in pair_list and [to, deter] not in pair_list:
                pair_list.append([deter, to])
                pair_list.append([to, deter])
                print("%s <-> %s" % (deter, to))

np.savetxt(out_name, m)
print("matrix saved")