import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

labels = ['FDX', 'GL', 'PYRO', 'TANE', 'CORDS', 'RFI(.3)', 'RFI(.5)', 'RFI(1)']

f1_dict = {
    'tlarge_rlarge_dlarge_nhigh': np.array([0.336, 0.207, 0.022, -1, 0.148, -1, -1, -1]),
    'tlarge_rlarge_dlarge_nlow': np.array([0.939, 0.514, 0.021, -1, 0.276, -1, -1, -1]),

    # 'tlarge_rsmall_dlarge_nhigh': np.array([0.333, 0.320, 0.163, 0.163, 0.400, -1, -1, -1]),
    'tlarge_rsmall_dlarge_nhigh': np.array([0.400, 0.320, 0.163, 0.163, 0.400, -1, -1, -1]),
    'tlarge_rsmall_dlarge_nlow': np.array([0.889, 0.250, 0.163, 0.163, 0.400, -1, -1, -1]),

    'tsmall_rlarge_dlarge_nhigh': np.array([0.451, 0.000, 0.024, -1, 0.024, -1, -1, -1]),
    'tsmall_rlarge_dlarge_nlow': np.array([0.884, 0.240, 0.024, -1, 0.029, -1, -1, -1]),

    'tsmall_rsmall_dlarge_nhigh': np.array([0.667, 0.091, 0.114, 0.114, 0.000, 0.667, 0.667, 0.667]),
    'tsmall_rsmall_dlarge_nlow': np.array([0.667, 0.320, 0.114, 0.114, 0.200, 0.667, 0.667, 0.571]),

    'tsmall_rsmall_dsmall_nhigh': np.array([
        0.800, 0.174, 0.070, 0.070, 0.500, 0.500, 0.364, 0.308]),
    'tsmall_rsmall_dsmall_nlow': np.array([
        0.800, 0.160, 0.070, 0.070, 0.500, 0.500, 0.364, 0.308])
}
# tlarge_rlarge_dlarge_nhigh = [0.336, 0.207, 0.022, 0, 0, 0, 0]
# tlarge_rlarge_dlarge_nlow = [0.939, 0.514, 0.021, 0, 0, 0, 0]

# tsmall_rlarge_dlarge_nhigh = [0.451, 0.000, 0.024, 0, 0, 0, 0]
# tsmall_rlarge_dlarge_nlow = [0.884, 0.240, 0.024, 0, 0, 0, 0]

# tsmall_rsmall_dlarge_nhigh = [0.667, 0.091, 0.114, 0.114, 0.667, 0.667, 0.667]
# tsmall_rsmall_dlarge_nlow = [0.667, 0.320, 0.114, 0.114, 0.667, 0.667, 0.571]

# tsmall_rsmall_dsmall_nhigh = [0.800, 0.174, 0.070, 0.070, 0.500, 0.364, 0.308]
# tsmall_rsmall_dsmall_nlow = [0.800, 0.160, 0.070, 0.070, 0.500, 0.364, 0.308]

# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]


def gen_fig(ds_name):
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars
    m_labels = [labels[i] if f1_dict[ds_name][i] >=
                0 else '-' for i in range(len(labels))]
    fig, ax = plt.subplots(frameon=False)
    f1_dict[ds_name][f1_dict[ds_name] < 0] = 0
    rects1 = ax.bar(x, f1_dict[ds_name], width)
    # rects2 = ax.bar(x + width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1-score')
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_title('t=%s r=%s d=%s n=%s' % (
        'large' if 'tlarge' in ds_name else 'small',
        'large' if 'rlarge' in ds_name else 'small',
        'large' if 'dlarge' in ds_name else 'small',
        'high' if 'nhigh' in ds_name else 'low'))
    # ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def autolabel(rects, m_labels):
        """Attach a text label above each bar in *rects*, displaying its height."""
        print(m_labels)
        for i in range(len(rects)):
            rect = rects[i]
            height = rect.get_height()
            ax.annotate('{}'.format(height if m_labels[i] != '-' else '-'),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, m_labels)
    # autolabel(rects2)
    fig.set_size_inches(5, 3)
    fig.tight_layout()
    plt.savefig('./figure/%s.pdf' % name, dpi=200)


for name in f1_dict.keys():
    gen_fig(name)
