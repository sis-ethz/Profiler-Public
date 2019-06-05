import time
import pandas as pd
import itertools, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def visualize_heatmap(heatmap, title=None, filename="heatmap.png", save=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,8))
    snsplt = sns.heatmap(heatmap, ax=ax, cmap=sns.color_palette("RdBu_r", 1000), center=0)
    if title:
        snsplt.set_title(title)
    if save:
        snsplt.get_figure().savefig(filename, bbox_inches='tight')
    plt.show()


def find_all_subsets(S):
    subsets = [find_subsets(S, i) for i in range(1, len(S)+1)]
    return list(itertools.chain.from_iterable(subsets)) + [set()]

def find_subsets(S, m):
    '''

    :param S: a set
    :param m:
    :return: all subset of set S with size of m
    '''
    return list(itertools.combinations(S, m))


def FDtoDC(a, b):
    dc = "t1&t2"
    if isinstance(a, str):
        a = [a]
    for i in a:
        dc += "&EQ(t1.{},t2.{})".format(i,i)
    dc += "&IQ(t1.{},t2.{})".format(b,b)
    dc += "\n"
    return dc


class GlobalTimer(object):

    def __init__(self,log=True):
        self.log = log
        if log:
            self.time_log = []
        self.origin = time.time()
        self.start = self.origin

    def time_point(self,msg):
        curr = time.time()
        time_pt=curr-self.origin
        info = "[{time_pt}] {msg}\n".format(time_pt=time_pt,msg=msg)
        if self.log:
            self.time_log.append([time_pt,msg,0])
        logger.info(info)

    def time_start(self,msg):
        curr = time.time()
        time_pt=curr-self.origin
        info = "[{time_pt}] {msg} start\n".format(time_pt=time_pt,msg=msg)
        if self.log:
            self.time_log.append([time_pt,"{} start".format(msg),0])
        self.start = curr
        logger.info(info)

    def time_end(self,msg):
        curr = time.time()
        time_pt = curr-self.origin
        exec_time = curr - self.start
        info = "[{time_pt}] {msg} execution time: {t}\n".format(time_pt=time_pt, msg=msg, t=exec_time)
        if self.log:
            self.time_log.append([time_pt,"end: {}".format(msg),exec_time])
        self.start = curr
        logger.info(info)
        return exec_time

    def to_csv(self):
        log = pd.DataFrame(data=self.time_log, columns=['time_point', 'msg', 'execution_time'])
        log.to_csv("time_points.csv", index=False)

    def get_stat(self):
        return pd.DataFrame(data=self.time_log, columns=['time_point', 'msg', 'execution_time'])
