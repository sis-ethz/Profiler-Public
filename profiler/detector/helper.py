from __future__ import division
from ..utility import FDtoDC, Timer, find_all_subsets
import logging
import numpy as np


logger = logging.getLogger('profiler.helper')
logger.setLevel(logging.INFO)


def process_heatmap(df, display=True, out_dc=False, outfile="dc.txt", column=False, undirect=False, 
                    k=6, topk=False, subsets=False, t0=0, above_threshold=False,
                    normalize=True, take_abs=False):
    df = df.copy()
    if take_abs:
        df = df.abs()
    # init
    timer = Timer()
    space = []
    out = None
    count = 0
    if out_dc:
        out = open(outfile,"w")
    attributes = df.columns.values
    
    # threshold
    threshold = 0
    if topk:
        n = len(attributes)
        threshold = (n-k)/(n*k)
    elif above_threshold:
        threshold = t0
    
    candidates = {}
    
    # helper function
    def add_candidate(left, right):
        c = {}
        c['left'] = left
        c['right'] = right
        space.append(c)
        if display:
            print("[{}] -> [{}]".format(",".join(left), right))
        if out_dc:
            out.write(FDtoDC(left,right))
    
    def add_pre_candidate(left, right):
        # helper method for undirect version to collapse all fd with same rhs
        if right not in candidates.keys():
            candidates[right] = set()
        if isinstance(left, str):
            candidates[right].add(left)
        elif isinstance(left, set):
            candidates[right] = candidates[right].union(left)
        else:
            left = set(left)
            candidates[right] = candidates[right].union(left)
        
    if not undirect:
            
        # traverse heatmap
        for cid, i in enumerate(attributes):
            if column:
                # normalize
                if normalize:
                    v = np.nan_to_num(df[i].values)
                    norm = np.linalg.norm(v)
                    if norm == 0:
                        continue
                    df[i] = v / norm
                # check columns
                row = df[i]
            else:
                # normalize
                if normalize:
                    v = np.nan_to_num(df.loc[i].values)
                    norm = np.linalg.norm(v)
                    if norm == 0:
                        continue
                    df.loc[i] = v / norm
                # check rows
                row = df.loc[i]
            # remove target
            target = i
            row[cid] = 0
            row = row.values
            if topk or above_threshold:
                fired = attributes[row >= threshold]
            else:
                fired = attributes[row != threshold]
            if len(fired) >= 1:
                if not subsets:
                    # add fired -> target
                    add_candidate(fired, target)
                    count += 1
                else:
                    for item in find_all_subsets(fired):
                        # add fired -> target
                        add_candidate(item, target)
                        count += 1
    else:
        # traverse heatmap
        for cid, i in enumerate(attributes):
            if column:
                # check columns
                row = df[i]
            else:
                # check rows
                row = df.loc[i]
            row[cid] = 0
            row = row.values
            if topk:
                fired = list(attributes[row >= threshold])
            else:
                fired = list(attributes[row != threshold])
            target = i
            if len(fired) >= 2:
                # add fired -> target
                if target in fired:
                    fired.remove(target)
                add_pre_candidate(fired, target)
                # add target -> fired
                for item in fired:
                    add_pre_candidate(target, item)
        # summarize results
        for right in candidates.keys():
            add_candidate(list(candidates[right]), right)
                
    dtime = timer.end()
    logger.info("Execution Time: {} Output {} Candidates".format(dtime, count))
    
    return space, dtime, count
