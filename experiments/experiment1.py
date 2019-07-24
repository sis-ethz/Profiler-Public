import pandas as pd
import sys

sys.path.append("../")
from profiler.core import *
from profiler.app.od import *


def load_data(dataname):
    df = pd.read_csv('../datasets/OD/%s/meta_data/%s.original.csv'%(dataname, dataname))
    gt = pd.read_csv('../datasets/OD/%s/meta_data/%s.diff.csv'%(dataname, dataname))['ground.truth']
    gt_idx = gt.index.values[gt == 'anomaly']
    return df, gt_idx


def get_constraints(tol, df):
    pf = Profiler(workers=2, tol=tol, eps=0.05, embedtxt=False)
    pf.session.load_data(src=DF, df=df, check_param=False)
    pf.session.load_training_data(multiplier = None)
    pf.session.learn_structure(sparsity=0, nfer_order=True)
    parent_sets = pf.session.get_dependencies(score="fit_error")
    return pf, parent_sets


def write(f, dataname, method, t, tol, knn, size_neighbor, min_neighbor, structured, overall, combined,
          t1, t2, param, high_dim):
    f.write(("{},"*15+"\n").format(dataname, method, t, tol, knn, size_neighbor, min_neighbor,
                                  structured, overall, combined, t1, t2, t1+t2, param, high_dim))

def run_ocsvm(pf, tol, gt_idx, parent_sets, knn, size, f):
    neighbors = None
    for nu in [0.1, 0.2, 0.3, 0.4, 0.5]:
        param = "nu_{}".format(nu)
        detector = ScikitDetector(pf.session.ds.df, attr=pf.session.ds.dtypes,
                                  method="ocsvm", gt_idx=gt_idx,
                                  nu=nu, gamma='auto',
                                  tol=tol, knn=knn, neighbor_size=size)
        if neighbors is None:
            # parameter for od should not affect neighbors
            detector.neighbors = neighbors
        overall_time = detector.run_overall(parent_sets)
        detector.evaluate_overall()
        # run with different min_neighbors
        for min_neighbor in [10, 25, 50]:
            detector.min_neighbors = min_neighbor
            structured_time = detector.run_structured(parent_sets)
            # run with different t
            for t in [0, 0.01, 0.05, 0.1]:
                detector.evaluate_structured(t)
                write(f, "ocsvm", t, tol, knn, size, min_neighbor,
                      detector.eval['structured'],
                      detector.eval['overall'],
                      detector.eval['combined'],
                      overall_time, structured_time,
                      param)

def run_od(method, dataname, neighbors, pf, tol, gt_idx, parent_sets, knn, size, f, high_dim):
    #for method in ["ocsvm", "isf"]:
    for nu in [0.1, 0.5]:
        param = "nu_{}".format(nu)
        if method == "ocsvm":
            detector = ScikitDetector(pf.session.ds.df, attr=pf.session.ds.dtypes,
                                      method=method, gt_idx=gt_idx,
                                      nu=nu, gamma='auto',
                                      tol=tol, knn=knn, neighbor_size=size, high_dim=high_dim)
        elif method == "isf":
            detector = ScikitDetector(pf.session.ds.df, attr=pf.session.ds.dtypes,
                                      method=method, gt_idx=gt_idx,
                                      contamination=nu,
                                      tol=tol, knn=knn, neighbor_size=size, high_dim=high_dim)
        else:
            detector = ScikitDetector(pf.session.ds.df, attr=pf.session.ds.dtypes,
                                      method=method, gt_idx=gt_idx,
                                      contamination=nu,
                                      tol=tol, knn=knn, neighbor_size=size, high_dim=high_dim)
        # parameter for od should not affect neighbors
        detector.neighbors = neighbors
        overall_time = detector.run_overall(parent_sets)
        detector.evaluate_overall()
        # run with different min_neighbors
        for min_neighbor in [10, 50]:
            detector.min_neighbors = min_neighbor
            structured_time = detector.run_structured(parent_sets)
            # run with different t
            for t in [0, 0.01, 0.05, 0.1]:
                detector.evaluate_structured(t)
                write(f, dataname, method, t, tol, knn, size, min_neighbor,
                      detector.eval['structured'],
                      detector.eval['overall'],
                      detector.eval['combined'],
                      overall_time, structured_time,
                      param, high_dim)

def main():
    method = sys.argv[1]
    high_dim = int(sys.argv[2]) == 1
    filename = 'exp1/experiment1_{}'.format(method)
    if high_dim:
      filename += "_high_dim"
    result = open(filename+".csv", "a+")
    result.write("dataname,method,t,tol,knn,size_neighbor,min_neighbor,s_p,s_r,s_f1,o_p,o_r,o_f1,c_p,c_r,c_f1,"
                 "overall_runtime,structured_runtime,combined_runtime,param,high_dim\n")
    for dataname in ["yeast", "abalone"]:
        df, gt_idx = load_data(dataname)
        for tol in [1e-6, 1e-4, 1e-2]:
            pf, parent_sets = get_constraints(tol, df)
            for knn in [True, False]:
                if knn:
                    for size in [3,5,10]:
                        if tol==1e-6 and knn and size<=300 and dataname=="yeast":
                            continue
                        detector = OutlierDetector(df, neighbor_size=size, knn=knn)
                        run_od(method, dataname, detector.neighbors, pf, tol, gt_idx, parent_sets, knn, size, result, 
                          high_dim)
                else:
                    detector = OutlierDetector(df, tol=tol, knn=knn)
                    run_od(method, dataname, detector.neighbors, pf, tol, gt_idx, parent_sets, knn, size, result, 
                      high_dim)

    result.close()

if __name__ == "__main__":
    main()
