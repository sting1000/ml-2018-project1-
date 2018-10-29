from proj1_helpers import *
from implementations import *
import numpy as np

def load_categrized_data(data_name):
    """
    load csv data and categorize it with tag number. Standardize and return with x_cate, y_cate, ids_cate
    
    input:
        data_name   -the csv file name in the current dirctory
    output:
        x_cate, y_cate, ids_cate, total_num    -with four dimension
    """
    y, x, ids = load_csv_data(data_name)
    x[:, 0] = replace_nan(x[:, 0], -999)
    
    x_cate = dict()
    y_cate = dict()
    ids_cate = dict()
    for it in range(4):
        x_cate[it] = []
        y_cate[it] = []
        ids_cate[it] = []
    for it in range(4):
        x_cate[it] = x[x[:, 22]==it]
        y_cate[it] = y[x[:, 22]==it]
        ids_cate[it] = ids[x[:, 22]==it]
    for it in range(4):
        t=0
        column = 0

        # Remove columns which have high correlation - refer from notebook ##
        # Delete these features since they are correlated with DER_deltaeta_jet_jet
        # and have cyclic correlation DER_lep_eta_centrality, DER_prodeta_jet_jet,
        # PRI_jet_subleading_eta, PRI_jet_subleading_phi, PRI_jet_subleading_pt
        x_cate[it] = np.delete(x_cate[it], 6, 1)
        x_cate[it] = np.delete(x_cate[it], 11, 1)
        x_cate[it] = np.delete(x_cate[it], 20, 1)
        x_cate[it] = np.delete(x_cate[it], 23, 1)
        x_cate[it] = np.delete(x_cate[it], 23, 1)

        while column < x_cate[it].shape[1]:
            if (x_cate[it][:, column]==-999).all() or (x_cate[it][:, column]==0).all():
                x_cate[it] = np.delete(x_cate[it], column, 1)
                column -= 1
            column+= 1
        #x_cate[it], _, _ = standardize(x_cate[it])
    return x_cate, y_cate, ids_cate, len(y)
