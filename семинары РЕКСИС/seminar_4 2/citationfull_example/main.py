import numpy as np
from numba import jit
import torch

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math_support import compute_qr_factorization
from nnet import Sheaf_NNet
from sheaf_calculator import compute_neural_network_parameters
from data_loader import read_data, perform_ttv_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import ndcg_score

import torch_geometric

def main_citeseer():
    print('inside the main function')
    loss_data_train = np.load('./results_citeseer/loss_data_train_CiteSeer_realization_9_embed_dime_64_dimy_16.npy')[:-1, :]
    loss_data_ttest = np.load('./results_citeseer/loss_data_ttest_CiteSeer_realization_9_embed_dime_64_dimy_16.npy')[:-1, :]
    loss_data_valid = np.load('./results_citeseer/loss_data_valid_CiteSeer_realization_9_embed_dime_64_dimy_16.npy')[:-1, :]
    ndata = loss_data_train.shape[0]
    t = np.linspace(0, ndata - 1, ndata, dtype=np.int64) 
    plt.figure(figsize=(8, 6))
    plt.scatter(t, loss_data_train[:, -1], c='r', label='train')
    plt.scatter(t, loss_data_ttest[:, -1], c='b', label='ttest')
    plt.scatter(t, loss_data_valid[:, -1], c='g', label='valid')
    plt.legend()
    plt.grid()
    plt.show()
    print('loss_data_train[-1, -1] = ' + str(loss_data_train[-1, -1]))
    print('loss_data_ttest[-1, -1] = ' + str(loss_data_ttest[-1, -1]))
    print('loss_data_valid[-1, -1] = ' + str(loss_data_valid[-1, -1]))

    accs_train = 0.0
    accs_ttest = 0.0
    accs_valid = 0.0
    nreal = 10
    for kreal in range(nreal):
        loss_data_train = np.load('./results_citeseer/loss_data_train_CiteSeer_realization_' + str(kreal) + '_embed_dime_64_dimy_16.npy')[:-1, :]
        loss_data_ttest = np.load('./results_citeseer/loss_data_ttest_CiteSeer_realization_' + str(kreal) + '_embed_dime_64_dimy_16.npy')[:-1, :]
        loss_data_valid = np.load('./results_citeseer/loss_data_valid_CiteSeer_realization_' + str(kreal) + '_embed_dime_64_dimy_16.npy')[:-1, :]
        accs_train += loss_data_train[-1, -1] / nreal
        accs_ttest += loss_data_ttest[-1, -1] / nreal
        accs_valid += loss_data_valid[-1, -1] / nreal
    print('accs_train = ' + str(accs_train))
    print('accs_ttest = ' + str(accs_ttest))
    print('accs_valid = ' + str(accs_valid))
    return 0

def results_processing():
    print('results_processing')
    loss_data_train = np.load('./results_pubmed/loss_data_train_Cora_realization_1_embed_dime_64_dimy_16.npy')[:-1, :]
    loss_data_ttest = np.load('./results_pubmed/loss_data_ttest_Cora_realization_1_embed_dime_64_dimy_16.npy')[:-1, :]
    loss_data_valid = np.load('./results_pubmed/loss_data_valid_Cora_realization_1_embed_dime_64_dimy_16.npy')[:-1, :]
    ndata = loss_data_train.shape[0]
    t = np.linspace(0, ndata - 1, ndata, dtype=np.int64) 
    plt.figure(figsize=(8, 6))
    plt.scatter(t, loss_data_train[:, -1], c='r', label='train')
    plt.scatter(t, loss_data_ttest[:, -1], c='b', label='ttest')
    plt.scatter(t, loss_data_valid[:, -1], c='g', label='valid')
    plt.legend()
    plt.grid()
    plt.show()
    print('loss_data_train[-1, -1] = ' + str(loss_data_train[-1, -1]))
    print('loss_data_ttest[-1, -1] = ' + str(loss_data_ttest[-1, -1]))
    print('loss_data_valid[-1, -1] = ' + str(loss_data_valid[-1, -1]))

    accs_train = 0.0
    accs_ttest = 0.0
    accs_valid = 0.0
    nreal = 2
    for kreal in range(nreal):
        loss_data_train = np.load('./results_pubmed/loss_data_train_Cora_realization_' + str(kreal) + '_embed_dime_64_dimy_16.npy')[:-1, :]
        loss_data_ttest = np.load('./results_pubmed/loss_data_ttest_Cora_realization_' + str(kreal) + '_embed_dime_64_dimy_16.npy')[:-1, :]
        loss_data_valid = np.load('./results_pubmed/loss_data_valid_Cora_realization_' + str(kreal) + '_embed_dime_64_dimy_16.npy')[:-1, :]
        accs_train += loss_data_train[-1, -1] / nreal
        accs_ttest += loss_data_ttest[-1, -1] / nreal
        accs_valid += loss_data_valid[-1, -1] / nreal
    print('accs_train = ' + str(accs_train))
    print('accs_ttest = ' + str(accs_ttest))
    print('accs_valid = ' + str(accs_valid))
    return 0

### you have to convert it to the list of lists

def convert_edge_index_to_list(edge_index):
    nedge = edge_index.shape[1]
    idx_vert = 0
    vert_list = []
    edge_list = []
    for idx_edge in range(nedge):
        if idx_vert != edge_index[0, idx_edge]:
            edge_list.append(vert_list)
            vert_list = []
            idx_vert += 1
        if idx_vert == edge_index[0, idx_edge]:
            vert_list.append(edge_index[1, idx_edge])
    edge_list.append(vert_list)
    return edge_list

def main():
    print('inside the main function')
    dataset_name = 'citeseer'
    data = torch_geometric.datasets.CitationFull('./data_torch_geometric', dataset_name)[0]
    eindex = data['edge_index'].cpu().detach().numpy().astype(np.int64)
    print(eindex.shape)
    print(eindex[0, 0])
    print(eindex[1, 0])
    ellist = convert_edge_index_to_list(eindex)
    print(ellist[0])
    print(ellist[1])
    print(ellist[2])
    print(eindex[1, :2])
    print(eindex[1, 2 : 8])
    print(eindex[1, 8 : 9])
    return 0

main()



