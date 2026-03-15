import numpy as np
import pandas as pd
from numba import jit

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import roc_auc_score


### let me do the same thing in the 
### what do you need for that?
### replace matrix by a sparse one

# @jit(nopython=True, fastmath=True)
def generate_random_martrix(n0, n1):
    amat = np.random.normal(0.0, 1.0, (n0, n1))
    return amat

@jit(nopython=True, fastmath=True)
def convert_dense_to_sparse(amat):
    n0, n1 = amat.shape
    ### what is the next step
    smat = np.zeros((n0 * n1))
    iidx = np.zeros((n0 * n1), dtype=np.int64)
    jidx = np.zeros((n0 * n1), dtype=np.int64)
    idx_sparse = 0
    for idx0 in range(n0):
        for idx1 in range(n1):
            smat[idx_sparse] = amat[idx0, idx1]
            iidx[idx_sparse] = idx0
            jidx[idx_sparse] = idx1
            idx_sparse = idx_sparse + 1
    return (smat, iidx, jidx)

@jit(nopython=True, fastmath=True)
def gram_schmidt_orth(wdata, eps=1.0e-6):
    numw, dimw = wdata.shape
    udata = 0.0 + wdata
    for idx0 in range(numw):
        for idx1 in range(idx0):
            norm_sq = max(eps, np.dot(udata[idx1, :], udata[idx1, :]))
            udata[idx0, :] = udata[idx0, :] - np.dot(udata[idx1, :], udata[idx0, :]) / norm_sq * udata[idx1, :]        
        norm = max(eps, np.sqrt(np.dot(udata[idx0, :], udata[idx0, :])))
        udata[idx0, :] = udata[idx0, :] / norm
    return udata
    
@jit(nopython=True, fastmath=True)
def compute_eigh_pair(hmat, nvec, niter=1000):
    dims = hmat.shape[0]
    wdata = np.random.normal(0.0, 1.0, (nvec, dims))
    for kiter in range(niter):
        wdata = gram_schmidt_orth(np.dot(0.0 + wdata, hmat))
    eigen_val = np.sum(np.dot(wdata, hmat) * wdata, axis=1)
    return (wdata, eigen_val)

@jit(nopython=True, fastmath=True)
def compute_eigen_data(amat, nvec, niter=1000):
    hmat = np.dot(np.transpose(amat), amat)
    imat, lval = compute_eigh_pair(hmat, nvec, niter=niter)
    imat = np.transpose(imat)
    umat = np.dot(amat, imat)
    for idx in range(nvec):
        imat[:, idx] = np.sqrt(np.sqrt(lval[idx])) * imat[:, idx]
        umat[:, idx] = umat[:, idx] / np.sqrt(np.sqrt(lval[idx]))
    return (umat, imat)

    
@jit(nopython=True, fastmath=True)
def compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, u, dims):
    numu, dimu = np.shape(u)
    # dimv = np.max(iidx) + 1
    v = np.zeros((numu, dims))
    for idx in range(smat.size):
        v[:, iidx[idx]] += smat[idx] * u[:, jidx[idx]]
    return v

    
@jit(nopython=True, fastmath=True)
def compute_sparse_transpose_matrix_dense_matrix_product(smat, iidx, jidx, u, dims):
    numu, dimu = np.shape(u)
    dimv = np.max(jidx) + 1
    v = np.zeros((numu, dims))
    for idx in range(smat.size):
        v[:, jidx[idx]] += smat[idx] * u[:, iidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_eigh_pair_sparse(amat, iidx, jidx, dims0, dims1, nvec, niter=1000):
    wdata = np.random.normal(0.0, 1.0, (nvec, dims0))
    for kiter in range(niter):
        # wdata0 = 0.0 + wdata
        wdata = compute_sparse_matrix_dense_matrix_product(amat, iidx, jidx, 0.0 + wdata, dims1)
        wdata = compute_sparse_transpose_matrix_dense_matrix_product(amat, iidx, jidx, 0.0 + wdata, dims0)
        wdata = gram_schmidt_orth(0.0 + wdata)
        # diff = np.max(np.absolute(wdata - wdata0))
        # print('diff = ' + str(diff))
        
    vdata = compute_sparse_matrix_dense_matrix_product(amat, iidx, jidx, 0.0 + wdata, dims1)
    vdata = compute_sparse_transpose_matrix_dense_matrix_product(amat, iidx, jidx, 0.0 + vdata, dims0)
    eigen_val = np.sum(vdata * wdata, axis=1)
    return (wdata, eigen_val)

@jit(nopython=True, fastmath=True)
def compute_eigen_data_sparse(amat, iidx, jidx, nitem, nuser, nvec, niter=1000):
    imat, lval = compute_eigh_pair_sparse(amat, iidx, jidx, nitem, nuser, nvec, niter=niter)
    ### now you have to multiply the matrix
    umat = compute_sparse_matrix_dense_matrix_product(amat, iidx, jidx, imat, nuser)
    ### now you have to perform normalisation
    imat = np.transpose(imat)
    umat = np.transpose(umat)
    for idx in range(nvec):
        imat[:, idx] = np.sqrt(np.sqrt(lval[idx])) * imat[:, idx]
        umat[:, idx] = umat[:, idx] / np.sqrt(np.sqrt(lval[idx]))
    return (umat, imat)


def eigen_vector_calculation_test():
    print('inside the main function')
    n0 = 60
    n1 = 30
    nvec = 30
    amat = generate_random_martrix(n0, n1)
    smat, iidx, jidx = convert_dense_to_sparse(amat)
    umat, imat = compute_eigen_data(amat, nvec, niter=1000)
    print('umat:')
    print(umat)
    print('imat:')
    print(imat)
    ### should you check that the inner product is the same
    pred = np.dot(umat, np.transpose(imat))
    res = np.max(np.absolute(pred - amat))
    print('res = ' + str(res))
    
    nuser = n0
    nitem = n1
    umat, imat = compute_eigen_data_sparse(smat, iidx, jidx, nitem, nuser, nvec, niter=1000)
    print('umat:')
    print(umat)
    print('imat:')
    print(imat)
    ### should you check that the inner product is the same
    pred = np.dot(umat, np.transpose(imat))
    res = np.max(np.absolute(pred - amat))
    print('res = ' + str(res))
    return 0
    
# eigen_vector_calculation_test()
    

def get_positives_and_negatives(idata, rtarget):
    nedge = idata.shape[0]
    emat = np.zeros(nedge)
    uidx = np.zeros((nedge), dtype=np.int64)
    iidx = np.zeros((nedge), dtype=np.int64)
    nneg = 0
    npos = 0

    for idx in range(idata.shape[0]):
        uidx[idx] = idata[idx, 0]
        iidx[idx] = idata[idx, 1]
        if idata[idx, 2] >= rtarget:
            npos += 1
            emat[idx] = 1.0
        else:
            nneg += 1
            emat[idx] = 0.0
    print('npos = ' + str(npos))
    print('nneg = ' + str(nneg))
    return (emat, uidx, iidx)
    
def train_test_split(emat, uidx, iidx, frac_train=0.7):
    nedge = emat.size
    ntrain = np.int64(frac_train * nedge)
    nttest = nedge - ntrain
    perm = np.random.permutation(nedge)
    idx_train = perm[:ntrain]
    idx_ttest = perm[ntrain:]
    ### what happens here?
    emat_train = emat[idx_train[:]]
    uidx_train = uidx[idx_train[:]]
    iidx_train = iidx[idx_train[:]]
    emat_ttest = emat[idx_ttest[:]]
    uidx_ttest = uidx[idx_ttest[:]]
    iidx_ttest = iidx[idx_ttest[:]]
    
    return (emat_train, uidx_train, iidx_train,
            emat_ttest, uidx_ttest, iidx_ttest)

def recom_power_method():
    print('inside the main function')
    ratings = pd.read_csv('./ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    print(ratings.head())
    idata = ratings.values
    print(idata.shape)
    idata[:, 0] -= 1
    idata[:, 1] -= 1
    nuser = np.max(idata[:, 0]) + 1
    nitem = np.max(idata[:, 1]) + 1
    print(np.min(idata[:, 0]))
    print(np.max(idata[:, 0]))
    print(np.min(idata[:, 1]))
    print(np.max(idata[:, 1]))
    print(idata[:10, 2])
    print('nuser = ' + str(nuser))
    print('nitem = ' + str(nitem))
    print(np.min(idata[:, 2]))
    print(np.max(idata[:, 2]))
    
    rtarget = 4.0
    emat, uidx, iidx = get_positives_and_negatives(idata, rtarget)
    emat_train, uidx_train, iidx_train, emat_ttest, uidx_ttest, iidx_ttest = train_test_split(emat, uidx, iidx, frac_train=0.7)
    nvec = 60
    print('np.max(uidx_train) = ' + str(np.max(uidx_train)))
    print('np.max(uidx_ttest) = ' + str(np.max(uidx_ttest)))
    print('np.max(iidx_train) = ' + str(np.max(iidx_train)))
    print('np.max(iidx_ttest) = ' + str(np.max(iidx_ttest)))
    umat, imat = compute_eigen_data_sparse(emat_train, uidx_train, iidx_train, nitem, nuser, nvec, niter=10000)
    print('umat.shape = ' + str(umat.shape))
    print('imat.shape = ' + str(imat.shape))
    score_train = np.sum(umat[uidx_train[:], :] * imat[iidx_train[:], :], axis=1)
    label_train = emat_train
    score_ttest = np.sum(umat[uidx_ttest[:], :] * imat[iidx_ttest[:], :], axis=1)
    label_ttest = emat_ttest
    
    roc_auc_train = roc_auc_score(label_train, score_train)
    roc_auc_ttest = roc_auc_score(label_ttest, score_ttest)
    print('roc_auc_train = ' + str(roc_auc_train))
    print('roc_auc_ttest = ' + str(roc_auc_ttest))
    return 0

main()



