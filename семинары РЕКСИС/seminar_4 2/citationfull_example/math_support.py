import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def compute_sparse_matrix_vector_product(smat, iidx, jidx, u):
    v = np.zeros((u.size))
    for idx in range(smat.size):
        v[iidx[idx]] += smat[idx] * u[jidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_sparse_transpose_matrix_vector_product(smat, iidx, jidx, u):
    v = np.zeros((u.size))
    for idx in range(smat.size):
        v[jidx[idx]] += smat[idx] * u[iidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, u):
    numu, dimu = np.shape(u)
    dimv = np.max(iidx) + 1
    v = np.zeros((numu, dimv))
    for idx in range(smat.size):
        v[:, iidx[idx]] += smat[idx] * u[:, jidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def compute_sparse_transpose_matrix_dense_matrix_product(smat, iidx, jidx, u):
    numu, dimu = np.shape(u)
    dimv = np.max(jidx) + 1
    v = np.zeros((numu, dimv))
    for idx in range(smat.size):
        v[:, jidx[idx]] += smat[idx] * u[:, iidx[idx]]
    return v

@jit(nopython=True, fastmath=True)
def perform_orthogonalization(ymat, eps=1.0e-6):
    nvec, dimy = ymat.shape
    umat = ymat.copy()
    for idx in range(nvec):
        umat[idx, :] = umat[idx, :] / max(eps, np.sqrt(np.sum(umat[idx, :] * umat[idx, :])))
        for idx_orth in range(idx + 1, nvec):
            product_value = np.dot(umat[idx, :], umat[idx_orth, :])
            umat[idx_orth, :] = umat[idx_orth, :] - product_value * umat[idx, :]
    return umat

@jit(nopython=True, fastmath=True)
def compute_qr_factorization(smat, iidx, jidx, nvec, dimy, niter, eps=1.0e-6):
    rmat = np.random.normal(0.0, 1.0, (nvec, dimy))
    rmat = perform_orthogonalization(rmat)

    for kiter in range(niter):
        qmat = compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, rmat)
        umat = compute_sparse_transpose_matrix_dense_matrix_product(smat, iidx, jidx, qmat)
        zmat = perform_orthogonalization(umat)
        diff = np.max(np.absolute(zmat - rmat))
        rmat = 0.0 + zmat
        cmat = np.dot(rmat, np.transpose(rmat))
        if diff < eps:
            break
    qmat = compute_sparse_matrix_dense_matrix_product(smat, iidx, jidx, rmat)
    qmat = perform_orthogonalization(qmat)
    return (qmat, rmat)












