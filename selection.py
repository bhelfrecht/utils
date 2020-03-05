#!/usr/bin/env python

import os
import sys
import numpy as np
from regression import LR
from tools import sorted_eigh, sorted_svd

def _compute_G(X, Y, alpha=0.0, reg=1.0E-15):
    """
        Compute the PCovR "kernel"

        ---Arguments---
        X: centered independent (predictor) data
        Y: centered dependent (response) data
        alpha: PCovR alpha
        reg: regularization parameter for linear regression

        ---Returns---
        G: PCovR kernel
    """

    # Build linear regression model
    lr = LR(reg=reg)
    lr.fit(X, Y)
    Yhat = lr.transform(X)

    # Compute G matrix
    G_pca = np.matmul(X, X.T)/np.linalg.norm(X)**2
    G_lr = np.matmul(Yhat, Yhat.T)/np.linalg.norm(Y)**2
    G = alpha*G_pca + (1.0-alpha)*G_lr

    return G

def _compute_S(X, Y, alpha=0.0, reg=1.0E-15, tiny=1.0E-15):
    """
        Compute the PCovR "covariance"

        ---Arguments---
        X: centered independent (predictor) data
        Y: centered dependent (response) data
        alpha: PCovR alpha
        tiny: cutoff for discarding small eigenvalues
        reg: regularization parameter for linear regression

        ---Returns---
        S: PCovR covariance
    """

    # Build linear regression model
    lr = LR(reg=reg)
    lr.fit(X, Y)
    W = lr.W

    # Compute covariance matrix
    C = np.matmul(X.T, X)

    # Compute eigendecomposition of the covariance
    Uc, Vc = sorted_eigh(C, tiny=tiny)

    # Compute square root of the covariance
    C_sqrt = np.matmul(Vc, np.diagflat(np.sqrt(Uc)))
    C_sqrt = np.matmul(C_sqrt, Vc.T)

    # Compute the S matrix
    S_pca = C/np.trace(C)
    S_lr = np.matmul(C_sqrt, W)
    S_lr = np.matmul(S_lr, S_lr.T)/np.linalg.norm(Y)**2

    S = alpha*S_pca + (1.0-alpha)*S_lr

    return S

def _CUR_select(X, Y=None, n=0, k=1, alpha=0.0, mode='covariance', tiny=1.0E-15, reg=1.0E-15):
    """
        Perform CUR column index selection

        ---Arguments---
        X: matrix to decompose
        Y: property matrix (for PCovR selection)
        n: number of points to select
        k: number of top singular values to consider
        alpha: PCovR alpha (PCovR)
        mode: 'covariance' for selecting features, 'kernel' for selecting samples
        tiny: cutoff for discarding small eigenvalues in S (PCovR)
        reg: regularization for regression of Y on selected columns of X (PCovR)

        ---Returns---
        idxs: indices of selected columns of X
    """

    # If n is zero, exit and return empty slice
    if n == 0:
        return slice(None, None, None)

    # If n < zero, return all indices ordered by leverage score
    elif n < 0:
        n = X.shape[0]

    # Initialize indices
    idxs = []

    # Make a copy of X
    X_copy = X.copy()

    # Make a copy of Y and initialize the LR model
    if Y is not None:
        Y_copy = Y.copy()
        lr = LR(reg=reg)

        # Check for valid mode
        if mode != 'covariance' and mode != 'kernel':
            print("Error: unrecognized mode. Valid modes are 'covariance' and 'kernel'")
            return

    # Check for symmetric X
    try:
        sym = np.allclose(X_copy, X_copy.T)
    except ValueError:
        sym = False

    # Loop over the column selections...
    for i in range(0, n):

        # Compute S and use eigendecomposition
        # if we have properties
        if Y is not None:

            # Compute S/G
            if mode == 'kernel':
                SG = _compute_G(X_copy, Y_copy, alpha=alpha, reg=reg)
            else:
                SG = _compute_S(X_copy, Y_copy, alpha=alpha, tiny=tiny, reg=reg)

            # Compute (sparse) eigendecomposition of X
            U, VT = sorted_eigh(SG, k=k, tiny=tiny)
            VT = V.T

        # Use eigendecomposition if symmetric
        elif sym:

            # Compute (sparse) eigendecomposition of X
            U, VT = sorted_eigh(X_copy, k=k, tiny=tiny)
            VT = V.T

        # SVD
        else:

            # Compute (sparse) SVD of X
            U, S, VT = sorted_svd(X_copy, k=k, tiny=tiny)
            
        # Compute leverage score with
        # right singular vectors as rows
        pi = np.sum(VT[0:k, :]**2, axis=0)
        print(pi)
        print(np.amax(pi), np.amin(pi))

        # Pick column index with highest score
        pi_idx = np.argmax(pi)
        idxs.append(pi_idx)

        # Gram-Schmidt Orthogonalization
        X_select = X_copy[:, pi_idx]
        X_select_norm = X_select / np.dot(X_select, X_select)
        X_copy -= np.outer(X_select_norm, np.matmul(X_select, X_copy))

        # Eliminate Y components that are described by selected features
        # TODO: should the fit be based on Xc or X_copy?
        if Y is not None:
            lr.fit(X_copy[:, idxs], Y_copy)
            Y_copy -= lr.transform(X_copy[:, idxs]) 

    idxs = np.asarray(idxs)
    return idxs

def CUR(X, Y=None, n_col=0, n_row=0, k=1, alpha=0.0, tiny=1.0E-15, reg=1.0E-15,
        compute_U=False, compute_Q=False):
    """
        Perform CUR matrix decomposition

        ---Arguments---
        X: matrix to decompose
        Y: property matrix (for PCovR selection)
        n: number of points to select
        k: number of top singular values to consider
        alpha: PCovR alpha (PCovR)
        tiny: cutoff for discarding small eigenvalues in S (PCovR)
        reg: regularization for regression of Y on selected columns of X (PCovR)
        compute_U: compute the U matrix such that X = CUX
        compute_Q: compute Q so that one can build a projection T = CQ

        ---Returns---
        idxs: indices of selected columns of X
        U: U matrix (if compute_U is True)
        Q: Q matrix (if compute_Q is True)
    """

    # Initialize outputs and indices
    outputs = []

    # Select column indices
    idxs_c = _CUR_select(X, Y=Y, n=n_col, k=k, 
            alpha=alpha, mode='covariance', tiny=tiny, reg=reg)

    # Select row indices
    # (PCovR selection only valid on columns due to regression on Y)
    idxs_r = _CUR_select(X.T, Y=Y, n=n_row, k=k, 
            alpha=alpha, mode='kernel', tiny=tiny, reg=reg)

    # Append indices to outputs
    outputs.append(idxs_c)
    outputs.append(idxs_r)

    # Compute U
    if compute_U:
        Xc = X[:, idxs_c]
        Xr = X[idxs_r, :]
        Uc = np.linalg.pinv(Xc)
        Ur = np.linalg.pinv(Xr)
        U = np.matmul(Uc, X)
        U = np.matmul(U, Ur)
        outputs.append(U)

    # Compute Q
    if compute_Q:
        Q = np.matmul(Uc, X)
        Q = np.matmul(Q, Q.T)
        Uq, Vq = np.linalg.eigh(Q)
        Uq = np.sqrt(Uq)
        Q = np.matmul(Vq, Uq)
        Q = np.matmul(Q, Vq.T)
        outputs.append(Q)

    return outputs

def FPS(X, n=0):
    """
        Does Farthest Point Selection on a set of points X
        Adapted from a routine by Michele Ceriotti

        ---Arguments---
        X: data on which to perform the FPS
        n: number of points to select (<= 0 for all points)

        ---Returns---
        fps_idxs: indices of the FPS points
        d: min max distances at each iteration
    """
    N = X.shape[0]

    # If desired number of points less than or equal to zero,
    # select all points
    if n <= 0:
        n = N

    # Initialize arrays to store distances and indices
    fps_idxs = np.zeros(n, dtype=int)
    d = np.zeros(n)

    # Pick first point at random
    idx = np.random.randint(0, N)
    fps_idxs[0] = idx

    # Compute distance from all points to the first point
    d1 = np.linalg.norm(X-X[idx], axis=1)**2

    # Loop over the remaining points...
    for i in range(1, n):

        # Get maximum distance and corresponding point
        fps_idxs[i] = np.argmax(d1)
        d[i-1] = np.amax(d1)

        # Exit if we have exhausted the unique points
        # (in which case we select a point we have selected before)
        if fps_idxs[i] in fps_idxs[0:i]:
            fps_idxs = fps_idxs[0:i]
            d = d[0:i]
            break

        # Compute distance from all points to the selected point
        d2 = np.linalg.norm(X-X[fps_idxs[i]], axis=1)**2

        # Set distances to minimum among the last two selected points
        d1 = np.minimum(d1, d2)

    return fps_idxs, d

def random_selection(n_total, n=0):
    """
        Select a random number of samples

        ---Arguments---
        n_total: input data to sample
        n: number of points to select

        ---Returns---
        idxs: indices for the selection
    """

    # Select random indices
    idxs = np.arange(0, n_total)
    np.random.shuffle(idxs)

    # Retain n indices
    idxs = idxs[0:n]
    
    return idxs

def std_selection(X, n=0, cutoff=1.0E-3):
    """
        Selects at most n samples with relative 
        standard deviation larger than the cutoff

        ---Arguments---
        X: input data to sample
        n: number of points to select
        cutoff: points with relative standard deviation
            greater than cutoff are selected (at most n)

        ---Returns---
        idxs: indices for the selection
    """

    # Select sampples where the relative standard deviation
    # is greater than the cutoff
    idxs = np.arange(0, X.shape[0])
    idxs = idxs[np.std(X, axis=1)/np.mean(X, axis=1) > cutoff]
    
    # Retain only n indices
    if n > 0:
        idxs = idxs[0:n]
        
    return idxs
