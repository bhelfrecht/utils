#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh
from regression import LR

# TODO: sorted svd

def sorted_eigh(X, k=None, tiny=None):
    # TODO: sparse option
    """
        Compute eigendecomposition and sort
        eigenvalues and eigenvectors in descending order,
        and remove small eigenvalues (and corresponding eigenvectors)

        ---Arguments---
        X: matrix to decompose
        tiny: cutoff for discarding small eigenvalues

        ---Returns---
        U: sorted eigenvalues
        V: eigenvectors corresponding to the sorted eigenvalues
    """

    # Eigendecomposition
    U, V = np.linalg.eigh(X)

    # Sort in descending order
    U = np.flip(U, axis=0)
    V = np.flip(V, axis=1)

    # Remove small eigenvalues and corresponding eigenvectors
    if tiny is not None:
        V = V[:, U > tiny]
        U = U[U > tiny]

    return U, V

def _compute_S(X, Y, alpha=0.0, tiny=1.0E-15, reg=1.0E-15):
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
    Uc, Vc = np.linalg.eigh(C)
    Uc = np.flip(Uc, axis=0)
    Vc = np.flip(Vc, axis=1)
    Vc = Vc[:, Uc > tiny]
    Uc = Uc[Uc > tiny]

    # Compute square root of the covariance
    C_sqrt = np.matmul(Vc, np.diagflat(np.sqrt(Uc)))
    C_sqrt = np.matmul(C_sqrt, Vc.T)

    # Compute the S matrix
    S_pca = C/np.trace(C)
    S_lr = np.matmul(C_sqrt, W)
    S_lr = np.matmul(S_lr, S_lr.T)/np.linalg.norm(Y)**2

    S = alpha*S_pca + (1.0-alpha)*S_lr

    return S

def _CUR_select(X, Y=None, n=0, k=1, alpha=0.0, tiny=1.0E-15, reg=1.0E-15):
    """
        Perform CUR column index selection

        ---Arguments---
        X: matrix to decompose
        Y: property matrix (for PCovR selection)
        n: number of points to select
        k: number of top singular values to consider
        alpha: PCovR alpha (PCovR)
        tiny: cutoff for discarding small eigenvalues in S (PCovR)
        reg: regularization for regression of Y on selected columns of X (PCovR)

        ---Returns---
        idxs: indices of selected columns of X
    """

    # Exit straight away if no points are to be selected;
    # return slice giving whole set of columns
    if n <= 0:
        return slice(None, None, None)

    min_dim = np.min(X.shape)
    if k > min_dim:
        print("Error: k must be less than"
                " or equal to the smallest matrix dimension")
        return

    # Initialize indices
    idxs = []

    # Make a copy of X
    X_copy = X.copy()

    # Make a copy of Y and initialize the LR model
    if Y is not None:
        Y_copy = Y.copy()
        lr = LR(reg=reg)

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

            # Compute S
            S = _compute_S(X_copy, Y_copy, alpha=alpha, tiny=tiny, reg=reg)

            # Compute (sparse) eigendecomposition of S
            if k < min_dim:
                U, V = eigsh(S, k=k)

                # Sort the largest eigenvalues,
                # since the sparse order is not guaranteed
                idxs_U = np.argsort(U)
                idxs_U = np.flip(idxs_U)
                U = U[idxs_U]
                V = V[:, idxs_U]
                V = V.T

            # Compute full eigendecomposition of S
            else:
                U, V = np.linalg.eigh(S)
                U = np.flip(U, axis=0)
                V = np.flip(V, axis=1)
                V = V.T


        # Use eigendecomposition if symmetric
        elif sym:

            # Compute (sparse) eigendecomposition of X
            if k < min_dim:
                U, V = eigsh(X_copy, k=k)

                # Sort the largest eigenvalues,
                # since the sparse order is not guaranteed
                idxs_U = np.argsort(U)
                idxs_U = np.flip(idxs_U)
                U = U[idxs_U]
                V = V[:, idxs_U]
                V = V.T

            # Compute full eigendecomposition of X
            else:
                U, V = np.linalg.eigh(X_copy)
                U = np.flip(U, axis=0)
                V = np.flip(V, axis=1)
                V = V.T

        # SVD
        else:

            # Compute (sparse) SVD of X
            if k < min_dim:
                U, S, V = svds(X_copy, k=k)

                # Sort the largest eigenvalues,
                # since the sparse order is not guaranteed
                idxs_S = np.argsort(S)
                idxs_S = np.flip(idxs_S)
                S = S[idxs_S]
                U = U[:, idxs_S]
                V = V[idxs_S, :]

            # Compute full SVD of X
            else:
                U, S, V = np.linalg.svd(X_copy)
            
        # Compute leverage score with
        # right singular vectors as rows
        pi = np.sum(V[0:k, :]**2, axis=0)
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


def simple_split(X, Y, f_train):
    # TODO: add option for structure selection via structure_idxs argument
    """
        Perform train-test split of a dataset

        ---Arguments---
        X: independent variable
        Y: dependent variable

        ---Returns---
        X_train: X training set
        X_test: X test set
        Y_train: Y data corresponding to the X training set
        Y_test: Y data corresponding to the X test set
    """

    # Check for consistent shapes
    if X.shape[0] != Y.shape[0]:
        print("Error: X and Y must have same length")
        return

    # Build array of all sample indices
    n_total = X.shape[0]
    idxs = np.arange(0, n_total, dtype=np.int)
    np.random.shuffle(idxs)

    # Number of training points
    n_train = int(np.floor(f_train*n_total))

    # Build lists of indices of training and test sets
    idxs_train = idxs[0:n_train]
    idxs_test = idxs[n_train:]
    
    # Split the dataset according to training and test indices
    X_train = X[idxs_train]
    X_test = X[idxs_test]
    Y_train = Y[idxs_train]
    Y_test = Y[idxs_test]

    return X_train, X_test, Y_train, Y_test

def cv_split(X, Y, k, stratified=False):
    # TODO: add option for structure selection via structure_idxs argument
    """
        Performs a k-fold cross-validation split of a dataset

        ---Arguments---
        X: independent variable
        Y: dependent variable
        k: number of folds
        stratified: flag to perform stratified cross validation

        ---Returns---
        X_folds: list of k subarrays of X data (i.e., split into k folds)
        Y_folds: list of k subarrays of Y data (i.e., split into k folds)
    """

    # Check for consistent shapes
    if X.shape[0] != Y.shape[0]:
        print("Error: X and Y must have same length")
        return

    # Total number of samples
    n_samples = X.shape[0]

    # Check for valid splitting
    try:
        idxs_split = np.split(np.arange(0, n_samples), k)
    except ValueError:
        print("Error: number of samples must be divisible by k; "
                "choose a different k or change the sample size")

    # Stratified sampling
    if stratified:

        # Sort by property
        idxs_sort = np.argsort(Y, axis=0)
        Y = Y[idxs_sort]
        X = X[idxs_sort]

        # Split the sorted data
        Y = [Y[i] for i in idxs_split]
        X = [X[i] for i in idxs_split]

        # Shuffle the data in the splits
        for x, y in zip(X, Y):
            idxs = np.arange(0, y.shape[0])
            np.random.shuffle(idxs)
            y = y[idxs]
            x = x[idxs]

        # Concantenate the shuffled splits
        Y = np.concatenate(Y)
        X = np.concatenate(X)

        # Get new folds
        Y_folds = [Y[i::k] for i in range(0, k)]
        X_folds = [X[i::k] for i in range(0, k)]

    # Standard sampling
    else:
        X_folds = [X[i] for i in idxs_split]
        Y_folds = [Y[i] for i in idxs_split]

    return X_folds, Y_folds

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
            alpha=alpha, tiny=tiny, reg=reg)

    # Select row indices
    # (PCovR selection only valid on columns due to regression on Y)
    idxs_r = _CUR_select(X.T, Y=None, n=n_row, k=k, 
            alpha=alpha, tiny=tiny, reg=reg)

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
