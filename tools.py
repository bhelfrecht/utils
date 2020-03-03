#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh

def sorted_svd(X, k=None, tiny=None):
    """
        Compute singular value decomposition
        and sort singular values and vectors in descending order,
        and remove small singular values (and corresponding vectors)

        ---Arguments---
        X: matrix to decompose
        k: number of top singular vectors to compute
        tiny: cutoff for discarding small singular values

        ---Returns---
        U: left singular vectors (as columns)
        S: singular values
        VT: right singular vectors (as rows)
    """

    # Minimum matrix dimension
    min_dim = np.min(X.shape)

    # Compute full SVD
    if k is None or k == min_dim:
        U, S, VT = np.linalg.svd(X) # Already sorted in descending order

    # Invalid k
    elif k > np.min(X.shape) or k < 1:
        print("Error: k must be a positive integer less than the"
                " minimum dimension of X or None")
        return

    # Comute sparse SVD
    else:
        U, V = eigsh(X, k=k)

        # Sort in descending order
        # (can't just flip because order isn't guaranteed)
        idxs_U = np.argsort(U)
        idxs_U = np.flip(idxs_U, axis=0)
        U = U[idxs_U]
        V = V[:, idxs_U]

    # Discard small singular values and corresponding vectors
    if tiny is not None:
        U = U[:, S > tiny]
        VT = VT[S > tiny, :]
        S = S[S > tiny]

    return U, S, VT

def sorted_eigh(X, k=None, tiny=None):
    """
        Compute eigendecomposition and sort
        eigenvalues and eigenvectors in descending order,
        and remove small eigenvalues (and corresponding eigenvectors)

        ---Arguments---
        X: matrix to decompose
        k: number of top eigenvalues to compute
        tiny: cutoff for discarding small eigenvalues

        ---Returns---
        U: sorted eigenvalues
        V: eigenvectors corresponding to the sorted eigenvalues
    """

    # Minimum matrix dimension
    min_dim = np.min(X.shape)

    # Compute full eigendecomposition
    if k is None or k == min_dim:
        U, V = np.linalg.eigh(X)

        # Sort in descending order
        U = np.flip(U, axis=0)
        V = np.flip(V, axis=1)

    # Invalid k
    elif k > min_dim or k < 1:
        print("Error: k must be a positive integer less than the"
                " minimum dimension of X or None")
        return

    # Compute sparse eigendecomposition
    else:
        U, V = eigsh(X, k=k)

        # Sort in descending order
        # (can't just flip because order isn't guaranteed)
        idxs_U = np.argsort(U)
        idxs_U = np.flip(idxs_U, axis=0)
        U = U[idxs_U]
        V = V[:, idxs_U]

    # Remove small eigenvalues and corresponding eigenvectors
    if tiny is not None:
        V = V[:, U > tiny]
        U = U[U > tiny]

    return U, V
