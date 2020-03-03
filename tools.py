#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh

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
