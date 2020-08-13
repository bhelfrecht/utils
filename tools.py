#!/usr/bin/env python

import os
import sys
import gzip
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

def save_model(model_instance, output):
    """
        JSONify an object through its __dict__ attribute

        ---Arguments---
        model_instance: model to save
        output: output file
    """

    model_dict = model_instance.__dict__.copy()

    # Convert arrays to lists
    for k, v in model_dict.items():
        if isinstance(v, np.ndarray):
            kpcovr_dict[k] = v.tolist()
    
    # Save
    with open(output, 'w') as f:
        json.dump(model_dict, f)

def load_model(model_instance, input_file):
    """
        Load a JSONified object

        ---Arguments---
        model_instance: class instance in which
            to load the JSONified object
        input_file: JSON file containing the
            __dict__ for the class instance

        ---Returns---
        model_instance: loaded object instance
    """

    with open(input_file, 'r') as f:
        model_dict = json.load(f)

    # Turn lists into arrays
    for k, v in model_dict.items():
        if isinstance(v, list):
            model_dict[k] = np.array(v)
    
    model_instance.__dict__ = model_dict
    
    return model_instance

def load_json(json_file):
    """
        Shorthand for loading a JSON file

        ---Arguments---
        json_file: JSON file to load

        ---Returns---
        json_object: object read from the JSON file
    """

    if json_file.endswith('.gz'):
        with gzip.GzipFile(json_file, 'r') as f:
            json_object = json.load(f)
    else:
        with open(json_file, 'r') as f:
            json_object = json.load(f)

    return json_object

