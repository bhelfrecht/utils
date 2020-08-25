#!/usr/bin/env python

import os
import sys
import gzip
import json
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

def load_json(json_file, array_convert=False):
    """
        Shorthand for loading a JSON file

        ---Arguments---
        json_file: JSON file to load
        array_convert: whether to convert lists to arrays.
            If the loaded object is a list of lists,
            returns a list of arrays (which can be concatenated
            with np.vstack). A dict of lists is converted
            to a list of arrays, and a simple list is left as-is.

        ---Returns---
        json_object: object read from the JSON file
    """

    if json_file.endswith('.gz'):
        with gzip.GzipFile(json_file, 'r') as f:
            json_object = json.load(f)
    else:
        with open(json_file, 'r') as f:
            json_object = json.load(f)

    if array_convert:
        if isinstance(json_object, list):
            for idx, i in enumerate(json_object):
                if isinstance(i, list):
                    json_object[idx] = np.asarray(i)
        elif isinstance(json_object, dict):
            for k, v in json_object.items():
                if isinstance(v, list):
                    json_object[k] = np.asarray(v)

    return json_object

def save_json(json_object, output, array_convert=False):
    """
        JSONify an object

        ---Arguments---
        json_object: container to save
        output: output file
        array_convert: whether to convert numpy arrays to lists.
            If the provided object is an array or a list of arrays,
            it is saved as a list of lists (and then can be loaded
            as a list of arrays with `load_json` and concatenated as desired).
            If the provided object is a dict of arrays, it is saved
            as a dict of lists.
    """

    if array_convert:
        if isinstance(json_object, np.ndarray):
            json_object = json_object.tolist()
        elif isinstance(json_object, list):
            for idx, i in enumerate(json_object):
                if isinstance(i, np.ndarray):
                    json_object[idx] = i.tolist()
        elif isinstance(json_object, dict):
            for k, v in json_object.items():
                if isinstance(v, np.ndarray):
                    json_object[k] = v.tolist()

    if output.endswith('.gz'):
        with gzip.GzipFile(output, 'w') as f:
            json.dump(json_object, f)
    else:
        with open(output, 'w') as f:
            json.dump(json_object, f)

def save_structured_array(output, array, dtype):
    """
        Save a structured (2D) numpy array in plaintext

        ---Arguments---
        output: output file
        array: structured array to save
        dtype: numpy dtype of the structured array
    """

    columns = []
    header = []
    for name in dtype.names:
        column = array[name]
        if column.ndim == 1:
            column = np.reshape(column, (-1, 1))
            header.append(name)
        else:
            n_cols = column.shape[1]
            header.append(f'{name}({n_cols})')
        columns.append(column)
    header = ' '.join(header)
    np.savetxt(f'{output}', np.hstack(columns), header=header)

