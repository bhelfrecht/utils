#!/usr/bin/env python

import os
import sys
import gzip
import json
import h5py
import numpy as np
from copy import deepcopy
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

def recursive_convert(obj):
    """
        Go through a collection
        (dict, list, etc.) and convert all
        numpy arrays to lists for JSON
        serialization

        ---Arguments---
        obj: The object to convert

        ---Returns---
        obj: The object with all arrays
            converted to lists
    """
    # Convert tuples to list before continuing
    if isinstance(obj, tuple):
        obj = list(obj)

    # TODO: maybe raise an exception for object arrays?
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    # Numpy int32 isn't serializable for some reason
    elif isinstance(obj, np.int32):
        obj = int(obj)
    elif isinstance(obj, list):
        for idx, i in enumerate(obj):
            obj[idx] = recursive_convert(i)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursive_convert(v)
    elif hasattr(obj, '__dict__'):
        obj = obj.__dict__
        for k, v in obj.items():
            obj[k] = recursive_convert(v)

    return obj

def recursive_array_convert(obj):
    """
        Go through a collection
        (dict, list, etc.) and convert all
        lists to numpy arrays

        ---Arguments---
        obj: The object to convert

        ---Returns---
        obj: The object with all lists 
            converted to numpy arrays
    """

    # Convert tuples to list before continuing
    if isinstance(obj, list):
        obj = np.array(obj)

    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursive_array_convert(v)

    return obj

def save_json(json_object, output, array_convert=False):
    """
        JSONify an object

        ---Arguments---
        json_object: container to save
        output: output file
        array_convert: whether to convert numpy arrays to lists.
            The json_object is searched and all
            numpy arrays are converted to lists
    """
    
    # Make sure we aren't modifying the original
    json_object = deepcopy(json_object)

    if array_convert:
        json_object = recursive_convert(json_object)

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

