#!/usr/bin/env python3

import os
import sys
import numpy as np
from ase.io import read, write
import argparse

# NOTE: If you have multiple sets of properties corresponding
# to different sets of species, you can run this script multiple times,
# iteratively appending properties to the XYZ file (i.e.,
# by saving the XYZ after appending properties for one set of species
# and using that XYZ as the input for appending the properties
# for the next set of species

def load_property(filename):
    """
        Load data from well-formatted text
        or numpy archives

        ---Arguments---
        filename: name of the file to load

        ---Returns---
        p: data contained in the file
    """

    # Determine extension and use appropriate
    # loading scheme
    ext = os.path.splitext(filename)[1]
    if ext == '.npy' or ext == '.npz':
        p = np.load(filename)
    else:

        # Use genfromtxt to automatically determine dtype;
        # use system default for string encoding
        p = np.genfromtxt(filename, dtype=None, encoding=None)
        #p = np.loadtxt(filename)

    return p

# Setup the parser
parser = argparse.ArgumentParser()

# Atomic structure and properties
parser.add_argument('xyz', type=str,
        help='Input xyz file with properties')
parser.add_argument('-ap', type=str, default=[], nargs='+',
        help='Files containing atomic properties')
parser.add_argument('-apn', type=str, default=[], nargs='+',
        help='List of atomic property names in same order as files')
parser.add_argument('-sp', type=str, default=[], nargs='+',
        help='Files containing structure properties')
parser.add_argument('-spn', type=str, default=[], nargs='+',
        help='List of structure property names in same order as files')
parser.add_argument('-c', type=float, default=None,
        help='Atomic environment cutoff')
parser.add_argument('-Z', type=str, default=None, nargs='+',
        help='Species (symbols) associated with atomic properties')

# Metadata
parser.add_argument('-n', type=str, default='name',
        help='Name of dataset')
parser.add_argument('-d', type=str, default='description',
        help='Description of dataset')
parser.add_argument('-a', type=str, default=None, nargs='+',
        help='Authors')
parser.add_argument('-r', type=str, default=None, nargs='+',
        help='References')

# Chemiscope directory
parser.add_argument('-dir', type=str, default=None,
        help='Directory containing chemiscope_input.py')

# File name and outputs
parser.add_argument('-json', type=str, default=None,
        help='Name of output JSON file')
parser.add_argument('-extxyz', type=str, default=None, 
        help='Name of output XYZ file with properties')

# Parse commands
args = parser.parse_args()

# Load chemiscope input builder
if args.dir is not None:
    sys.path.append(args.dir)
from chemiscope_input import write_chemiscope_input

# Read structures
frames = read(args.xyz, index=':')
n_atoms = [f.get_number_of_atoms() for f in frames]
n_atoms_cumul = np.cumsum(n_atoms)
n_atoms_total = n_atoms_cumul[-1]
all_species = np.concatenate([f.get_chemical_symbols() for f in frames])

# Species selection
if args.Z is None:
    atom_idxs = np.arange(0, n_atoms_total)
else:
    atom_idxs = np.in1d(all_species, args.Z)

# Initialize "extra" for structure properties
# since chemiscope and extxyz handle
# arrays differently
extra = {}

# Append atomic properties
for ap, apn in zip(args.ap, args.apn):

    # Load raw data (for center environments)
    raw_data = load_property(ap)

    # Initialize property data for ALL environments
    if len(raw_data.shape) == 1:
        atom_data = np.full(n_atoms_total, np.nan, dtype=float)
    else:
        atom_data = np.full((n_atoms_total, raw_data.shape[1]), np.nan, dtype=float)

    # Fill in the data for center environments
    atom_data[atom_idxs] = raw_data

    # Split into structures
    atom_data = np.split(atom_data, n_atoms_cumul[0:-1])

    # Append atomic properties to frame
    for f, a in zip(frames, atom_data):
        f.arrays[apn] = a

# Append structure properties
for sp, spn in zip(args.sp, args.spn):

    # Load raw data (all structures)
    structure_data = load_property(sp)

    # Add structure properties to dictionary
    extra[spn] = {'target': 'structure', 
            'values': structure_data}
    
# Write chemiscope input
if args.json is not None:

    # Metadata
    meta = dict(name=args.n, description=args.d)

    # Authors
    if args.a is not None:
        meta['authors'] = args.a
    else:
        meta['authors'] = 'None'

    # References
    if args.r is not None:
        meta['references'] = args.r
    else:
        meta['references'] = 'None'

    # Write input
    write_chemiscope_input(args.json, meta, frames, extra, cutoff=args.c)

# Write new extended XYZ with appended properties
if args.extxyz is not None:

    # Append structure properties to frame
    for key, value in extra.items():
        for f, s in zip(frames, value['values']):
            f.info[key] = s

    write(args.extxyz, frames, format='extxyz')

