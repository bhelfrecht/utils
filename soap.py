#!/usr/bin/env python

import os
import sys
import numpy as np
from copy import deepcopy
from scipy.linalg import fractional_matrix_power
from scipy.special import gamma, legendre, roots_legendre, eval_legendre
from quippy.descriptors import Descriptor
import h5py
from tqdm import tqdm
from rascal.representations import SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species
import itertools
from ase.neighborlist import neighbor_list

# TODO: multiprocessing on computing soaps and writing
# TODO: decide one or multiple hdf5 files, take multiprocessing
# into account

def _truncate_average(soap, component_idxs=None, average=False):
    """
        Helper function to truncate and average 
        the SOAP vectors over central atoms
        in a structure

        ---Arguments---
        soap: soap vector to operate on
        component_idxs: list of component indices
        average: compute average SOAP over centers

        ---Returns---
        soap: SOAP vector
    """

    # Truncate SOAP vectors
    if component_idxs is not None:
        soap = soap[:, component_idxs]

    # Average SOAP vectors over centers
    if average:
        soap = np.mean(soap, axis=0)

    return soap

# TODO: make Z and species_Z optional arguments, where if not given
# they default to all species
def quippy_soap(
    structures, 
    Z, 
    species_Z, 
    n_max=6, 
    l_max=6, 
    cutoff=3.0,     
    cutoff_transition_width=0.5, 
    atom_sigma=0.1,
    cutoff_scale=1.0, 
    cutoff_rate=1.0, 
    cutoff_dexp=0,
    covariance_sigma0=0.0, 
    central_weight=1.0, 
    basis_error_exponent=10.0,
    normalise=True, 
    central_reference_all_species=False, 
    diagonal_radial=False, 
    quippy_average=False,
    average=False, 
    component_idxs=None, 
    concatenate=False, 
    chunks=None, 
    output=None
):
    """
        Compute SOAP vectors with quippy
        (see https://libatoms.github.io/QUIP/descriptors.html)

        ---Arguments---
        structures: list of ASE atoms objects
        Z: list of central atom species (as atomic number)
        species_Z: list of environment atom species (as atomic number)
        n_max: number of radial basis functions
        l_max: spherical harmonics basis band limit
        cutoff: radial cutoff
        cutoff_transition_width: cutoff transition width
        atom_sigma: width of atomic Gaussians
        cutoff_scale: cutoff decay scale
        cutoff_rate: cutoff decay rate
        cutoff_dexp: cutoff decay exponent
        covariance_sigma0: polynomial covariance parameter
        central_weight: weight of central atom
        basis_error_exponent: 10^(-x), difference between target and expansion
        normalise: normalize the SOAP vectors
        quippy_average: compute average of SOAP vectors with quippy 
            (one per Atoms object)
        average: compute average of SOAP vectors over central atoms
        central_reference_all_species: use Gaussian reference for all species
        diagonal_radial: return only n1 = n2 elements of power spectrum
        component_idxs: indices of SOAP components to retain
        concatenate: concatenate SOAP vectors from all structures into a single array
        chunks: if concatenate is True, chunk shape for HDF5
        output: output file for hdf5

        ---Returns---
        soaps: (if output=None) soap vectors
        output: (if output is not None) output hdf5 file
    """
    
    # Number of central atom species
    n_Z = len(Z)

    # Number of environment species
    n_species_Z = len(species_Z)

    # Central atom species
    Z_list = [str(ZZ) for ZZ in Z]
    Z_str = "{{{:s}}}".format(' '.join(Z_list))

    # Environment species
    species_Z_list = [str(zz) for zz in species_Z]
    species_Z_str = "{{{:s}}}".format(' '.join(species_Z_list))

    # Build string of SOAP parameters
    soap_str = [
        "soap",
        f"Z={Z_str:s}",
        f"n_Z={n_Z:d}",
        f"species_Z={species_Z_str:s}",
        f"n_species={n_species_Z:d}",
        f"n_max={n_max:d}",
        f"l_max={l_max:d}",
        f"cutoff={cutoff:f}",
        f"cutoff_transition_width={cutoff_transition_width:f}",
        f"atom_sigma={atom_sigma:f}",
        f"cutoff_scale={cutoff_scale:f}",
        f"cutoff_rate={cutoff_rate:f}",
        f"cutoff_dexp={cutoff_dexp:d}",
        f"covariance_sigma0={covariance_sigma0:f}",
        f"central_weight={central_weight:f}",
        f"basis_error_exponent={basis_error_exponent:f}",
        f"normalise={normalise:s}",
        f"average={quippy_average:s}",
        f"central_reference_all_species={central_reference_all_species:s}",
        f"diagonal_radial={diagonal_radial:s}"
    ]
    soap_str = ' '.join(soap_str)

    # Setup the descriptor
    descriptor = Descriptor(soap_str)

    # Write SOAP vectors to file
    if output is not None:

        # Initialize HDF5
        h = h5py.File(output, mode='w')

        # Add metadata
        # Have to set attributes individually;
        # can't set as a whole dictonary at once
        # TODO: do quippy descriptors have accessible parameter dictionaries?
        h.attrs['Z'] = Z 
        h.attrs['n_Z'] = n_Z
        h.attrs['species_Z'] = species_Z 
        h.attrs['n_species'] = n_species_Z
        h.attrs['n_max'] = n_max
        h.attrs['l_max'] = l_max
        h.attrs['cutoff'] = cutoff
        h.attrs['cutoff_transition_width'] = cutoff_transition_width
        h.attrs['atom_sigma'] = atom_sigma
        h.attrs['cutoff_scale'] = cutoff_scale
        h.attrs['cutoff_rate'] = cutoff_rate
        h.attrs['cutoff_dexp'] = cutoff_dexp
        h.attrs['covariance_sigma0'] = covariance_sigma0
        h.attrs['central_weight'] = central_weight
        h.attrs['basis_error_exponent'] = basis_error_exponent
        h.attrs['normalise'] = normalise
        h.attrs['quippy_average'] = quippy_average
        h.attrs['central_reference_all_species'] = \
                central_reference_all_species 
        h.attrs['diagonal_radial'] = diagonal_radial
        h.attrs['average'] = average
        if component_idxs is not None:
            h.attrs['component_idxs'] = component_idxs
        else:
            h.attrs['component_idxs'] = 'all'

        # Number of digits for structure numbers
        n_digits = len(str(len(structures) - 1))

        # Compute SOAP vectors
        if concatenate:
            if component_idxs is not None:
                n_features = len(component_idxs)
            else:
                n_features = descriptor.ndim

            if average or quippy_average:
                n_centers = len(structures)
            else:
                n_centers = 0
                for z in Z:
                    n_centers += np.sum(
                        [np.count_nonzero(s.numbers == z) for s in structures]
                    )

            dataset = h.create_dataset(
                '0', shape=(n_centers, n_features), 
                chunks=chunks, dtype='float64'
            )

            n = 0
            for sdx, structure in enumerate(tqdm(structures)):
                soap = descriptor.calc(structure,
                        cutoff=descriptor.cutoff())['data']
                soap = _truncate_average(soap, component_idxs=component_idxs,
                        average=average)
                if soap.ndim == 1:
                    soap = np.reshape(soap, (1, -1))
                dataset[n:n + len(soap)] = soap
                n += len(soap)

        else:
            for sdx, structure in enumerate(tqdm(structures)):
                soap = descriptor.calc(structure, 
                        cutoff=descriptor.cutoff())['data']
                soap = _truncate_average(soap, component_idxs=component_idxs,
                        average=average)
                dataset = h.create_dataset(str(sdx).zfill(n_digits), data=soap)

        # Close output file
        h.close()

        return output

    # SOAP vectors in memory
    else:
        soaps = [] 

        # Compute SOAP vectors
        for structure in tqdm(structures):
            soap = descriptor.calc(structure, 
                    cutoff=descriptor.cutoff())['data']
            soap = _truncate_average(soap, component_idxs=component_idxs,
                    average=average)
            soaps.append(soap)

        if concatenate:
            soaps = np.vstack(soaps)

        return soaps

# TODO: make Z optional -- if not given, use all species
# TODO: pass all SOAP parameters as kwargs? Then need to
# find a way to reliably write all parameters to the HDF5 attributes
def librascal_soap(structures, Z, max_radial=6, max_angular=6, 
        interaction_cutoff=3.0, cutoff_smooth_width=0.5,
        gaussian_sigma_constant=0.5, soap_type='PowerSpectrum', 
        cutoff_function_type='ShiftedCosine', gaussian_sigma_type='Constant',
        radial_basis='GTO', expansion_by_species_method='environment wise',
        global_species=None, compute_gradients=False,
        inversion_symmetry=True, normalize=True,
        optimization_args={}, cutoff_function_parameters={},
        coefficient_subselection=None,
        average=False, component_idxs=None, concatenate=False, 
        chunks=None, output=None):
    """
        Compute SOAP vectors with Librascal

        ---Arguments---
        structures: list of ASE Atoms objects
        Z: list of central atom species (atomic number)
            All species are used as environment atoms
        max_radial: number of radial basis functions
        max_angular: highest angular momentum number in spherical harmonics expansion
        cutoff: radial cutoff
        cutoff_width: distance the cutoff is smoothed to zero
        gaussian_sigma_constant: atomic Gaussian widths
        soap_type: type of representation
        cutoff_function_type: type of cutoff function
        gaussian_sigma_type: fixed atomic Gaussian widths ('Constant'),
            vary by species ('PerSpecies'), 
            or vary by distance from the central atom ('Radial')
        radial_basis: basis to use for the radial part
        inversion_symmetry: enforce inversion variance
        normalize: normalize SOAP vectors
        compute_gradients: compute gradients of the SOAP vectors
        expansion_by_species_method: setup of the species
        global_species: list of species that will obey the species setup
        cutoff_function_parameters: additional parameters for the cutoff function
        optimization_args: optimization parameters
        coefficient_subselection: list of species, n, and l indicies to select
        average: average SOAP vectors over the central atoms in a structure
        component_idxs: indices of SOAP components to retain; 
            discard all other components
        concatenate: concatenate SOAP vectors from all structures into a single array
        chunks: if concatenate is True, chunk shape for HDF5
        output: output file for hdf5

        ---Returns---
        soaps: (if output=None) soap vectors
        output: (if output is not None) output hdf5 file
    """

    # Center and wrap the frames
    # TODO: need safeguard for concatenate=True,
    # where an error is raised if not all structures
    # have the same species
    structures_copy = deepcopy(structures)
    species_list = []
    for structure in structures_copy:
        structure.center()
        structure.wrap(eps=1.0E-12)

        # Mask central atoms
        mask_center_atoms_by_species(structure, species_select=Z)

        # Extract environment species
        structure_species = np.unique(structure.numbers)
        species_list.extend(np.setdiff1d(structure_species, species_list))

    species_list.sort()

    # Setup the descriptor
    descriptor = SphericalInvariants(
        soap_type=soap_type,
        max_radial=max_radial,
        max_angular=max_angular,
        interaction_cutoff=interaction_cutoff,
        cutoff_smooth_width=cutoff_smooth_width,
        cutoff_function_type=cutoff_function_type,
        gaussian_sigma_constant=gaussian_sigma_constant,
        gaussian_sigma_type=gaussian_sigma_type,
        radial_basis=radial_basis,
        global_species=global_species,
        expansion_by_species_method=expansion_by_species_method,
        normalize=normalize,
        inversion_symmetry=inversion_symmetry,
        compute_gradients=compute_gradients,
        optimization_args=optimization_args,
        cutoff_function_parameters=cutoff_function_parameters,
        coefficient_subselection=coefficient_subselection
    )

    # Write SOAP vectors to file
    if output is not None:

        # Initialize HDF5
        h = h5py.File(output, mode='w')

        # Add metadata
        # Have to set attributes individually;
        # can't set as a whole dictonary at once
        h.attrs['Z'] = Z 
        for key, value in descriptor._get_init_params().items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    h.attrs[f'{key}:{subkey}'] = subvalue
            else:
                h.attrs[key] = value
        # TODO: remove `coefficient_subselection` once it makes its way into _get_init_params
        if coefficient_subselection is None:
            coefficient_subselection = []
        h.attrs['coefficient_subselection'] = coefficient_subselection
        h.attrs['average'] = average
        if component_idxs is not None:
            h.attrs['component_idxs'] = component_idxs
        else:
            h.attrs['component_idxs'] = 'all'

        # Number of digits for structure numbers
        n_digits = len(str(len(structures) - 1))

        # Compute SOAP vectors
        if concatenate:
            if component_idxs is not None:
                n_features = len(component_idxs)
            else:
                n_features = descriptor.get_num_coefficients(len(species_list))

            if average:
                n_centers = len(structures)
            else:
                n_centers = 0
                for z in Z:
                    n_centers += np.sum(
                        [np.count_nonzero(s.numbers == z) for s in structures]
                    )

            dataset = h.create_dataset(
                '0', shape=(n_centers, n_features), 
                chunks=chunks, dtype='float64'
            )

            n = 0
            for sdx, structure in enumerate(tqdm(structures_copy)):
                soap_rep = descriptor.transform(structure)
                soap = soap_rep.get_features(descriptor)
                soap = _truncate_average(soap, component_idxs=component_idxs,
                        average=average)
                if soap.ndim == 1:
                    soap = np.reshape(soap, (1, -1))
                dataset[n:n + len(soap)] = soap
                n += len(soap)
        else:
            for sdx, structure in enumerate(tqdm(structures_copy)):
                soap_rep = descriptor.transform(structure)
                soap = soap_rep.get_features(descriptor)
                soap = _truncate_average(soap, component_idxs=component_idxs,
                        average=average)
                dataset = h.create_dataset(str(sdx).zfill(n_digits), data=soap)

        # Close output file
        h.close()

        return output

    # SOAP vectors in memory
    else:
        soaps = []    

        # Compute SOAP vectors
        for structure in tqdm(structures_copy):
            soap_rep = descriptor.transform(structure)
            soap = soap_rep.get_features(descriptor)
            soap = _truncate_average(soap, component_idxs=component_idxs,
                    average=average)
            soaps.append(soap)

        if concatenate:
            soaps = np.vstack(soaps)

        return soaps

def gto_sigma(cutoff, n, n_max):
    """
        Compute GTO sigma

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---Arguments---
        cutoff: environment cutoff
        n: order of the GTO
        n_max: maximum order of the GTO

        ---Returns---
        sigma: GTO sigma parameter
    """
    return np.maximum(np.sqrt(n), 1) * cutoff / n_max

def gto_width(sigma):
    """
        Compute GTO width

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---Arguments---
        sigma: GTO sigma parameter

        ---Returns---
        b: GTO (Gaussian) width
    """
    return 1.0 / (2 * sigma ** 2)

def gto_prefactor(n, sigma):
    """
        Compute GTO prefactor

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---Arguments---
        n: order of the GTO
        sigma: GTO sigma parameter

        ---Returns---
        N: GTO prefactor (normalization factor)
    """
    return np.sqrt(2 / (sigma ** (2 * n + 3) * gamma(n + 1.5)))

def gto(r, n, sigma):
    """
        Compute GTO

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---Arguments---
        r: grid on which to evaluate the GTO
        n: order of the GTO
        sigma: GTO sigma parameter

        ---Returns---
        R_n: GTO of order n evaluated on the provided grid
    """
    b = gto_width(sigma)
    N = gto_prefactor(n, sigma)
    return N * r ** (n + 1) * np.exp(-b * r ** 2) # why n+1?

def gto_overlap(n, m, sigma_n, sigma_m):
    """
        Compute overlap of two GTOs

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---Arguments---
        n: order of the first GTO
        m: order of the second GTO
        sigma_n: sigma parameter of the first GTO
        sigma_m: sigma parameter of the second GTO

        ---Returns---
        S: overlap of the two GTOs
    """
    b_n = gto_width(sigma_n)
    b_m = gto_width(sigma_m)
    N_n = gto_prefactor(n, sigma_n)
    N_m = gto_prefactor(m, sigma_m)
    nm = 0.5 * (3 + n + m)
    return 0.5 * N_n * N_m * (b_n + b_m) ** (-nm) * gamma(nm) # why 0.5?

def orthogonalized_gto(cutoff, n_max, r_grid):
    """
        Compute orthogonalized GTOs

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---Arguments---
        cutoff: interaction cutoff
        n_max: maximum number of radial functions
        r_grid: grid of radial distances on which to evaluate the GTOs

        ---Returns---
        R_n: orthogonalized GTOs evaluated on `r_grid`
    """

    # Setup grids of the expansion orders
    n_grid = np.arange(0, n_max)
    sigma_grid = gto_sigma(cutoff, n_grid, n_max)
    
    # Compute radial normalization factor based on the GTO overlap
    S = gto_overlap(
        n_grid[:, np.newaxis],
        n_grid[np.newaxis, :],
        sigma_grid[:, np.newaxis],
        sigma_grid[np.newaxis, :]
    )
    S = fractional_matrix_power(S, -0.5)
    
    # Compute GTOs, shape (n_max, len(r_grid))
    R_n = np.matmul(S, gto(
        r_grid[np.newaxis, :],
        n_grid[:, np.newaxis],
        sigma_grid[:, np.newaxis]
    ))

    return R_n

def legendre_polynomials(l, x):
    """
        Evaluate Legendre Polynomials

        ---Arguments---
        l: order of the Legendre polynomial
        x: grid on which to compute the legendre polynomials,
            must be on [-1, 1]

        ---Returns---
        P_l: Legendre polynomial of order l computed on the provided grid
    """
    return eval_legendre(l, x)

def dvr(cutoff, n_max, gaussian_sigma, r_grid):
    """
        Compute DVR polynomials

        Adapted from a routine originally 
        written by Alexander Goscinski

        ---References--- 
        J. C. Light and T. Carrington, Jr.
        Discrete-variable representations and their utilization,
        Advances in Chemical Physics 114, 263--310 (2000) [Section 2]

        ---Arguments---
        cutoff: interaction cutoff
        n_max: maximum number of radial functions
        gaussian_sigma: Gaussian width for DVR polynomial cutoff
        r_grid: radial grid on which to compute the polynomials

        ---Returns---
        R_n: DVR radial basis functions
    """

    pts, wts, _ = roots_legendre(n_max)
    
    grid_cutoff = cutoff + 3 * gaussian_sigma
    # TODO: check that the cutoff and normalization to [-1, 1] is correct
    # TODO: must r_grid already have the correct cutoff?
    grid = (r_grid - grid_cutoff / 2) / (grid_cutoff / 2)

    legendre_polynomials = [
        np.sqrt((2 * n + 1) / 2) * legendre(n) for n in range(0, n_max)
    ]

    # TODO: clean this up, double for loop over shape of T
    T = np.array(
        [
            [
                np.sqrt(wts[n]) * legendre_polynomials[m](pts[n]) 
                for n in range(0, n_max)
            ] 
            for m in range(0, n_max)
        ]
    )

    legendre_polynomials = np.array([
        legendre_polynomials[n](grid) for n in range(0, n_max)
    ])

    # TODO: get shape
    R_n = T.T @ legendre_polynomials
    return R_n

def reshape_soaps(soaps, n_pairs, n_max, l_max=None):
    """
        Reshape a SOAP vector to have the shape
        (n_centers, n_species_pairs, n_max, n_max, l_max+1)
        in the case of the power spectrum, and
        (n_centers, n_species, n_max)
        in the case of the radial spectrum

        ---Arguments---
        soaps: soap vectors to reshape, size (n_centers, n_features).
            n_features must equal n_pairs * n_max ** 2 * (l_max + 1)
            for the power spectrum, and n_pairs * n_max for the radial spectrum
        n_pairs: for the power spectrum, 
            the number of unique pairings of the environment species
            used to build the SOAP vector. For the radial spectrum,
            the number of environment species
        n_max: maximum order of the radial GTO
        l_max: maximum order of the angular Legendre polynomials.
            If None, reshapes SOAPs for the radial spectrum

        ---Returns---
        soap: reshaped SOAP with shape 
            (n_centers, n_pairs, n_max, n_max, l_max+1)
            for the power spectrum, or shape 
            (n_centers, n_pairs, n_max) for the radial spectrum
    """
    
    # Reshape for power spectrum
    if l_max is not None:
        if soaps.ndim == 1:
            return np.reshape(soaps, (1, n_pairs, n_max, n_max, l_max+1))
        else:
            return np.reshape(soaps, (soaps.shape[0], n_pairs, n_max, n_max, l_max+1))

    # Reshape for radial spectrum
    else:
        if soaps.ndim == 1:
            return np.reshape(soaps, (1, n_pairs, n_max))
        else:
            return np.reshape(soaps, (soaps.shape[0], n_pairs, n_max))

def compute_soap_density(
    soaps, 
    cutoff, 
    n_max, 
    r_grid, 
    l_max=None, 
    p_grid=None, 
    chunk_size_r=0, 
    chunk_size_p=0, 
    radial_basis='GTO', 
    gaussian_sigma=0.5
):
    """
        Compute SOAP density

        ---Arguments---
        soaps: soap vectors on which to compute the density,
            must have the shape (n_centers, n_pairs, n_max, n_max, l_max+1)
            for the power spectrum, and (n_centers, n_species, n_max),
            where n_pairs is the number of unique environment species pairings
            and n_species is the number of environment species 
            (see reshape_soaps)
        cutoff: environment cutoff
        r_grid: grid on which to compute the GTOs
        p_grid: grid on which to compute the Legendre polynomials
        n_max: maximum order of the radial GTO
        l_max: maximum order of the Legendre polynomials.
            If None, do density for radial spectrum
        chunk_size_r: if > 0, compute density in GTO-grid-based chunks
        chunk_size_p: if > 0, compute density in Legendre-polynomial-based chunks
        radial_basis: which radial basis to use ('GTO' or 'DVR')
        gaussian_sigma: Gaussian width for DVR polynomials

        ---Returns---
        density: SOAP reconstructed density with shape
            (n_centers, n_pairs, len(r_grid), len(r_grid), len(p_grid))
            for the power spectrum, and
            (n_centers, n_species, len(r_grid))
            for the radial spectrum
    """
    
    if radial_basis == 'GTO':
        R_n = orthogonalized_gto(cutoff, n_max, r_grid)
    elif radial_basis == 'DVR':
        R_n = dvr(cutoff, n_max, gaussian_sigma, r_grid)
    else:
        print("Error: radial_basis must be one of 'GTO' or 'DVR'")
        return

    # Set up the grid-based chunking to speed
    # up the density computation and reduce memory requirements
    if chunk_size_r <= 0:
        n_chunks_r = 1
        chunk_size_r = len(r_grid)
    else:
        n_chunks_r = len(r_grid) // chunk_size_r
        if len(r_grid) % chunk_size_r > 0:
            n_chunks_r += 1
    
    # Do radial spectrum
    if l_max is None:

        # Compute the density in grid-based chunks
        density = np.zeros((soaps.shape[0], soaps.shape[1], len(r_grid)))
            
        for n in range(0, n_chunks_r):
            slice_n = slice(n * chunk_size_r, (n + 1) * chunk_size_r, 1)
            r_n = R_n[:, slice_n]
            density[:, :, slice_n] = np.tensordot(soaps, r_n, axes=1)
    
    # Do power spectrum
    else:
        l_grid = np.arange(0, l_max + 1)

        # Compute Legendre polynomials, shape (l_max+1, len(p_grid))
        P_l = legendre_polynomials(l_grid[:, np.newaxis],
                                   p_grid[np.newaxis, :])
    
        if chunk_size_p <= 0:
            n_chunks_p = 1
            chunk_size_p = len(p_grid)
        else:
            n_chunks_p = len(p_grid) // chunk_size_p
            if len(p_grid) % chunk_size_p > 0:
                n_chunks_p += 1
                
        # Compute the density in grid-based chunks
        density = np.zeros((soaps.shape[0], soaps.shape[1], 
                            len(r_grid), len(r_grid), len(p_grid)))
            
        for n in range(0, n_chunks_r):
            for m in range(0, n_chunks_r):
                for p in range(0, n_chunks_p):
                    slice_n = slice(n * chunk_size_r, (n + 1) * chunk_size_r, 1)
                    slice_m = slice(m * chunk_size_r, (m + 1) * chunk_size_r, 1)
                    slice_p = slice(p * chunk_size_r, (p + 1) * chunk_size_p, 1)
                    r_n = np.reshape(R_n[:, slice_n], (n_max, 1, 1, -1, 1, 1))
                    r_m = np.reshape(R_n[:, slice_m], (1, n_max, 1, 1, -1, 1))
                    p_l = np.reshape(P_l[:, slice_p], (1, 1, l_max + 1, 1, 1, -1))
                    density[:, :, slice_n, slice_m, slice_p] = \
                            np.tensordot(soaps, r_n * r_m * p_l, axes=3)
                
    return density

def rrw_neighbors(frame, center_species, env_species, cutoff, self_interaction=False):
    """
        Compute the neighbor list for every atom of the central atom species
        and generate the r, r', w for each pair of neighbors

        ---Arguments---
        frame: atomic structure
        center_species: species of atoms to use as centers
        env_species: species of atoms to include in the environment
        cutoff: atomic environment cutoff
        self_interaction: include the central atom as its own neighbor

        ---Returns---
        rrw: list of a list of numpy 3D numpy arrays.
            The outer list corresponds to the atom centers, 
            and the inner list corresponds to the species groupings
            and contains several numpy arrays.
            Each numpy array is of shape (3, n_neighbors_a, n_neighbors_b),
            where the axes are organized as follows:
            axis=0: distances to neighbor A from the central atom
            axis=1: distances to neighbor B from the central atom
            axis=2: angle between the distance vectors to neighbors A and B 
            from the central atom
        idxs: same structure as rrw, but holds the indices of the atoms involved 
            in the tuple, i.e.,
            axis=0: index of central atom
            axis=1: index of neighbor A
            axis=2: index of neighbor B
    """
    # TODO: generalize to work also with the radial spectrum

    # Extract indices of central atoms and environment atoms
    center_species_idxs = [np.nonzero(frame.numbers == i)[0] for i in center_species]
    env_species_idxs = [np.nonzero(frame.numbers == i)[0] for i in env_species]

    # Build neighbor list for all atoms
    nl = {}
    nl['i'], nl['j'], nl['d'], nl['D'] = \
            neighbor_list('ijdD', frame, cutoff, self_interaction=self_interaction)

    rrw = []
    idxs = []

    # Loop over centers grouped by species
    # TODO: maybe generalize this so that when using multiple
    # central atom species the ordering isn't grouped by species,
    # but instead corresponds to the ordering in the ASE Atoms object
    for center_idxs in center_species_idxs:
        for center in center_idxs:

            # Build subset of neighbor list that just has the neighbors of
            # the center
            center_nl_idxs = np.nonzero(nl['i'] == center)[0]
            nl_center = {}
            for k, v in nl.items():
                nl_center[k] = v[center_nl_idxs]

            rrw_species = []
            idxs_species = []

            # Loop over combinations of environment species
            for env_species_a, env_species_b in \
                    itertools.combinations_with_replacement(env_species_idxs, 2):
                a = np.nonzero(np.isin(nl_center['j'], env_species_a))[0]
                b = np.nonzero(np.isin(nl_center['j'], env_species_b))[0]

                # Extract distances to neighbors from the central atom (r, r')
                da = nl_center['d'][a]
                db = nl_center['d'][b]
                Da = nl_center['D'][a]
                Db = nl_center['D'][b]
                r_n, r_m = np.meshgrid(da, db, indexing='ij')

                # Compute angles between neighbors and central atom (w)
                D = np.matmul(Da, Db.T)
                d = np.outer(da, db)
                d[d <= 0.0] = 1.0
                w = D / d

                # Extract indices of the atoms in the rr'w triplet
                ia = nl_center['j'][a]
                ib = nl_center['j'][b]
                j_n, j_m = np.meshgrid(ia, ib, indexing='ij')
                j_center = np.full(j_n.shape, center, dtype=int)

                # Build 3D matrix of rr'w triplets
                rrw_species.append(np.stack((r_n, r_m, w)))
                idxs_species.append(np.stack((j_center, j_n, j_m)))

            rrw.append(rrw_species)
            idxs.append(idxs_species)

    return rrw, idxs

def make_tuples(data):
    """
        Take a list of lists of rr'w formatted 3D arrays (see rrw_neighbors)
        and reshape into a list of lists of 2D arrays of shape (n_neighbor_pairs, 3),
        where each row is a rr'w triplet and the columns are in the order r, r', w

        ---Arguments---
        data: list of lists of arrays to "reshape"

        ---Returns---
        center_tuple: "reshaped" data list
    """
    n_centers = len(data)
    center_tuple = []

    # Loop over centers
    for nctr in range(0, n_centers):
        n_pairs = len(data[nctr])
        pair_tuple = []

        # Loop over species pairs
        for npr in range(0, n_pairs):
            data_shape = np.shape(data[nctr][npr])

            # Reshape the 3D array to a 2D array
            tuple_array = np.reshape(np.moveaxis(data[nctr][npr], 0, -1),
                                     (np.prod(data_shape[1:]), data_shape[0]))

            pair_tuple.append(tuple_array)

        center_tuple.append(pair_tuple)

    return center_tuple

def extract_species_pair_groups(n_features, n_species, 
        spectrum_type='power', combinations=False):
    """
        Extract the librascal SOAP feature indices grouped according
        to species pairing

        ---Arguments---
        n_features: number of features of the SOAP vector
        n_species: number of species (in the environment) of the SOAP vector
        spectrum_type: extract the feature groups for the power spectrum
            ('power') or the radial spectrum ('radial')
        combinations: whether to also include combinations of species pairs
            in the feature index groups

        ---Returns---
        feature_groups: list of arrays of indices corresponding to the
            species groupings
    """

    if spectrum_type == 'power':
        n_species_pairs = n_species * (n_species + 1) / 2

    elif spectrum_type == 'radial':
        n_species_pairs = n_species

    else:
        print("Error: invalid spectrum type; use 'power' or 'radial'")
        return

    if n_features % n_species != 0:
        print("Error: number of features incompatible with number of species")
        return

    feature_idxs = np.arange(0, n_features, dtype=int)
    feature_groups = np.split(feature_idxs, n_species_pairs)

    if combinations:
        feature_groups_combinations = []
        for i in range(2, len(feature_groups) + 1):
            for j in itertools.combinations(feature_groups, i):
                feature_groups_combinations.append(np.sort(np.concatenate(j)))

        feature_groups.extend(feature_groups_combinations)

    return feature_groups

def species_pairings(species_list):
    """
        Get a list of species pairings. The order of the pairings
        will correspond to the feature groups from `extract_species_pair_groups`,
        as librascal sorts the pairings by atomic number

        ---Arguments---
        species_list: list of atomic numbers

        ---Returns---
        species_pairings: list of tuples of atomic numbers,
            with each tuple containing the atomic numbers in the pairing
    """

    sorted_species = sorted(species_list)
    return list(itertools.combinations_with_replacement(sorted_species, 2))

