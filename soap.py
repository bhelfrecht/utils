#!/usr/bin/env python

import os
import sys
import numpy as np
from quippy.descriptors import Descriptor
import h5py
from tqdm import tqdm
from rascal.representations import SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_species

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
def quippy_soap(structures, Z, species_Z, n_max=6, l_max=6, cutoff=3.0, 
        cutoff_transition_width=0.5, atom_sigma=0.1,
        cutoff_scale=1.0, cutoff_rate=1.0, cutoff_dexp=0,
        covariance_sigma0=0.0, central_weight=1.0, basis_error_exponent=10.0,
        normalise=True, central_reference_all_species=False, 
        diagonal_radial=False, quippy_average=False,
        average=False, component_idxs=None, output=None):
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
    soap_str = ["soap",
            "Z={:s}".format(Z_str),
            "n_Z={:d}".format(n_Z),
            "species_Z={:s}".format(species_Z_str),
            "n_species={:d}".format(n_species_Z),
            "n_max={:d}".format(n_max),
            "l_max={:d}".format(l_max),
            "cutoff={:f}".format(cutoff),
            "cutoff_transition_width={:f}".format(cutoff_transition_width),
            "atom_sigma={:f}".format(atom_sigma),
            "cutoff_scale={:f}".format(cutoff_scale),
            "cutoff_rate={:f}".format(cutoff_rate),
            "cutoff_dexp={:d}".format(cutoff_dexp),
            "covariance_sigma0={:f}".format(covariance_sigma0),
            "central_weight={:f}".format(central_weight),
            "basis_error_exponent={:f}".format(basis_error_exponent),
            "normalise={:s}".format(str(normalise)),
            "average={:s}".format(str(quippy_average)),
            "central_reference_all_species={:s}".format(str(central_reference_all_species)),
            "diagonal_radial={:s}".format(str(diagonal_radial))]
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

        return soaps

# TODO: make Z optional -- if not given, use all species
def librascal_soap(structures, Z, max_radial=6, max_angular=6, 
        interaction_cutoff=3.0, cutoff_smooth_width=0.5,
        gaussian_sigma_constant=0.5, soap_type='PowerSpectrum', 
        cutoff_function_type='ShiftedCosine', gaussian_sigma_type='Constant',
        inversion_symmetry=True, normalize=True,
        average=False, component_idxs=None, output=None):
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
            vary by species ('PerSpecies'), or vary by distance from the central atom ('Radial')
        inversion_symmetry: enforce inversion variance
        normalize: normalize SOAP vectors
        average: average SOAP vectors over the central atoms in a structure
        component_idxs: indices of SOAP components to retain; discard all other components
        output: output file for hdf5

        ---Returns---
        soaps: (if output=None) soap vectors
        output: (if output is not None) output hdf5 file
    """

    # Center and wrap the frames
    structures_copy = structures.copy()
    for structure in structures_copy:
        structure.center()
        structure.wrap(eps=1.0E-12)

        # Mask central atoms
        mask_center_atoms_by_species(structure, species_select=Z)

    # Setup the descriptor
    descriptor = SphericalInvariants(soap_type=soap_type,
            max_radial=max_radial,
            max_angular=max_angular,
            interaction_cutoff=interaction_cutoff,
            cutoff_smooth_width=cutoff_smooth_width,
            cutoff_function_type=cutoff_function_type,
            gaussian_sigma_constant=gaussian_sigma_constant,
            gaussian_sigma_type=gaussian_sigma_type)

    # Write SOAP vectors to file
    if output is not None:

        # Initialize HDF5
        h = h5py.File(output, mode='w')

        # Add metadata
        # Have to set attributes individually;
        # can't set as a whole dictonary at once
        h.attrs['Z'] = Z 
        h.attrs['soap_type'] = soap_type
        h.attrs['max_radial'] = max_radial
        h.attrs['max_angular'] = max_angular
        h.attrs['interaction_cutoff'] = interaction_cutoff
        h.attrs['cutoff_smooth_width'] = cutoff_smooth_width
        h.attrs['cutoff_function_type'] = cutoff_function_type
        h.attrs['gaussian_sigma_constant'] = gaussian_sigma_constant
        h.attrs['gaussian_sigma_type'] = gaussian_sigma_type
        h.attrs['average'] = average
        if component_idxs is not None:
            h.attrs['component_idxs'] = component_idxs
        else:
            h.attrs['component_idxs'] = 'all'

        # Number of digits for structure numbers
        n_digits = len(str(len(structures) - 1))

        # Compute SOAP vectors
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

        return soaps
