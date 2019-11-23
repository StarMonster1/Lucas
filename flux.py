#!/usr/bin/env python3
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from functions import import_fits


def numflux(disk, reduced):
    """Calculate the numerical flux generateed by interpolation while rotating
    the disk in percentages of the original flux.

    Args
    -------
        disk: generated disk with different angles
        reduced: disk which went through ADI

    Returns
    -------
        numerical_flux: the numerical flux in percentages of oringal flux

    """
    middle_index = len(disk) // 2 - 1
    original_flux = np.sum(disk[middle_index])

    disk_diff = reduced - disk[middle_index]
    disk_diff[disk_diff < 0] = 0
    iteration_flux = np.sum(disk_diff, axis=(1, 2))
    numerical_flux = iteration_flux/original_flux * 100

    return numerical_flux


def recovflux(disk, reduced):
    """Calculate recovered flux in percentages of the original amount of flux.

    Args
    -------
        disk: generated disk with different angles
        reduced: disk which went through ADI

    Returns
    -------
        recovered_flux: the recovered flux in percentages of oringal flux

    """
    middle_index = len(disk) // 2 - 1
    original_flux = np.sum(disk[middle_index])

    reduced[reduced < 0] = 0
    iteration_flux = np.sum(reduced, axis=(1, 2))
    recovered_flux = iteration_flux/original_flux * 100

    return recovered_flux


def main():
    disk = import_fits('Fits/disk')
    disk_reduced = import_fits('Fits/disk_open_cv_reduced')

    numerical_flux = numflux(disk, disk_reduced)
    np.save('../flux_files/numflux', numerical_flux)


if __name__ == '__main__':
    main()
