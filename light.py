#!/usr/bin/env python3
"""Definitions of effects of light on the disk."""
import numpy as np


def r2_law(array, i):
    """Generate an 1/r^2 law for light.

    Args
    -------
        array: incoming array of the disk to implement the law on
        i: inclination of the disk

    Returns
    -------
        light * array: the incoming array multiplied with the light 1/r^2
                        dependence array

    """
    npix = np.shape(array)[1]
    M = np.array([int(npix/2), int(npix/2)])
    yy, xx = np.ogrid[-M[0]:npix-M[0], -M[1]:npix-M[1]]

    np.seterr(divide='ignore')
    light = npix / (xx ** 2 + (yy / (np.cos(np.deg2rad(i)))) ** 2)

    light[light == np.inf] = 0
    return light * array


def scattering(array, i):
    """Generate array modeling the scattering angle dependign on viewing angle.

    Args
    -------
        array: incoming array of the disk
        i: inclination of the disk

    Returns
    -------
        scatter * array: the incoming array multiplied with the array
                        containing the dependence of the scattering angle on
                        the viewing angle

    """
    npix = np.shape(array)[1]

    x = np.arange(-1*npix//2, npix//2)
    y = np.arange(-1*npix//2, npix//2)
    xx, yy = np.meshgrid(x, y / np.cos(np.deg2rad(i)))

    zero_vector = np.array([0, -1])

    coords = np.stack((xx, yy))
    dot_prduct = np.dot(np.moveaxis(coords, 0, -1), zero_vector)
    norm_product = np.linalg.norm(coords, axis=0) * np.linalg.norm(zero_vector)
    alpha = np.arccos(dot_prduct/np.ma.masked_array(norm_product,
                                                    norm_product == 0))

    scatter = np.ones([npix, npix])+100*np.cos(0.5*alpha)*np.sin(np.deg2rad(i))

    return scatter * array


def add_star(array, star_data, disk_star_ratio=0.001):
    """Adding a star to the disk model from a real dataset.

    Args
    -------
        array: incoming 3D array of the model of the disk
        star_data: 3D array of the star

    Returns
    -------
        star_addition: added star to model array

    """
    left_bound = np.shape(star_data)[1]//2 - np.shape(array)[1]//2
    right_bound = np.shape(star_data)[1]//2 + np.shape(array)[1]//2

    # Cutting star data into the shape of the model
    star_data = star_data[:, left_bound:right_bound, left_bound:right_bound]
    star_data /= np.amax(star_data)

    star_addition = array * (disk_star_ratio) + star_data * (1-disk_star_ratio)

    return star_addition


def main():
    """Test defnitions mentioned above."""
    pass


if __name__ == '__main__':
    main()
