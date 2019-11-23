#!/usr/bin/env python3
import numpy as np
from scipy.ndimage import gaussian_filter
import functions as fn
import light


def disk(r, i, angles, npix, light_effects=False, height=False, blur=False):
    """Generate disk with multiple rings and rotated some amount of times.

    Args
    -------
        r: Array of inner and outer radii of the rings of the disk
        i: inclination angle in degrees
        angles: angles over which the disk should be rotated in degrees
        npix: number of pixels of the final array (npix x npix)
        light_effects: turn on scattering and r^-2 law effects
        height: turn on height of the disk

    Returns
    -------
        disk: an array with [len(angles), npix, npix] dimensions

    """

    print('Generating disk ...')

    nrings = len(r)

    # Centre of ring and generating values for x and y
    M = np.array([int(npix/2), int(npix/2)])
    y,x = np.ogrid[-M[0]:npix-M[0], -M[1]:npix-M[1]]


    # Masking the part of the ring
    mask = np.zeros([nrings, npix, npix])
    for index in range(nrings):
        if height:
            offset = height_offset(r, i)
            mask[index, :, :] = (fn.circle(x, y, i, offset[index, 0]) >= r[index, 0] ** 2) & \
             (fn.circle(x, y, i, offset[index, 1]) <= r[index, 1] ** 2)
        else:
            mask[index, :, :] = (fn.circle(x, y, i) >= r[index, 0] ** 2) & \
             (fn.circle(x,y, i) <= r[index, 1] ** 2)

    mask = np.any(mask, axis=0)

    # Generating the simulated picture
    disk = np.zeros([npix, npix])
    disk[mask] = 1

    if light_effects:
        disk = light.r2_law(disk, i)
        disk = light.scattering(disk, i)

    # Normalize disk
    disk = disk / np.amax(disk)

    # Rotate disk
    disk = fn.rotate_cv(disk, angles)

    if blur:
        disk = gaussian_filter(disk, 3)

    return disk


def disk_with_star(r, i, npix, starname, disk_star_ratio,
                   light_effects=False, height=False, blur=False):
    """ Generates disk with multiple rings and rotated some amount of times

    Args:
        r: Array of inner and outer radii of the rings of the disk
        i: inclination angle in degrees
        angles: angles over which the disk should be rotated in degrees
        npix: number of pixels of the final array (npix x npix)
        light_effects: turn on scattering and r^-2 law effects
        height: turn on height of the disk

    Returns:
        disk: an array with [len(angles), npix, npix] dimensions

    """

    print('Generating disk ...')

    nrings = len(r)
    angles = np.loadtxt('/home/lucas/Documents/master_thesis/Fits/Stars_test_data/' + starname + '_parangle.txt')
    star_data = fn.import_fits('Fits/Stars_test_data/' + starname + '_frames_combined')

    # Centre of ring and generating values for x and y
    M = np.array([int(npix/2), int(npix/2)])
    y,x = np.ogrid[-M[0]:npix-M[0], -M[1]:npix-M[1]]


    # Masking the part of the ring
    mask = np.zeros([nrings, npix, npix])
    for index in range(nrings):
        if height:
            offset = height_offset(r, i)
            mask[index, :, :] = (fn.circle(x, y, i, offset[index, 0]) >= r[index, 0] ** 2) & \
             (fn.circle(x,y, i, offset[index, 1]) <= r[index, 1] ** 2)
        else:
            mask[index, :, :] = (fn.circle(x, y, i) >= r[index, 0] ** 2) & \
             (fn.circle(x,y, i) <= r[index, 1] ** 2)

    mask = np.any(mask, axis=0)

    # Generating the simulated picture
    disk = np.zeros([npix, npix])
    disk[mask] = 1

    if light_effects:
        disk = light.r2_law(disk, i)
        disk = light.scattering(disk, i)

    # Normalize disk
    disk = disk / np.amax(disk)

    # Rotate disk
    disk = fn.rotate_cv(disk, angles)

    if blur:
        disk = gaussian_filter(disk, 3)

    disk = light.add_star(disk, star_data, disk_star_ratio)

    return disk, angles




def planet(coords, angles, npix, FWHM):
    """ Generating an array with a 2D gaussian at a specific coordinate
    representing a planet

    Args:
        coords: coordinates of the planet with the origin in the middle of the picture
        i: inclination
        angles: rotation angles
        npix: number of pixels (npix x npix) of the picture
        FWHM: full width at half maximum of the 2D gaussian

    Returns:
        picture: a npix x npix array with the planet in it
    """
    picture = fn.gauss_array(npix, FWHM, coords)
    picture = fn.rotate(picture, angles)
    return picture


def height_offset(r, i):
    """ Calculates the ofset depending on the radius and the inclination

    Args:
        r: radii of the rings of the disk
        i: inclination

    Returns:
        u: offset of the centre of the ring of the disk along the semi-minor axis
    """
    h = 0.0064 * r ** 1.73
    u = h * np.sin(np.deg2rad(i))
    return u.astype('int')


def main():
    pass

if __name__ == '__main__':
    main()
