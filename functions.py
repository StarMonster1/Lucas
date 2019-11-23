#!/usr/bin/env python3
"""Different functions for model."""
import numpy as np
from scipy import ndimage
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import cv2 as cv
from skimage import feature


def import_fits(filename):
    """Import fits file and changes it into numpy array.

    Args
    -------
        filename: name of the fits file

    Returns
    -------
        data: numpy array of fits file data

    """
    hdul = fits.open('/home/lucas/Documents/master_thesis/' + filename +
                     '.fits')
    data = hdul[0].data
    hdul.close()

    return data


def circle(x, y, i=0, offset=0):
    """Circle with certain inclination angle i and offset.

    Args
    -------
        x: Array of x coordinates in pixels
        y: Array of y coordinates in pixels
        i: inclination angle in degrees
        offset: offset of centre in pixels

    Returns
    -------
        radius squared of the circle

    """
    i = np.deg2rad(i)
    return x ** 2 + (y / np.cos(i) - offset) ** 2


def rotate_np(array, angles, order=1):
    """Rotate an array over angles.

    Args
    -------
        array: array in either 2 or 3 dimensions which will be rotated
        angles: angles over which array will be rotated
        order: order of the spline interpolation in ndimage.rotate

    Returns
    -------
        array_new: array with extra dimension where 'array' is rotated over
            all angles in 'angles' array
        array: all entries are rotated over the angles in 'angles' array

    """
    if len(np.shape(array)) == 2:
        array_new = np.zeros([len(angles), *np.shape(array)])
        for step, angle in np.ndenumerate(angles):
            array_new[step[0], :, :] = ndimage.rotate(array, angle, reshape=False, order=order)

        return array_new

    else:
        for step, angle in np.ndenumerate(angles):
            array[step[0], :, :] = ndimage.rotate(array[step[0], :, :], angle, reshape = False, order = order)

        return array


def rotate_cv(array, angles):
    """Rotate an array over angles.

    Args
    -------
        array: array in either 2 or 3 dimensions which will be rotated
        angles: angles over which array will be rotated

    Returns
    -------
        array_new: array with extra dimension where 'array' is rotated over
            all angles in 'angles' array
        array: all entries are rotated over the angles in 'angles' array

    """

    if len(np.shape(array)) == 2:
        image_centre = (np.shape(array)[1] // 2, np.shape(array)[0] // 2)
        array_new = np.zeros([len(angles), *np.shape(array)])
        for step, angle in np.ndenumerate(angles):
            rot_mat = cv.getRotationMatrix2D(image_centre, angle, 1)
            array_new[step[0], :, :] = cv.warpAffine(array, rot_mat,
                                                     np.shape(array),
                                                     flags = cv.INTER_NEAREST)

        array_new = blur_edges(array_new)
        return array_new

    else:
        image_centre = (np.shape(array)[2] // 2, np.shape(array)[1] // 2)
        for step, angle in np.ndenumerate(angles):
            rot_mat = cv.getRotationMatrix2D(image_centre, angle, 1)
            array[step[0], :, :] = cv.warpAffine(array[step[0], :, :],
                                                 rot_mat, np.shape(array)[1:],
                                                 flags = cv.INTER_NEAREST)

        array = blur_edges(array)
        return array


def gauss(x, y, FWHM, coords):
    sigma = FWHM / (2 * np.sqrt(2*np.log(2)))
    return 50 / (2 * np.pi * sigma ** 2) * np.exp( - ((x - coords[0]) ** 2 + (y - coords[1]) ** 2) / (2 * sigma ** 2))


def gauss_array(npix, FWHM, coords):
    """ Generates an npix x npix array of a 2D gaussian

    Args:
        npix: number of pixels
        FWHM: Full width at half maximum


    """

    M = np.array([int(npix/2), int(npix/2)])
    y,x = np.ogrid[-M[0]:npix-M[0], -M[1]:npix-M[1]]

    array_gaus = gauss(x, y, FWHM, coords)

    return array_gaus


def parallactic_angle(hour_angle, dec, lat):
    """ Calculates the parallactic angle indegrees using the hour angle and declination of the
    target and the latitude of the observatory

    Args:
        hour_angle: hour angle of the target in degrees
        dec: declination of the target in degrees
        lat: latitude of the observatoty in degrees

    Returns:
        angle: the parallactic angle in degrees

    """
    hour_angle = np.deg2rad(hour_angle)
    dec = np.deg2rad(dec)
    lat = np.deg2rad(lat)

    angle = np.arctan(np.sin(hour_angle)/(np.cos(dec)*np.tan(lat)-np.sin(dec)*np.cos(hour_angle)))

    return np.rad2deg(angle)


def blur_edges(array_cube):
    """Detect edges of incoming image via canny edge detection.

    Args
    -------
        array_cube: array of image

    Returns
    -------
        array_cube: array with edges of original image blurred

    """
    for i, disk in enumerate(array_cube):
        copy = np.copy(disk)
        copy_scaled = 100*(copy - np.amin(copy)) / np.ptp(copy)
        edges = feature.canny(copy_scaled, sigma=3)

        edges += np.roll(edges, 1, axis=1)
        edges += np.roll(edges, 1, axis=0)
        edges += np.roll(edges, -1, axis=1)
        edges += np.roll(edges, -1, axis=0)

        copy_blur = gaussian_filter(copy, 2)
        array_cube[i][edges] = copy_blur[edges]

    return array_cube


def main():
    pass


if __name__ == '__main__':
    main()
