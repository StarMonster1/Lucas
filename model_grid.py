#!/usr/bin/env python3
"""Grid of reduced disks with different inclinations and declinations."""
import numpy as np
import matplotlib.pyplot as plt
import datetime

import disk_structure as ds
import functions as fn
import adi
import flux


def plot_grid(image_grid, flux_grid, dec_arr, i_arr):
    """Plot a grid of 2D and 1D arrays from two 4D arrays.

    Args
    -------
        image_grid: 4D array with (len(dec_arr) x len(i_arr)) 2D arrays of the
            recovered disks after niter iterations
        flux_grid: 4D array with (2 x len(dec_arr) x len(i_arr)) 1D arrays of
            the numerical flux and recovered flux
        dec_arr: array of different declinations
        i_arr: array of different inclinations

    Returns
    -------
        None

    """
    fig, ax = plt.subplots(nrows=len(dec_arr)*2, ncols=len(i_arr),
                           figsize=(20, 40))

    for (k,), dec in np.ndenumerate(dec_arr):
        for (m,), i in np.ndenumerate(i_arr):

            recovered_flux = flux_grid[k, m, 1, :]
            numerical_flux = flux_grid[k, m, 0, :]
            corrected_flux = flux_grid[k, m, 1, :] - flux_grid[k, m, 0, :]

            x = np.arange(1, len(numerical_flux) + 1, 1)

            ax[2*k, m].imshow(image_grid[k, m, :, :], vmin=0, vmax=1,
                              origin='lower')
            ax[2*k+1, m].plot(x, recovered_flux, lw=4, label='Recovered flux '
                              '= {:.3} %'.format(recovered_flux[-1]))
            ax[2*k+1, m].plot(x, numerical_flux, lw=4, label='Numerical flux '
                              '= {:.3} %'.format(numerical_flux[-1]))
            ax[2*k+1, m].plot(x, corrected_flux, lw=4, label='Corrected flux '
                              '= {:.3} %'.format(corrected_flux[-1]))
            ax[2*k+1, m].set_ylim(top=110)
            ax[2*k, m].set_yticks([])
            ax[2*k, m].set_xticks([])
            ax[2*k+1, m].legend(loc='lower center', bbox_to_anchor=(0.5, 0.9),
                                framealpha=0.9)

            if k == 0:
                ax[2*k, m].set_title('Incl={}$\\degree$'.format(i),
                                     fontsize=40)

            if m == 0:
                ax[2*k, m].set_ylabel('Dec={}$\\degree$'.format(dec),
                                      fontsize=40)
                ax[2*k+1, m].set_ylabel('Flux (%)')

            if 2*k+1 == 7:
                ax[2*k+1, m].set_xlabel('Iteration')

    plt.tight_layout()
    fig.savefig('../Images/image_grid_test_edge_blur.png', dpi=100)


def gen_image_grid(dec_arr, i_arr, r, npix, hour_angles, niter):
    """Generate images of disk with different declinations and inclinations.

    Args
    -------
        dec_arr: array of declinations
        i_arr: array of inclinations
        r: radii of the different rings of the disk
        npix: number of pixels of the image
        hour_angles: array of hour angles
        niter: number of iterations

    Returns
    -------
        image_grid: 4D array with (len(dec_arr) x len(i_arr)) 2D arrays of the
            recovered disks after niter iterations
        flux_grid: 4D array with (2 x len(dec_arr) x len(i_arr)) 1D arrays of
            the numerical flux and recovered flux

    """
    # Making image grid
    step = 1
    nsteps = len(dec_arr) * len(i_arr)
    image_grid = np.zeros([len(dec_arr), len(i_arr), npix, npix])
    flux_grid = np.zeros([len(dec_arr), len(i_arr), 2, niter])
    for (x, ), dec in np.ndenumerate(dec_arr):
        angles = fn.parallactic_angle(hour_angles, dec, lat=-24.6270)

        for (y, ), i in np.ndenumerate(i_arr):
            print('\n---------- ({} / {}) ----------'.format(step, nsteps))
            print('\t   ' + str(datetime.datetime.now().strftime('%H:%M:%S')))
            print(' ')

            disk = ds.disk(r, i, angles, npix=400, light_effects=True,
                           height=False, blur=False)
            disk_reduced = adi.reduce_iter(disk, angles, niter)

            flux_grid[x, y, 0, :] = flux.numflux(disk, disk_reduced)
            flux_grid[x, y, 1, :] = flux.recovflux(disk, disk_reduced)

            image_grid[x, y, :, :] = disk_reduced[-1]
            step += 1

    return image_grid, flux_grid


def main():
    """Generate disks for different inclinations and declinations."""
    # Inlcinations and declinations over which to iterate
    dec_arr = np.array([-30, -50])
    i_arr = np.array([25, 85])

    # parameters for pictures and disk
    npix = 400
    r = np.array([(30, 80), (100, 120), (140, 160)])
    niter = 100
    hour_angles = np.linspace(-7.5, 7.5, 60)

    images, fluxes = gen_image_grid(dec_arr, i_arr, r, npix,
                                    hour_angles, niter)
    plot_grid(images, fluxes, dec_arr, i_arr)


if __name__ == '__main__':
    main()
