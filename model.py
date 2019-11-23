#!/usr/bin/env python3
"""Main file for generating a disk and reduce it."""
import numpy as np
from astropy.io import fits

# Own imports
import disk_structure as ds
import functions as fn
import adi


def main():
    """Combine functions in other files."""
    r = np.array([(30, 80), (100, 120), (140, 160)])

    hour_angles = np.linspace(-7.5, 7.5, 60)
    angles = fn.parallactic_angle(hour_angles, dec=-30, lat=-24.6270)
    disk = ds.disk(r, 45, angles, npix=400, light_effects=True,
                   height=True, blur=True)

    hdu = fits.PrimaryHDU(disk)
    hdu.writeto('../Fits/disk.fits', overwrite=True)

    adi_picture = adi.reduce_iter(disk, angles, niter=1000)

    hdu = fits.PrimaryHDU(adi_picture)
    hdu.writeto('../Fits/disk_open_cv_reduced.fits', overwrite=True)


if __name__ == '__main__':
    main()
