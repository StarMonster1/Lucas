#!/usr/bin/env python3
"""ADI pipeline."""
import numpy as np
from functions import rotate_cv, blur_edges
from tqdm import tqdm


def reduce_threshold(array, angles, niter, threshold=0):
    """Pipeline for reduction of array.

    Args
    -------
        array: the cube of pictures to be reduced
        angles: angles over which entries of 'array' where rotated
        niter: number of iterations

    Returns
    -------
        C: reduced version of incoming cube of pictures

    """
    print('Started pipeline ...')
    C_rotated = np.zeros(np.shape(array))
    C = np.zeros([niter, *np.shape(array[0])])
    A = np.copy(array)

    for i in tqdm(range(niter)):
        # Subtract previous final image
        C_rotated[C_rotated < threshold] = 0
        A = array - C_rotated

        # Calculate median and subtract it from original array
        B = np.median(A, axis=0)
        Sub = array - B

        # Rotate images back and combine by taking the median
        D = rotate_cv(Sub, -1 * angles)
        C[i] = np.median(D, axis=0)

        # Rotate final image so it can be used at the start again
        C_rotated = rotate_cv(C[i], angles)

    print('\nDone!')
    return C


def reduce_iter(array, angles, niter):
    """Pipeline for reduction of array.

    Args
    -------
        array: the array to be reduced
        angles: angles over which entries of 'array' where rotated
        niter: number of iterations

    Returns
    -------
        C: reduced version of incoming 3D array of pictures

    """
    print('Started pipeline ...')
    C_rotated = np.zeros(np.shape(array))
    C = np.zeros([niter, *np.shape(array[0])])
    A = np.copy(array)

    for i in tqdm(range(niter)):
        # Subtract previous final image
        C_rotated[C_rotated < 0] = 0
        A = array - C_rotated

        # Calculate median and subtract it from original array
        B = np.median(A, axis=0)
        Sub = array - B

        # Rotate images back and combine by taking the median
        D = rotate_cv(Sub, -1 * angles)
        C[i] = np.median(D, axis=0)

        # Rotate final image so it can be used at the start again
        C_rotated = rotate_cv(C[i], angles)

        # Blur edges of disk
        # C_rotated = blur_edges(C_rotated)

    print('\nDone!')
    return C


def reduce_diff(array, angles, delta, miter):
    """Pipeline for reduction of array.

    Args
    -------
        array: the array to be reduced
        angles: angles over which entries of 'array' where rotated
        delta: difference between the sum of the mean of consecutive iterations
                for which the while loop will be stopped
        miter: maximum number of iterations

    Returns
    -------
        C: reduced version of incoming 3D array of pictures

    """
    print('Started pipeline ...')
    C_rotated = np.zeros(np.shape(array))
    A = np.copy(array)
    change = np.inf
    sum_arr = np.array([np.inf, 0])
    i = 0

    while change > delta:
        print('\tChange={0}, iteration = {1}   '.format(change, i+1), end='\r')

        sum_arr[1] = sum_arr[0]
        # Subtract previous final image
        C_rotated[C_rotated < 0] = 0
        A = array - C_rotated

        # Calculate median and subtract it from original array
        B = np.median(A, axis=0)
        sum_arr[0] = np.sum(B)

        change = np.diff(sum_arr)[0]

        Sub = array - B

        # Rotate images back and combine by taking the median
        D = rotate_cv(Sub, -1 * angles)
        C = np.median(D, axis=0)

        # Rotated final image so it can be used at the start again
        C_rotated = rotate_cv(C, angles)
        i += 1
        if i >= miter:
            print('\nMaximum number of iterations surpassed')
            break

    print('\nFinal change = {0} after {1} iterations'.format(change, i))
    print('Done!')
    return C


def main():
    """Test defnitions in this file."""
    pass


if __name__ == '__main__':
    main()
