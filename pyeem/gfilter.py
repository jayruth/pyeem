import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def spectrasmooth(z, sig, trun):
    '''This function does a gaussian_blurr on the 2D spectra image from
    the input sigma and truncation sigma.

    ==========  ================================================
    Input       Description
    ==========  ================================================
    *ex*        Excitation wavelength as meshgrid of numpy.arrays
    *em*        Emission wavelength as meshgrid of numpy.arrays
    *z*         Intensity as meshgrid of numpy.arrays
    *sig*       Sigma of the gaussian distribution weight for the data smoothing
    *trun*      truncate in 'sigmas' the gaussian distribution
    ==========  ================================================'''

    # smooth the data with the gaussian filter
    z_blurred = gaussian_filter(z, sigma=sig, truncate=trun)
    return z_blurred


def plot_data(ex, em, z, sig, trun):
    '''This function does a gaussian_blurr on the 2D spectra image from
    the input sigma and truncation sigma and then plots the data.

    ==========  ================================================
    Input       Description
    ==========  ================================================
    *ex*        Excitation wavelength as meshgrid of numpy.arrays
    *em*        Emission wavelength as meshgrid of numpy.arrays
    *z*         Intensity as meshgrid of numpy.arrays
    *sig*       Sigma of the gaussian distribution weight for the data smoothing
    *trun*      truncate in 'sigmas' the gaussian distribution
    ==========  ================================================'''
        # first smooth the data with the gaussian filter
    z_blurred = gaussian_filter(z, sigma=sig, truncate=trun)

    # plot the data with the appropriate variables in title
    fig = plt.figure(figsize=(8,8))
    plt.title('Gaussian Blurr Contour Plot, ' + r'$\sigma$'+' = %1.0f' %
              sig + ', Truncation'+' = %1.0f' % trun + r'$\sigma$') #  makes auto title
    plt.contourf(ex, em, z_blurred)
    plt.xlabel('ex '+r'$\lambda$')  # Label the axis

    plt.ylabel('em '+r'$\lambda$')
    return fig

