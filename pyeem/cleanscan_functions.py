import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate


def cleanscan(ex, em, fl, tol='Default', coeff='Default'):
    """Function for removing Raleigh and Raman scatter by excising values in the areas where scatter is expected
    and replacing the missung values using 2d interpolation.  This function is based on the following publication: 
    Zepp et al. Dissolved organic fluorophores in southeastern US coastal waters: correction method for eliminating 
    Rayleigh and Raman scattering peaks in excitation–emission matrices. Marine Chemistry. 2004   
    
    Args:
        ex (np.array): vector of excitation wavelengths
        em (np.array): vector of emission wavelengths
        fl (np.array): matrix of flourescence intensities having shape ex x em
    
    Returns:
        meta_data (pandas DataFrame): meta data in a pandas data frame
    """

    # define default fitting coefficients (values from Zepp et. al. 2004)
    if coeff is 'Default':
        coeff = np.array(([0, 1.0000, 0],
                        [0.0006, 0.8711, 18.7770],
                        [0, 2.0000, 0],
                        [-0.0001, 2.4085, -47.2965]))

    if tol is 'Default':
        tol=np.array([[10,  10],
                     [10,   10],
                     [10,   10],
                     [10,   10]]) 
    
    # create a meshgrid of excitation and emission wavelenghts
    grid_ex,grid_em = np.meshgrid(ex, em)

    # initialize a filter matrix (logical array with all values 'False")
    values_to_excise = np.zeros(fl.shape, dtype=bool)

    for n in range(len(tol)):
        # only remove scatter if the tolerance is greater than 0
        if tol[n,0] > 0 or tol[n,1] > 0: 

            peaks = np.polyval(coeff[n,:], ex)
            # peaks is a 1 x length(ex) vector containing the emission wavelength
            # of expected scatter peaks at each exctation wavelenth
            peaks_grid = np.tile(peaks.reshape(1,-1), (em.size,1))

            # create logical arrays with 'True' where flourescent values should be kept
            keep_above = (grid_em - np.subtract(peaks_grid, tol[n,0])) <= 0
            keep_below = (grid_em - np.add(peaks_grid, tol[n,1])) >= 0

            #update locations of flourecent values to excise
            values_to_excise = values_to_excise + np.invert(keep_above + keep_below)

    # create a boolean array of values to keep to use when interpolating
    values_to_keep = np.invert(values_to_excise)
    
    # create an array with 'nan' in the place of values where scatter is located
    # this may be used for vizualizing the locations of scatter removal
    fl_NaN = np.array(fl)
    fl_NaN[values_to_excise] = np.nan
    
    # interpolate to fill the missing values
    # 'points' is a 'Number of Points' x 2 array containing coordinates 
    # of datapoints to be used when interpolating to fill in datapoints
    points = np.array([np.reshape(grid_ex[values_to_keep], (-1)),
                      np.reshape(grid_em[values_to_keep], (-1))])
    points = np.transpose(points)
    values = fl[values_to_keep]
    
    fl_interp = scipy.interpolate.griddata(points, values, (grid_ex, grid_em), fill_value=0 )
    
    # replace excised values with interpolated values
    fl_clean = np.array(fl)
    fl_clean[values_to_excise] = fl_interp[values_to_excise]
    
    return fl_clean, fl_NaN, fl_interp 


def cleanscan_viz(ex, em, fl, fl_NaN, fl_interp, fl_clean):
    fig = plt.figure(figsize=(20,5))
    fig.suptitle('Scatter Removal', fontsize=20) # suptitle is the main title for the figure

    plt.subplot(1, 4, 1)
    plt.contourf(ex, em, fl)
    plt.title('Before Scatter Removal')
    plt.xlabel('Excitation')
    plt.ylabel('Emission')
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.contourf(ex, em, fl_NaN)
    plt.title('Excised Values')
    plt.xlabel('Excitation')
    plt.ylabel('Emission')
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.contourf(ex, em, fl_interp)
    plt.title('Interpolation Results')
    plt.xlabel('Excitation')
    plt.ylabel('Emission')
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.contourf(ex, em, fl_clean)
    plt.title('Cleaned Spectra')
    plt.xlabel('Excitation')
    plt.ylabel('Emission')
    plt.colorbar()
    return fig


def trucate_below_excitation(ex, em, fl):
    """Replace values below the excitation wavelength with zero   
    
    Args:
        ex (np.array): vector of excitation wavelengths
        em (np.array): vector of emission wavelengths
        fl (np.array): matrix of flourescence intensities with shape ex x em
    
    Returns:
        eem_trunc (np.array): matrix of flourescence intensities with values below
                                the excitation wavelength replaced with zeros
    """
   
    # create a meshgrid of excitation and emission wavelenghts
    grid_ex,grid_em = np.meshgrid(ex, em)

    # create a logical array with 'True' where values should be set to zero
    values_to_zero = (grid_em - grid_ex) < 0

    # create an array with 'nan' in the place of values where scatter is located
    # this may be used for vizualizing the locations of scatter removal
    eem_trunc = fl
    eem_trunc[values_to_zero] = 0

    return eem_trunc 


