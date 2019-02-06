import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyeem import cleanscan
from pyeem import spectrasmooth


def update_eem_database(filename, data_dict):
    with h5py.File(filename+".hdf5", 'a') as f:
        for key in data_dict.keys():
            # check for existing dataset so data can be overwritten, if dataset doesn't exist pass
            try:
                del f[key]
                print('Overwriting existing dataset:', key)
            except KeyError:
                pass
            dset = f.create_dataset(key, data_dict[key].shape, compression="gzip")
            dset[:] = data_dict[key]
            print("Dataset saved:", key, "... Shape = ", data_dict[key].shape)


def blank_subtract(filename):
    try:
        with h5py.File(filename + ".hdf5", 'r') as f:

            eem_raw = f['Raw Data'][:]
            blanks = f['Blank'][:]
            file_name = f['File_Name'][:]

    except OSError:
        print(filename + '.hdf5 not found - please run meta_data_collector.py first!')
        return

    blanks_subtracted = np.zeros(eem_raw.shape)
    for i in range(eem_raw.shape[0]):
        #Find the index of the associated solvent blank
        #Note: np.where retuns an array of tuples, adding [0] retuns a numpy array
        blank_index = np.where(blanks[i] == file_name)[0]
        if len(blank_index) == 0:
            errMsg = 'Solvent Blank Associated with sample index ' + str(i) + ' not found.' + \
            ' Correct this error in the metadata spreadsheet and re-run meta_data_saver'
            raise Exception(errMsg)
        if len(blank_index) > 1:
            errMsg = 'Multiple solvent blank files for sample index ' + str(i) + ' found at indicies ' \
            + str(blank_index) + ' Correct this error in the metadata spreadsheet and re-run meta_data_saver'
            raise Exception(errMsg)
        
        #Take the integer value of blank_index
        blank_index = blank_index[0] 
        
        blanks_subtracted[i, :, :] = eem_raw[i, :, :] - eem_raw[blank_index, :, :]

    # update the database
    update_eem_database(filename, {'blanks_subtracted': blanks_subtracted,
                                   'eems': blanks_subtracted})

    return


def apply_cleanscan(filename, tol='Default', coeff='Default'):
    """This applies the scatter removal function 'cleanscan' function to the dataset stored in the specified file"""
    try:
        with h5py.File(filename + ".hdf5", 'r') as f:
            blanks_subtracted = f['blanks_subtracted'][:]
            ex = f['Excitation'][:]
            em = f['Emission'][:]

    except OSError:
        print(filename + '.hdf5 not found - please run meta_data_collector.py first!')
        return
        # loop through call to clean function:
    scatter_removed = np.zeros(blanks_subtracted.shape)
    excised_values = np.zeros(blanks_subtracted.shape)
    print('Removing Scatter')
    for i in tqdm(range(blanks_subtracted.shape[0])):
        scatter_removed[i, :, :], excised_values[i, :, :], _ = cleanscan(ex[i], em[i], blanks_subtracted[i],
                                                                                 tol, coeff)

    print('scatter_removed', scatter_removed.shape)
    print('excised_values', excised_values.shape)

    # update the database
    update_eem_database(filename, {'scatter_removed': scatter_removed,
                                   'excised_values': excised_values,
                                   'eems': scatter_removed})

    return


def apply_spectrasmooth(filename, sigma='default', truncate='default'):
    """This function takes in a filename, which was previously altered by the DeScatter function,
    clean_data.  It relies upon the presence of 'scatter_removed', a column created in that section.
    ==============  ================================================
    Input            Description
    ==============  ================================================
    *filename*       Name (including directory) of desired output h5 file.
    ==============  ================================================"""
    try:
        with h5py.File(filename+".hdf5", 'r') as f:
            eem = f['scatter_removed'][:]
    except OSError:
        print(filename+'.hdf5 not found - please run meta_data_collector.py first!')
        return
    if sigma == 'default':
        sigma = 2
    else:
        pass
    if truncate == 'default':
        truncate = 4
    else:
        pass

    smoothed = []
    for i in tqdm(range(eem.shape[0])):
        smoothed.append(spectrasmooth(eem[i], sigma, truncate))
    smoothed = np.array(smoothed)

    # zero negative values
    smoothed[smoothed < 0] = 0

    update_eem_database(filename, {'eems_smooth': smoothed,
                                   'eems': smoothed})

    # with h5py.File(filename+".hdf5", 'a') as f:
    #     try:
    #         del f['eems_smooth']
    #     except KeyError:
    #         pass
    #     dset = f.create_dataset("eem_smooth", smoothed.shape, compression="gzip")
    #     dset[:] = smoothed

    print("Finished smoothing, negative values set to zero")
    return


def crop_eems(filename, crop_spec='default'):
    """Crops eems according to crop_spec (dictionary) and saves them to the hdf5 file"""
    try:
        with h5py.File(filename + ".hdf5", 'r') as f:

            eems_smooth = f['eems_smooth'][:]
            ex = f['Excitation'][:]
            em = f['Emission'][:]
    except OSError:
        print(filename + '.hdf5 not found - please run meta_data_collector.py first!')
        return

    if crop_spec == 'default':
        crop_spec = {'ex': (500, 224),
                     'em': (245.917, 572.284)}
    else:
        pass

    # find the indicies for each sepecified wavelength and convert to integers for slicing
    # this is done assuming all ex and em spectra are the same
    crop_spec['ex_ind'] = (np.where(ex[0] == crop_spec['ex'][0]), np.where(ex[0] == crop_spec['ex'][1]))
    if len(crop_spec['ex_ind'][0][0]) == 0 or len(crop_spec['ex_ind'][1][0]) == 0:
        errMsg = "Excitation crop wavelength not found.  Verify 'crop_spec['ex']' contains wavelengths found in Excitation"
        raise Exception(errMsg)
    crop_spec['ex_ind'] = (int(crop_spec['ex_ind'][0][0]), int(crop_spec['ex_ind'][1][0]))
    
    crop_spec['em_ind'] = (np.where(em[0] == crop_spec['em'][0]), np.where(em[0] == crop_spec['em'][1]))
    if len(crop_spec['em_ind'][0][0]) == 0 or len(crop_spec['em_ind'][1][0]) == 0:
        errMsg = "Emission crop wavelength not found.  Verify 'crop_spec['em']' contains wavelengths found in Emission"
        raise Exception(errMsg)
    
    crop_spec['em_ind'] = (int(crop_spec['em_ind'][0][0]), int(crop_spec['em_ind'][1][0]))

    # create cropped ex and em vectors
    ex_crop = ex[:, crop_spec['ex_ind'][0]:crop_spec['ex_ind'][1] + 1]
    em_crop = em[:, crop_spec['em_ind'][0]:crop_spec['em_ind'][1] + 1]
    # crop eem data
    eems_cropped = eems_smooth[:,
                   crop_spec['em_ind'][0]:crop_spec['em_ind'][1] + 1,
                   crop_spec['ex_ind'][0]:crop_spec['ex_ind'][1] + 1]

    update_eem_database(filename, {'eems_cropped': eems_cropped,
                                   'eems': eems_cropped,
                                   'ex_crop': ex_crop,
                                   'em_crop': em_crop})

    print("EEMs cropped according to crop_spec")
    print(crop_spec)
    print("Starting shape", eems_smooth.shape, '(sample x em x ex)')
    print("Cropped shape", eems_cropped.shape, '(sample x em x ex)')

    return


def raman_normalize(filename):
    """Raman normaization - element-wise division of the eem spectra by area under the ramam peak.
    See reference Murphy *** """
    try:
        with h5py.File(filename + ".hdf5", 'r') as f:

            eems_cropped = f['eems_cropped'][:]
            raman_area = f['Raman_Area'][:]

    except OSError:
        print(filename + '.hdf5 not found - please run meta_data_collector.py first!')
        return

    # Raman Unit Normalization
    eems_ru = np.zeros(eems_cropped.shape)
    for i in range(eems_cropped.shape[0]):
        eems_ru[i, :, :] = eems_cropped[i, :, :] / raman_area[i]

    update_eem_database(filename, {'eems_ru': eems_ru,
                                   'eems': eems_ru})

    # # save the raman normalized data into the hdf5
    # with h5py.File(filename + ".hdf5", 'a') as f:
    #     # check for existing dataset so data can be overwritten
    #     try:
    #         del f['eem_ru']
    #         print('Overwritting existing data raman normalized data')
    #
    #     except KeyError:
    #         pass
    #
    #     dset = f.create_dataset("eem_ru", eem_ru.shape, compression="gzip")
    #     dset[:] = eem_ru
    #     print('Ramam normalization complete')
    #     print(eem_ru.shape)

    return

