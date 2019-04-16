#standard library imports (os, sys)
import os
#general
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
#local imports
from pyeem import cleanscan
from pyeem import spectrasmooth

def load_eem_meta_data(excel_file):
    """Read EEM meta data into a pandas dataframe from excel template provided in the pyeem examples folder:
    
    Args:
        excel_file (str): relative path and file name of meta data excel file
    
    Returns:
        meta_data (pandas DataFrame): meta data in a pandas data frame
    """
    import pandas as pd
    meta_data = pd.read_excel(excel_file, sheet_name='Sample', skiprows=1)
    meta_data = meta_data.drop(columns='Index')
    return meta_data

def init_h5_database(database_name, meta_data, overwrite = False):
    """Initialize a h5 file for storing EEMs using a pandas DataFrame containing EEM meta data 
    
    Args:
        database_name (str): filename and relative path for h5 database
        meta_data (pandas DataFrame): DataFrame containing eem meta data from `pyeem.load_eem_meta_data` 
        function or created manually - see pyeem.load_eem_meta_data for required columns
        
    Returns:
        no retun - data is saved as h5 and may be loaded using `pyeem.load_eem_data`
    """
    from pandas import HDFStore
        
    # check if h5 file exists and overwrite or warn
    if os.path.isfile(database_name):
        if overwrite is True:
            print('overwriting ' + database_name)
            os.remove(database_name)
        else:
            raise ValueError("h5 file " + database_name + " exists. Choose new database name or set overwrite=True")
        
    # create a h5 file to store EEM meta data
    hdf = HDFStore(database_name)
    hdf.put('meta', meta_data, format='table', data_columns=True)
    hdf.close()
    return

def load_eems(database_name, data_dir):
    """Add eem spectra to the h5 file created with `init_h5_database`
    Args:
        database_name (str): filename and relative path for hdf5 file for saving EEMs
        data_dir (str): relative path to where EEM data is stored
         
    Returns:
        no retun - EEMs are saved to h5 file for processing with pyeem functions
    
    """
    import h5py
    import numpy as np
    from pandas import read_hdf
    
    #load EEM file names from the metadata stored in the h5 database as np.array
    file_names = np.array(read_hdf(database_name, 'meta')['File_Name'])
    folders = np.array(read_hdf(database_name, 'meta')['Folder'])
                          
    #initialize lists to store data
    eem_list = []
    names_list = []
    excitation_list = []
    emission_list = []
    
    for i, (folder, file) in enumerate(zip(folders, file_names)):
        eem_file = str(data_dir) + str(folder) + '/' + str(file) + '.dat'
        # first row of EEM file is excitation wavelengths, skip when reading in file
        eem = np.genfromtxt(eem_file, delimiter = '\t', skip_header=1)
        # emisson wavelengths stored in first column, store then remove
        emission = eem[:,0]
        eem = eem[:,1:]
        # load the excitaion wavelenths 
        excitation = np.genfromtxt(eem_file, delimiter = '\t', skip_header=0)[0,1:]
        eem_list.append(eem)
        excitation_list.append(excitation)
        emission_list.append(emission)
    # convert data to np arrays for saving
    eem_list = np.array(eem_list)
    excitation_list = np.array(excitation_list)
    emission_list = np.array(emission_list)
    print('EEM data collection complete, final data shape (Sample x Ex x Em):',eem_list.shape)
    
    # save data into the h5 file
    with h5py.File(database_name, "a") as f:
        dset = f.create_dataset("raw_eems", eem_list.shape, compression="gzip")
        dset[:] = eem_list
        dset2 = f.create_dataset("Excitation", excitation_list.shape, compression="gzip")
        dset2[:] = excitation_list
        dset3 = f.create_dataset("Emission", emission_list.shape, compression="gzip")
        dset3[:] = emission_list

    return



def update_eem_database(database_name, data_dict):
    """Helper function for updating and adding EEM data to h5 file as each step of data processing is completed:
    
    Args:
        database_name (str): filename and relative path for h5 database
        data_dict (dic): dictionary containing np.arrays of data to be saved 
    Returns:
        none
    """
    with h5py.File(database_name, 'a') as f:
        for key in data_dict.keys():
            # check for existing dataset so data can be overwritten, if dataset doesn't exist pass
            try:
                del f[key]
                print('Updating dataset:', key)
            except KeyError:
                pass
            dset = f.create_dataset(key, data_dict[key].shape, compression="gzip")
            dset[:] = data_dict[key]
            print("Dataset saved:", key, "... Shape = ", data_dict[key].shape)
    return


def load_eems(database_name, data_dir):
    """Add eem spectra to the h5 file created with `init_h5_database`
    Args:
        database_name (str): filename and relative path for hdf5 file for saving EEMs
        data_dir (str): relative path to where EEM data is stored
         
    Returns:
        no retun - EEMs are saved to h5 file for processing with pyeem functions
    
    """
    from pandas import read_hdf
    
    #load EEM file names from the metadata stored in the h5 database as np.array
    file_names = np.array(read_hdf(database_name, 'meta')['File_Name'])
    folders = np.array(read_hdf(database_name, 'meta')['Folder'])
                          
    #initialize lists to store data
    eem_list = []
    names_list = []
    excitation_list = []
    emission_list = []
    
    for i, (folder, file) in enumerate(zip(folders, file_names)):
        eem_file = str(data_dir) + str(folder) + '/' + str(file) + '.dat'
        # first row of EEM file is excitation wavelengths, skip when reading in file
        eem = np.genfromtxt(eem_file, delimiter = '\t', skip_header=1)
        # emisson wavelengths stored in first column, store then remove
        emission = eem[:,0]
        eem = eem[:,1:]
        # load the excitaion wavelenths 
        excitation = np.genfromtxt(eem_file, delimiter = '\t', skip_header=0)[0,1:]
        eem_list.append(eem)
        excitation_list.append(excitation)
        emission_list.append(emission)
    # convert data to np arrays for saving
    eem_list = np.array(eem_list)
    excitation_list = np.array(excitation_list)
    emission_list = np.array(emission_list)
    print('EEM data collection complete, final data shape (Sample x Ex x Em):',eem_list.shape)
    
    # save data into the h5 file
    
    update_eem_database(database_name, {'raw_eems': eem_list,
                                        'eems': eem_list,
                                        'raw_ex' : excitation_list,
                                        'ex' : excitation_list,                                        
                                        'raw_em' : emission_list,
                                        'em' : emission_list})
    return            

# def blank_subtract(database name):
#     """Subtract solvent blanks specified in metadata column 'Blanks' from EEMs 
    
#     Args:
#         database_name (str): filename for hdf5 database
       
#     Returns:
#         no retun - blank subtractions results are stored in h5 database under key 'blanks_subtracted'
#     """
#     try:
#         with h5py.File(filename + ".hdf5", 'r') as f:

#             eem_raw = f['Raw Data'][:]
#             blanks = f['Blank'][:]
#             file_name = f['File_Name'][:]

#     except OSError:
#         print(filename + '.hdf5 not found - please run meta_data_collector.py first!')
#         return

#     blanks_subtracted = np.zeros(eem_raw.shape)
#     for i in range(eem_raw.shape[0]):
#         #Find the index of the associated solvent blank
#         #Note: np.where retuns an array of tuples, adding [0] retuns a numpy array
#         blank_index = np.where(blanks[i] == file_name)[0]
#         if len(blank_index) == 0:
#             errMsg = 'Solvent Blank Associated with sample index ' + str(i) + ' not found.' + \
#             ' Correct this error in the metadata spreadsheet and re-run meta_data_saver'
#             raise Exception(errMsg)
#         if len(blank_index) > 1:
#             errMsg = 'Multiple solvent blank files for sample index ' + str(i) + ' found at indicies ' \
#             + str(blank_index) + ' Correct this error in the metadata and re-run pyeem.init_h5_database'
#             raise Exception(errMsg)
        
#         #Take the integer value of blank_index
#         blank_index = blank_index[0] 
        
#         blanks_subtracted[i, :, :] = eem_raw[i, :, :] - eem_raw[blank_index, :, :]

#     # update the database
#     update_eem_database(filename, {'blanks_subtracted': blanks_subtracted,
#                                    'eems': blanks_subtracted})

#     return


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

