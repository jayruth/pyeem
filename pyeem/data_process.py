#standard library imports (os, sys)
import os
#general
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
#local imports
from pyeem import cleanscan
from pyeem import trucate_below_excitation
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
        function or created manually - see pyeem.load_eem_meta_data for required columns.  NOTE: do not use
        spaces or decimals in column names as this causes a warning when saving to H5 file format
        
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
    EEMs data files must be tab delimited .dat files (standard export format from the Horibe Aqualog)
    The first row is excitation wavelengths and the first column is emission wavelengths.
    
    Args:
        database_name (str): filename and relative path for hdf5 file for saving EEMs
        data_dir (str): relative path to where EEM data is stored
         
    Returns:
        no retun - EEMs are saved to h5 file for processing with pyeem functions
    
    """
    from pandas import read_hdf

    try:
        #load EEM file names from the metadata stored in the h5 database as np.array
        file_names = np.array(read_hdf(database_name, 'meta')['File_Name'])
        folders = np.array(read_hdf(database_name, 'meta')['Folder'])
        
    except OSError:
        raise OSError(database_name + ' not found - please run pyeem.init_h5_database first')
        return

    #test if function has already run (dataset 'raw_eems' should not exist)
    with h5py.File(database_name, 'r') as f:
        try:
            test = f['raw_eems'][:]
            raise Exception('`load_eems` function has already run on this dataset')
        except KeyError:
            pass
            
                          
    #initialize list to store data
    eem_list = []
    
    for i, (folder, file) in enumerate(zip(folders, file_names)):
        eem_file = str(data_dir) + str(folder) + '/' + str(file) + '.dat'
        # first row of EEM file is excitation wavelengths, first column is emission wavelengths
        eem = np.genfromtxt(eem_file, delimiter = '\t')
        eem_list.append(eem)
    # convert data to np arrays for saving
    eem_list = np.array(eem_list)
    
    print('EEM data collection complete, final data shape (Sample x Ex+1 x Em+1):',eem_list.shape)
    
    # save data into the h5 file
    
    update_eem_database(database_name, {'raw_eems': eem_list,
                                        'eems': eem_list})
    return


def blank_subtract(database_name):
    """Subtract solvent blanks specified in metadata column 'Blanks' from EEMs 
    
    Args:
        database_name (str): filename for hdf5 database
       
    Returns:
        no retun - blank subtractions results are stored in h5 database under key 'blanks_subtracted'
    """
    from pandas import read_hdf
    
    #load EEMs to be blank subtracted
    try:
        with h5py.File(database_name, 'r') as f:
            eems = f['eems'][:]
        
    except OSError:
        raise OSError(database_name + ' not found - please run pyeem.init_h5_database and load_eems first')
        return
    except KeyError:
        raise KeyError('eem data not found - please run pyeem.load_eems first')
        return
    
    #test if function has already run (dataset 'blanks_subtracted' should not exist)
    with h5py.File(database_name, 'r') as f:
        try:
            test = f['blanks_subtracted'][:]
            raise Exception('`blank_subtract` function has already run on this dataset')
        except KeyError:
            pass
    #load EEM file names from the metadata stored in the h5 database as np.arrays
    file_names = np.array(read_hdf(database_name, 'meta')['File_Name'])
    folders = np.array(read_hdf(database_name, 'meta')['Folder'])
    blanks = np.array(read_hdf(database_name, 'meta')['Blank'])
    
    #intialize location to store blank subtraction results
    blanks_subtracted = np.zeros(eems.shape)

    for i in range(eems.shape[0]):
        #Find the index of the associated solvent blank
        #Note: np.where retuns an array of tuples, adding [0] retuns a numpy array
        blank_index = np.where(blanks[i] == file_names)[0]
        if len(blank_index) == 0:
            errMsg = 'Solvent Blank Associated with sample index ' + str(i) + ' not found.' + \
            ' Correct this error in the metadata and re-run'
            raise Exception(errMsg)
        if len(blank_index) > 1:
            errMsg = 'Multiple solvent blank files for sample index ' + str(i) + ' found at indicies ' \
            + str(blank_index) + ' Correct this error in the metadata'
            raise Exception(errMsg)
        
        #Take the integer value of blank_index
        blank_index = blank_index[0] 
        
        # subtract flourescent intensities
        blanks_subtracted[i, 1:, 1:] = eems[i, 1:, 1:] - eems[blank_index, 1:, 1:]
        # add excitiaton then emisson wavelenths to the first row and column
        blanks_subtracted[i,0,1:] = eems[i,0,1:]
        blanks_subtracted[i,1:,0] = eems[i,1:,0]

    # update the database
    update_eem_database(database_name, {'blanks_subtracted': blanks_subtracted,
                                   'eems': blanks_subtracted})

    return


def apply_cleanscan(database_name, tol='Default', coeff='Default'):
    """Apply the scatter removal function 'cleanscan' to all EEMs in the the dataset.
     Args:
        database_name (str): filename for hdf5 database
        tol (np.array, optional):parameters for applying cleanscan (see `pyeem.cleanscan` documentation)
        coeff (np.array, optional):parameters for applying cleanscan (see `pyeem.cleanscan` documentation)
       
    Returns:
        no retun - scatter removal results are stored in h5 database under key 'scatter_removed' 
        and the intermediate results showing what values were removed and replaced by interpolation 
        are saved under key 'excised_values'
    """
    #load EEMs for scatter removal
    try:
        with h5py.File(database_name, 'r') as f:
            eems = f['eems'][:]
        
    except OSError:
        raise OSError(database_name + ' not found - please run `pyeem.init_h5_database` and `pyeem.load_eems` first')
        return
    except KeyError:
        raise KeyError('eem data not found - please run `pyeem.load_eems` first')
        return

    #test if function has already run (dataset 'scatter_removed' should not exist)
    with h5py.File(database_name, 'r') as f:
        try:
            test = f['scatter_removed'][:]
            raise Exception('`apply_cleansscan` function has already run on this dataset')
        except KeyError:
            pass

    
    # initalize storage for final and intermediate results
    scatter_removed = np.zeros(eems.shape)
    excised_values = np.zeros(eems.shape)
    print('Removing Scatter')
    for i in tqdm(range(eems.shape[0])):
        #separeate eems into excitaion and emisson wavelenghts and fluorescence values
        ex = eems[i,0,1:]
        em = eems[i,1:,0]
        fl = eems[i,1:,1:]
        
        #remove scatter using cleanscan
        scatter_removed[i, 1:, 1:], excised_values[i, 1:, 1:], _ = cleanscan(ex, em, fl, tol, coeff)
        
        #add excitation and emission values to new datasets
        scatter_removed[i,0,1:] = ex
        scatter_removed[i,1:,0] = em
        excised_values[i,0,1:] = ex
        excised_values[i,1:,0] = em

    # update the database
    update_eem_database(database_name, {'scatter_removed': scatter_removed,
                                   'excised_values': excised_values,
                                   'eems': scatter_removed})

    return

def apply_truncate(database_name):
    """Apply the function `trucate_below_excitation` to all EEMs in the the dataset.
     Args:
        database_name (str): filename for hdf5 database
               
    Returns:
        no retun - truncation results are stored in h5 database under key 'trunc' 
        and 'eems' are updated with these values
    """
    #load EEMs for truncation
    try:
        with h5py.File(database_name, 'r') as f:
            eems = f['eems'][:]
        
    except OSError:
        raise OSError(database_name + ' not found - please run `pyeem.init_h5_database` and `pyeem.load_eems` first')
        return
    except KeyError:
        raise KeyError('eem data not found - please run `pyeem.load_eems` first')
        return

    #test if function has already run (dataset 'trunc' should not exist)
    with h5py.File(database_name, 'r') as f:
        try:
            test = f['trunc'][:]
            raise Exception('`trucate_below_excitation` function has already run on this dataset')
        except KeyError:
            pass

    
    # initalize storage for final and intermediate results
    trunc = np.zeros(eems.shape)
    print('Truncating EEMs: replacing emission values below the excitation wavelength with zeros')
    for i in tqdm(range(eems.shape[0])):
        #separeate eems into excitaion and emisson wavelenghts and fluorescence values
        ex = eems[i,0,1:]
        em = eems[i,1:,0]
        fl = eems[i,1:,1:]
        
        #apply truncation function
        trunc[i, 1:, 1:] = trucate_below_excitation(ex, em, fl)
        
        #add excitation and emission values to new datasets
        trunc[i,0,1:] = ex
        trunc[i,1:,0] = em
        
    # update the database
    update_eem_database(database_name, {'trunc': trunc,
                                   'eems': trunc})

    return

def apply_spectrasmooth(database_name, sigma='default', truncate='default'):
    """Apply 2D gausian smoothing and zero negative values for all EEMs in the the dataset.
     Args:
        database_name (str): filename for hdf5 database
        sigma ( ,optional):
        truncate( ,optional):
       
    Returns:
        no retun - smoothing results are stored in h5 database under key 'eems_smooth' 
    """
    
    #load EEMs for smoothing
    try:
        with h5py.File(database_name, 'r') as f:
            eems = f['eems'][:]
        
    except OSError:
        raise OSError(database_name + ' not found - please run `pyeem.init_h5_database` and `pyeem.load_eems` first')
        return
    except KeyError:
        raise KeyError('eem data not found - please run `pyeem.load_eems` first')
        return

    #test if function has already run (dataset 'eems_smooth' should not exist)
    with h5py.File(database_name, 'r') as f:
        try:
            test = f['eems_smooth'][:]
            raise Exception('`apply_spectrasmooth` function has already run on this dataset')
        except KeyError:
            pass
    
    # set parameters for smoothing   
    if sigma == 'default':
        sigma = 2
    else:
        pass
    if truncate == 'default':
        truncate = 4
    else:
        pass

    # initalize storage for smoothing results
    smoothed = np.zeros(eems.shape)
    
    for i in tqdm(range(eems.shape[0])):
        #separeate eems into excitaion and emisson wavelenghts and fluorescence values
        ex = eems[i,0,1:]
        em = eems[i,1:,0]
        fl = eems[i,1:,1:]
        
        #apply gausian smoothing
        smoothed[i, 1:, 1:] = spectrasmooth(fl, sigma, truncate)
        
        #add excitation and emission values to new dataset
        smoothed[i,0,1:] = ex
        smoothed[i,1:,0] = em
      
        # zero negative values
        smoothed[i, 1:, 1:][smoothed[i, 1:, 1:] < 0] = 0

    print("Finished smoothing, negative values set to zero")
    
    update_eem_database(database_name, {'eems_smooth': smoothed,
                                       'eems': smoothed})   
    return


def crop_eems(database_name, crop_spec):
    """Crop all EEMs in the the dataset.
     Args:
        database_name (str): filename for hdf5 database
        crop_spec (dict): excitaiton and emission values to crop at for example:
                        {'ex': (500, 224), 'em': (245.917, 572.284)}
                        values must match exactly with excitaiton and emission values
                        and must be ordered as the values occur in the data
    Returns:
        no retun - cropped eems are stored in h5 database under key 'eems_cropped' 
    """
    
    #load EEMs for cropping
    try:
        with h5py.File(database_name, 'r') as f:
            eems = f['eems'][:]
            starting_shape = eems.shape
        
    except OSError:
        raise OSError(database_name + ' not found - please run `pyeem.init_h5_database` and `pyeem.load_eems` first')
        return
    except KeyError:
        raise KeyError('eem data not found - please run `pyeem.load_eems` first')
        return
   
    #separeate eems into excitaion and emisson wavelenghts and fluorescence values
    ex = eems[:,0,1:]
    em = eems[:,1:,0]
    fl = eems[:,1:,1:]
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
    fl_crop = fl[:,
                   crop_spec['em_ind'][0]:crop_spec['em_ind'][1] + 1,
                   crop_spec['ex_ind'][0]:crop_spec['ex_ind'][1] + 1]

    #intialize location to store cropping results
    new_shape = (fl_crop.shape[0], fl_crop.shape[1]+1, fl_crop.shape[2]+1)
    eems_cropped = np.zeros(new_shape)
    #store cropped flourescence values and cropped ex and em values together
    eems_cropped[:, 1:, 1:] = fl_crop
    #add excitation and emission values to new dataset
    eems_cropped[:,0,1:] = ex_crop
    eems_cropped[:,1:,0] = em_crop
                                   
    print("EEMs cropped according to crop_spec")
    print(crop_spec)
    print("Starting shape", starting_shape, '(Sample x Em+1 x Ex+1)')
    print("Cropped shape", eems_cropped.shape, '(Sample x Em+1 x Ex+1)')
    
    update_eem_database(database_name, {'eems_cropped': eems_cropped,
                                   'eems': eems_cropped})
    
    return


def raman_normalize(database_name):
    """Raman normaization - element-wise division of the eem spectra by area under the ramam peak.
    See reference Murphy et al. "Measurement of Dissolved Organic Matter Fluorescence in Aquatic 
    Environments: An Interlaboratory Comparison" 2010 Environmental Science and Technology.
     Args:
        database_name (str): filename for hdf5 database
        Note-  'Raman_Area' column is required in the metadata to use this function.
        
    Returns:
        no retun - raman normalized eems are stored in h5 database under key 'eems_ru' 
    """
    from pandas import read_hdf
    #load EEMs for normalization
    try:
        with h5py.File(database_name, 'r') as f:
            eems = f['eems'][:]

    except OSError:
        raise OSError(database_name + ' not found - please run `pyeem.init_h5_database` and `pyeem.load_eems` first')
        return
    except KeyError:
        raise KeyError('eem data not found - please run `pyeem.load_eems` first')
        return
    #load values for raman normalization
    try:
        #load raman area from the metadata stored in the h5 database as np.array
        raman_area = np.array(read_hdf(database_name, 'meta')['Raman_Area'])

    except KeyError:
        raise KeyError('Raman_Area not found.  This must be included in the meta data to use this function')
        return

    
    #test if function has already run (dataset 'eems_ru' should not exist)
    with h5py.File(database_name, 'r') as f:
        try:
            test = f['eems_ru'][:]
            raise Exception('`raman_normalize` function has already run on this dataset')
        except KeyError:
            pass
    
    #intialize storage for normaized eems
    eems_ru = np.zeros(eems.shape)
    
    for i in tqdm(range(eems.shape[0])):
        #separeate eems into excitaion and emisson wavelenghts and fluorescence values
        ex = eems[i,0,1:]
        em = eems[i,1:,0]
        fl = eems[i,1:,1:]
        
        #raman normailze
        eems_ru[i, 1:, 1:] = eems[i, 1:, 1:] / raman_area[i]
        
        #add excitation and emission values to new dataset
        eems_ru[i,0,1:] = ex
        eems_ru[i,1:,0] = em


    update_eem_database(database_name, {'eems_ru': eems_ru,
                                   'eems': eems_ru})

    return

def load_meta_data(database_name):
    """Load the pandas dataframe containing meta data from an h5 file created using pyeem data processing functions 
     Args:
        database_name (str): filename (and relative path) for hdf5 database
        
    Returns:
        meta_data - pandas data frame containing EEM meta data 
    """
    from pandas import read_hdf
    meta_data = read_hdf(database_name, 'meta')
    
    return meta_data