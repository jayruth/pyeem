import pandas as pd
import numpy as np


def conc_to_np_array(metadata):
    """This function converts concentrations entered as comma separated strings into a numpy array

    Parameters
    -------------
    metadata : pandas.DataFrame containing a column 'Conc' with n comma separated values in each row of the column
                where n corresponds to the number of species

    Return
    -------------
    concentration : numpy array size n x number of samples

    """

    # Find the size of array needed to store concentration data and create an an array of zeros
    conc_temp = metadata['Conc'][0]
    conc_temp = np.fromstring(conc_temp, sep=',')
    concentration = np.zeros([len(metadata), len(conc_temp)])

    rows = metadata.index

    for row in rows:
        conc_temp = metadata['Conc'][row]
        conc_temp = np.fromstring(conc_temp, sep=',')
        concentration[row, :] = conc_temp

    return concentration


def meta_data_collector(excel_file='Description.xlsx'):
    """This function gathers metadata from an excel sheet and outputs NP arrays containing the data

    Parameters
    -------------
    excel_file : Filename for the excel spreadsheet containing sample metadata. Default value is 'Description.xlsx'.

    Return
    -------------
    file_name_sample
    folder
    desc_sample
    conc
    raman_area
    blank
    sample_type

    """

    # Read sample data into a pandas dataframe
    # Drop the column 'Index' - this is used a reference when looking
    # at the sample list in excel
    metadata_sample = pd.read_excel(excel_file, sheet_name='Sample', skiprows=1)
    metadata_sample = metadata_sample.drop(columns='Index')
    
    # Create NP Arrays corresponding to all columns. 
    # 'conc_to_np_array' is used to create a np array from
    # concentation values that are entered as comma separated values 
    # in a single excel column

    file_name_sample = np.array(metadata_sample['File_Name'])
    folder = np.array(metadata_sample['Folder'])
    desc_sample = np.array(metadata_sample['Desc'])
    conc = conc_to_np_array(metadata_sample)
    raman_area = np.array(metadata_sample['Raman_Area'])
    blank = np.array(metadata_sample['Blank'])
    sample_type = np.array(metadata_sample['Type'])

    return file_name_sample, folder, desc_sample, conc, raman_area, blank, sample_type, metadata_sample.columns

def import_files(runs):
    """This function takes in a list of filenames and returns all 
    of the data imported into numpy arrays."""
    data_list = []
    names_list = []
    excitation_list = []
    emission_list = []
    for run in runs:
        filename = str(run) + '.dat'
        data = np.genfromtxt(filename, delimiter = '\t', skip_header=1)
        excitation = np.genfromtxt(filename, delimiter = '\t', skip_header=0)[0,1:]
        emission = data[:,0]
#        if data.shape == (250,152):
        data_list.append(data[:,1:])
        excitation_list.append(excitation)
        emission_list.append(emission)
#        else:
#            print(filename, " Of unexpected size")
    return (np.array(data_list), np.array(excitation_list), np.array(emission_list))

def meta_data_saver(filename, metadata_file, metadata_dir):
    """This function takes in a filename, which will become the name of the h5 file that it
    creates at the end of the data assembly.
    ==============  ================================================
    Input            Description
    ==============  ================================================
    *filename*       Name (including directory) of desired output h5 file.
    *metadata_file*  Name of the Excel file containing the metadata.
    *metadata_dir*   Name of directory holding the metadata and data folders.
    ==============  ================================================"""
    import h5py
    # run the meta data collector to gather data from excel file
    file_name_sample, folder, desc_sample, conc, raman_area, blank, sample_type, columns = \
    meta_data_collector(excel_file = metadata_dir+metadata_file)
    
    print('Metadata collection complete. Columns collected:')
    print(list(columns))
    
    # collect spectra from tab delimited .dat files 
    runs=[]
    for i in range(len(file_name_sample)):
        fname = file_name_sample[i]
        ffolder = folder[i]
        runs.append(metadata_dir+str(ffolder)+'/'+fname)
    # Data in [iterration, x, y]
    data, excitation, emission = import_files(runs)
    metadata_list = [file_name_sample, folder, desc_sample, conc, raman_area, blank, sample_type]
    print('Data collection complete, final data shape (Sample x Ex x Em):',data.shape)
    
    # save data into h5 array
    with h5py.File(filename+".hdf5", "w") as f:
        dset = f.create_dataset("Raw Data", data.shape, compression="gzip")
        dset[:] = data
        dset2 = f.create_dataset("Excitation", excitation.shape, compression="gzip")
        dset2[:] = excitation
        dset3 = f.create_dataset("Emission", emission.shape, compression="gzip")
        dset3[:] = emission
        for i in range(len(metadata_list)):
            typeof = type(metadata_list[i][0])
            if typeof == type(''):
                metadata_list[i] = metadata_list[i].astype(str)
                dset = f.create_dataset(columns[i], metadata_list[i].shape, dtype = 'S20', compression="gzip")
                metadata_list[i] = [a.encode('utf8') for a in np.array(metadata_list[i])]
                dset[:] = metadata_list[i]
            else:
                dset = f.create_dataset(columns[i], metadata_list[i].shape, dtype = 'f', compression="gzip")
                dset[:] = metadata_list[i][:]
    
    # test that it was written properly
    f = h5py.File(filename+".hdf5", "r")
    d2 = f['Raw Data']
    print('The following data was saved to', filename+'.hdf5')
    for i in f.keys():
        d2 = f[i]
        print(i, d2.shape)
    f.close()
    return
