import h5py
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from pandas import read_hdf
import numpy as np

def view_eems(database_name):
    file_names = np.array(read_hdf(database_name, 'meta')['File_Name'])
    def plot_eem(database_name, i):
        
        fig = plt.figure(figsize=(4,3))
        with h5py.File(database_name, 'r') as f:
            eem = f['eems'][i]
        ex = eem[0,1:]
        em = eem[1:,0]
        fl = eem[1:,1:]
        plt.contourf(ex, em, fl)
        plt.colorbar()
        plt.title(file_names[i])
        return 
    
    print(file_names.shape[0])
    interact(plot_eem, database_name=fixed(database_name), i=(0,file_names.shape[0]-1))
    return


#################################
####### Old visual functions from class work with Ben, need to adapt to new data structure
################################

import numpy as np
import pandas as pd
#import plotly.plotly as py
#import plotly.offline as offline
#import plotly.graph_objs as go
#from plotly.graph_objs import *
#offline.init_notebook_mode()

from scipy.ndimage import gaussian_filter

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

#Disabled all plotly features because not currently using

def contourslice(ex, em, z, n, lower_level_frac, higher_level_frac, title=''):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    '''Create a contour plot of your spectra data with a specific height and number of contour lines.
    ==========  ================================================
    Input    Description
    ==========  ================================================
    *ex*        Excitation wavelength as mesh grid of numpy.arrays
    *em*        Emission wavelength as mesh grid of numpy.arrays
    *em*        Intensity as mesh grid of numpy.arrays
    *n*         number of contour lines
    *lower*     percentile of lower contour
    *higher*    percentile of higher contour
    ==========  ================================================
    ex = excitation wavelengths as a np.meshgrid of an np.array
    em = emission wavelengths as a np.meshgrid of an np.array

    n is the number of contour lines that you want to view in the range selected.

    lower_level_frac and higher_level_frac are the percentile ranges of the
    intensity(z) levels to view. The range is from 0 to 1 over the range 
    and lower_level_frac must be less than higher_level_frac.
    ============================================================'''

    assert higher_level_frac > lower_level_frac, 'higher must be greater than lower and both between 1 and 0'
    assert type(ex) == type(np.array([1])), 'ex should be np array'
    assert type(em) == type(np.array([1])), 'em should be np array'
    assert type(z) == type(np.array([1])), 'ez should be np array'
#     assert len(np.shape(ex))==2, 'shape is not the right dimensions'
#     assert np.shape(ex)==np.shape(em), 'shape of ex and em are not the same'
#     assert np.shape(ex)==np.shape(z), 'shape of z and ex are not the same'

    x = ex
    y = em

    # This is how we select the contour levels based on the input to the function
    bottom = z.min()
    top = z.max()
    diff = top - bottom
    higher_level = higher_level_frac*diff+bottom
    lower_level = lower_level_frac*diff+bottom
    level = np.linspace(lower_level, higher_level, n)
    
    fig = plt.figure(figsize=(5, 4))
    plt.contourf(x, y, z, n, levels=level)  # plots contour plot
    plt.title(title + '\n Slice from %1.2f' % lower_level_frac +
              ' to %1.2f' % higher_level_frac + ' of max')  # name function with info
    plt.xlabel('ex '+r'$\lambda$')  # label axis
    plt.ylabel('em '+r'$\lambda$')
    plt.colorbar()
    return fig

def contourslice_manual(ex, em, z, n, low_level, high_level, title=''):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    '''Create a contour plot of your spectra data with a specific height and number of contour lines.
    ==========  ================================================
    Input    Description
    ==========  ================================================
    *ex*        Excitation wavelength as mesh grid of numpy.arrays
    *em*        Emission wavelength as mesh grid of numpy.arrays
    *em*        Intensity as mesh grid of numpy.arrays
    *n*         number of contour lines
    *low_level*     lower limit for contour levels
    *high_level*    high limit for contour levels
    ==========  ================================================
    ex = excitation wavelengths as a np.meshgrid of an np.array
    em = emission wavelengths as a np.meshgrid of an np.array

    n is the number of contour lines that you want to view in the range selected.

    lower_level_frac and higher_level_frac are the percentile ranges of the
    intensity(z) levels to view. The range is from 0 to 1 over the range 
    and lower_level_frac must be less than higher_level_frac.
    ============================================================'''

 

    x = ex
    y = em

    # This is how we select the contour levels based on the input to the function
    level = np.linspace(low_level, high_level, n)
    
    fig = plt.figure(figsize=(5, 4))
    plt.contourf(x, y, z, n, levels=level)  # plots contour plot
    plt.title(title)  # name function with info
    plt.xlabel('ex '+r'$\lambda$')  # label axis
    plt.ylabel('em '+r'$\lambda$')
    plt.colorbar()
    return fig


def contour_line_slice(ex, em, z, n, lower_level_frac, higher_level_frac):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    '''Create a contour plot of your spectra data with a specific height and number of contour lines.
    ==========  ================================================
    Input    Description
    ==========  ================================================
    *ex*        Excitation wavelength as mesh grid of numpy.arrays
    *em*        Emission wavelength as mesh grid of numpy.arrays
    *em*        Intensity as mesh grid of numpy.arrays
    *n*         number of contour lines
    *lower*     percentile of lower contour
    *higher*    percentile of higher contour
    ==========  ================================================
    ex = excitation wavelengths as a np.meshgrid of an np.array
    em = emission wavelengths as a np.meshgrid of an np.array

    n is the number of contour lines that you want to view in the range selected.

    lower_level_frac and higher_level_frac are the percentile ranges of the
    intensity(z) levels to view. The range is from 0 to 1 over the range 
    and lower_level_frac must be less than higher_level_frac.
    ============================================================'''

    assert higher_level_frac > lower_level_frac, 'higher must be greater than lower and both between 1 and 0'
    assert type(ex) == type(np.array([1])), 'ex should be np array'
    assert type(em) == type(np.array([1])), 'em should be np array'
    assert type(z) == type(np.array([1])), 'ez should be np array'
#     assert len(np.shape(ex))==2, 'shape is not the right dimensions'
#     assert np.shape(ex)==np.shape(em), 'shape of ex and em are not the same'
#     assert np.shape(ex)==np.shape(z), 'shape of z and ex are not the same'

    x = ex
    y = em

    # This is how we select the contour levels based on the input to the function
    bottom = z.min()
    top = z.max()
    diff = top - bottom
    higher_level = higher_level_frac*diff+bottom
    lower_level = lower_level_frac*diff+bottom
    level = np.linspace(lower_level, higher_level, n)

    plt.contour(x, y, z, n, levels=level)  # plots contour plot
    plt.title('Contour Plot from %1.2f' % lower_level_frac +
              ' to %1.2f' % higher_level_frac + ' of normalized Intensity')  # name function with info
    plt.xlabel('ex '+r'$\lambda$')  # label axis
    plt.ylabel('em '+r'$\lambda$')
    plt.colorbar()
    return


def contour_line_compare(ex, em, eem_1, eem_2, n):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
   

    '''Create a contour plot of your spectra data with a specific height and number of contour lines.
    ==========  ================================================
    Input    Description
    ==========  ================================================
    *ex*        Excitation wavelength vector
    *em*        Emission wavelength vector
    *eem_1*     First EEM to plot (scale is based on maximum of this eem)
    *eem_2*     Second EEM to plot
    *n*         number of contour lines
    ==========  ================================================
    '''

    
    # Choose countour levels based on eem_1
    level = np.linspace(eem_1.min(), eem_1.max(), n)
    print(level)
    
    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    ax0 = plt.subplot(gs[0])
    
    plt0 = ax0.contour(ex, em, eem_1, n, levels=level)  # plots contour plot
    plt.title('')
    plt.xlabel('ex '+r'$\lambda$')  # label axis
    plt.ylabel('em '+r'$\lambda$')
    fig.colorbar(plt0, ax=ax0)
        
    ax1 = plt.subplot(gs[1])
    plt1 = ax1.contour(ex, em, eem_2, n, levels=level)  # plots contour plot
    plt.title('')
    plt.xlabel('ex '+r'$\lambda$')  # label axis
    plt.ylabel('em '+r'$\lambda$')
    fig.colorbar(plt1, ax=ax1)
    
    return


# def fingerprint_3D_plot(ex, em, z):
#     '''Create a 3D contour plot with contour and finger prints.

#     ==========  ================================================
#     Input    Description
#     ==========  ================================================
#     *ex*        Excitation wavelength as meshgrid of numpy.arrays
#     *em*        Emission wavelength as meshgrid of numpy.arrays
#     *z*         Intensity as meshgrid of numpy.arrays
#     ==========  ================================================'''
#     import plotly.plotly as py
#     import plotly.offline as offline
#     import plotly.graph_objs as go
#     from scipy.ndimage import gaussian_filter
#     offline.init_notebook_mode()

#     from mpl_toolkits.mplot3d import axes3d
#     import matplotlib.pyplot as plt
#     from matplotlib import cm
    
#     # set the figure size and the 3D axis
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.gca(projection='3d')

#     assert type(ex) == type(np.array([1])), 'ex should be np array'
#     assert type(em) == type(np.array([1])), 'em should be np array'
#     assert type(z) == type(np.array([1])), 'ez should be np array'
# #     assert len(np.shape(ex))==2, 'shape is not the right dimensions'
# #     assert np.shape(ex)==np.shape(em), 'shape of ex and em are not the same'
# #     assert np.shape(ex)==np.shape(z), 'shape of z and ex are not the same'

#     # name the variables for the function
#     X = em
#     Y = ex
#     Z = z

#     fig.suptitle('3D Surface Plot with 2D spectra and Contour', fontsize=20)

#     # Surface plot and the csets for the 2D spectra and contour on the wall
#     surf = ax.plot_surface(X, Y, Z, alpha=1, cmap=cm.coolwarm)
#     cset = ax.contour(X, Y, Z, zdir='z', offset=Z.max(), # these add wall plots
#                       cmap=cm.coolwarm)  # extend3d=True)
#     cset = ax.contour(X, Y, Z, zdir='x', offset=250, cmap=cm.cool)
#     #cset = ax.contour(X, Y, Z, zdir='y', offset=500, cmap=cm.spring)

#     fig.colorbar(surf, shrink=0.7, aspect=15,)

#     ax.set_ylabel('Excitation '+r'$\lambda$')
#     ax.set_ylim(Y.max(), Y.min())
#     ax.set_xlabel('Emission '+r'$\lambda$')
#     ax.set_xlim(X.min(), X.max())
#     ax.set_zlabel('Intensity')
#     ax.set_zlim(Z.min(), Z.max())
#     return


# def Interactive_3D_Plot(ex, em, z):
    
#     '''Create an interactive 3D contour plot.

#     ==========  ================================================
#     Input    Description
#     ==========  ================================================
#     *ex*        Excitation wavelength as meshgrid of numpy.arrays
#     *em*        Emission wavelength as meshgrid of numpy.arrays
#     *z*         Intensity as meshgrid of numpy.arrays
#     ==========  ================================================'''
#     import plotly.plotly as py
#     import plotly.offline as offline
#     import plotly.graph_objs as go
    
#     offline.init_notebook_mode()
#     from scipy.ndimage import gaussian_filter
   
#     assert type(ex) == type(np.array([1])), 'ex should be np array'
#     assert type(em) == type(np.array([1])), 'em should be np array'
#     assert type(z) == type(np.array([1])), 'ez should be np array'
# #     assert len(np.shape(ex))==2, 'shape is not the right dimensions'
# #     assert np.shape(ex)==np.shape(em), 'shape of ex and em are not the same'
# #     assert np.shape(ex)==np.shape(z), 'shape of z and ex are not the same'

#     surface = Surface(x=em, y=ex, z=z)
#     data = Data([surface])

#     layout = Layout(
#         title='3D Intensity Plot',
#         scene=Scene(
#             aspectmode="manual",
#             aspectratio=dict(x=1, y=1, z=1),


#             xaxis=dict(
#                 title='emission',
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             yaxis=YAxis(
#                 title='excitation',
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             ),
#             zaxis=ZAxis(
#                 title='Intensity',
#                 gridcolor='rgb(255, 255, 255)',
#                 zerolinecolor='rgb(255, 255, 255)',
#                 showbackground=True,
#                 backgroundcolor='rgb(230, 230,230)'
#             )
#         )
#     )

#     fig = Figure(data=data, layout=layout)
#     offline.iplot(fig)  # , image='png')
#     return
