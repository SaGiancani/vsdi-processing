import datetime
import numpy as np
from scipy.interpolate import interp1d
import retinotopy as retino


def linear_extrapolation(signal, stop, start = 0):
    """
    ----------------------------------------------------------------------------------------------------------------------
    Author: Salvatore Giancani
    Email: sa.giancani@gmail.com
    ----------------------------------------------------------------------------------------------------------------------
    
    Perform linear extrapolation on a 3D data cube along the time axis.

    Args:
    signal (ndarray): The 3D data cube to extrapolate. Dimensions are (time, space_y, space_x).
    stop (int): The endpoint for extrapolation.
    start (int): The start point for extrapolation (default is 0).

    Returns:
    ndarray: The extrapolated 3D data cube with the same shape as the input.
    """
    assert len(signal) != 3, 'Datacube required'
    time, space_y, space_x = signal.shape
    fitted_cube = np.empty((time, space_y, space_x))
    for i in range(space_y):
        for j in range(space_x):
            tmp = retino.get_trajectory(np.arange(start, stop, 1), signal[start:stop, i, j], (0, time))
            fitted_cube[:, i, j] = tmp[1]
    return fitted_cube

def correction_windowframe(signal3d_, start, end):
    """
    ----------------------------------------------------------------------------------------------------------------------
    Author: Salvatore Giancani
    Email: sa.giancani@gmail.com
    ----------------------------------------------------------------------------------------------------------------------

    Perform correction on a 3D or 4D signal along the time axis by extrapolating and subtracting.

    Args:
    signal3d_ (ndarray): The input signal, which can be 3D (time, space_y, space_x) or 4D (frames, time, space_y, space_x).
    start (int): The start point for extrapolation.
    end (int): The endpoint for extrapolation.

    Returns:
    ndarray: The corrected signal with extrapolated data subtracted along the time axis.
    """
    if len(signal3d_.shape) == 3:
        signal3d = linear_extrapolation(signal3d_, end, start)
        outcome  = (signal3d_ - signal3d)+np.nanmean(signal3d) # Safety shift up of the subtraction

    elif len(signal3d_.shape) == 4:
        signal3d = list()
        for j, i in enumerate(signal3d_):
            signal3d.append(linear_extrapolation(i, end, start))
            print(f'Trial {j+1} linearly detrended')
        signal3d = np.array(signal3d)            
        outcome = (signal3d_ - signal3d)+np.nanmean(signal3d) # Safety shift up of the subtraction
    else:
        print('Error: shape incompatible. Either 3d or 4d')
        outcome = None
    return outcome 


def linear_fitting_filtering(raw_data_per_trial, mask):
    '''
    From Andrea Alamia's code. Linear fitting pixelwise of the signal.
    Input:
        raw_data_per_trial: np.array float, shape (n_Trials, time_bins, y, x). It corresponds to dF/F0 not rawdata.
        mask: np.array boolean, shape (y, x).
    Output:
        fittedY: np.array float, shape (n_Trials, time_bins, y, x) 
    '''
    # Timer
    start_time = datetime.datetime.now().replace(microsecond=0)

    # X for fitting
    x = np.arange(1, raw_data_per_trial.shape[1] + 1)
    # Instance output
    fittedY = np.zeros(raw_data_per_trial.shape)
    fittedY[np.where(fittedY==0)] = np.nan

    # Find the indices of significant pixels
    significant_pixels = np.argwhere(mask != 0)

    # Loop over trials
    for tt, trial in enumerate(raw_data_per_trial):
        # Loop over pixels
        for idx in significant_pixels:
            xx, yy = idx
            # Fitting
            f = np.polyfit(x, trial[:, xx, yy], 1)
            fittedY[tt, :, xx, yy] = np.polyval(f, x)
        print(f"Trial {tt + 1} completed.")

    print('Fitting by pixel performed in ' + str(datetime.datetime.now().replace(microsecond=0) - start_time) + '!\n')
    return fittedY

def get_template(detrended_data, 
                 data_heart_beat, 
                 data_sampling_rate = 100, 
                 heart_beat_sampling_rate = 1000, 
                 time_bins = [3, 30, 40, 68],
                 size_spatial_kernel = 2,
                 length_template = 50):
    '''
    From Andrea Alamia's code. Template extractor method.
    The method takes as input the detrended data and the heart beat tracks. It returns the template of the signal over the condition,
    the template single trial-basis, the alignment of the heart tracks, for the first and second peak in the heart beat signal.
    Input:
        detrended_data: np.array float, shape (trials, time, y, x)
        data_heart_beat: np.array float, shape (trials, time, y, x). The shape has to be coherent with the detrended data shape
            -taking in account the two corresponding sampling frequencies-.
        data_sampling_rate: int, default 100 (Hz).
        heart_beat_sampling_rate: int, default 1000 (Hz).
        time_bins: list, default [3, 30, 40, 68], respectively start and peak of the first period, and start and peak of the second period of heart beat.
        size_spatial_kernel: int, default 2. Subsampling for template building in data signal -not heart beat-.
        lenght_template: int, default 50. Length of the template, since the full length is not really useful.
    Output:
        the_template: np.array float, shape (length_template, y, x). Average template -over trials-
        template_per_trial: np.array float, shape (trials, length_template, x, y). 
            Template for each trial: interpolation 1d for each trial of the signal track.
        alignementP1: np.array int, shape (trials). Index of the first peak for each trial.
        alignementP2: np.array int, shape (trials). Index of the second peak for each trial.
        data_heart_beat_sr100: np.array float, shape (data_heart_beat).
            Template for each trial: interpolation 1d for each trial of the heart beat track.
    '''
    
    # Global timer started
    start_time = datetime.datetime.now().replace(microsecond=0)
    
    # Downsampling the heart to 100
    data_dims = detrended_data.shape
    data_heart_beat_dims = data_heart_beat.shape
    
    # Time in sec of the data tracks
    time_s_data = data_dims[1]/data_sampling_rate
    
    # Time in sec of the heart tracks
    time_s_heart = data_heart_beat_dims[1]/heart_beat_sampling_rate
    
    # Assess if the heart beat and signal have the same time dimension
    assert time_s_heart == time_s_data
    # Assess if the data cubes are 4 dimensional
    assert len((data_heart_beat.shape)) == 4
    assert len((detrended_data.shape)) == 4
    
    #Interpolation for heart beat tracks
    start_time_hb = datetime.datetime.now().replace(microsecond=0)
    print('Interpolation of heart beat signal starts...')
    
    # Instance of variables useful for interpolation
    x = np.arange(1, data_heart_beat_dims[1]+1)
    Xq = np.linspace(1, data_heart_beat_dims[1], dtype = int) 
    
    # Interpolation 1d of the heart tracks
    data_heart_beat_sr100 = interp1d(x, data_heart_beat, axis = 1)(Xq)
    print(f'Interpolation of heart beat signal ends in {(datetime.datetime.now().replace(microsecond=0) - start_time_hb)}\n')
    
    t1, t2, t3, t4 = time_bins 
    alignementP1 = np.argmax(data_heart_beat_sr100[:, t1-1:t2], axis=1) + t1
    alignementP2 = np.argmax(data_heart_beat_sr100[:, t3-1:t4], axis=1) + t3
        
    # Interpolation 1d of signal: pixelwise
    # Timer starting
    start_time_data = datetime.datetime.now().replace(microsecond=0)
    print('Interpolation of vsdi signal data starts...')      
    # Variable instance  
    template_per_trial = np.zeros((detrended_data.shape[0], length_template, detrended_data.shape[2], detrended_data.shape[3]))
    
    # Loop over trials
    for tt, data_ in enumerate(detrended_data):
        temp_map = data_[alignementP1[tt]-1:, :, :]
        # Loop over pixels
        for xx in range(size_spatial_kernel, detrended_data.shape[2]-size_spatial_kernel):
            for yy in range(size_spatial_kernel, detrended_data.shape[3]-size_spatial_kernel):
                # If size_spatial_kernel >1, it performs an average over pixels, neighbors of the picked one.    
                temp_signal = np.nanmean(temp_map[:, xx-size_spatial_kernel:xx+size_spatial_kernel, yy-size_spatial_kernel:yy+size_spatial_kernel], axis=(-1,-2))
                # Interpolation of the signal pixelwise
                x = np.arange(1, len(temp_signal)+1)
                Xq = np.linspace(1, len(temp_signal), length_template)
                template_per_trial[tt, :, xx, yy] = interp1d(x, temp_signal)(Xq)
        print(f"Trial {tt + 1} completed.")

    # Mean over trial-templates 
    the_template = np.mean(template_per_trial, axis=0)

    print('VSDI signal data template built in ' + str(datetime.datetime.now().replace(microsecond=0) - start_time_data) + '!\n')
    print('Global template building time: ' + str(datetime.datetime.now().replace(microsecond=0) - start_time) + '!\n')    
    return the_template, template_per_trial, alignementP1, alignementP2, data_heart_beat_sr100

def removing_the_template(detrended_data, template_per_trial, alignment_P1):
    # Filter the data with the template
    length_template = template_per_trial.shape[1]
    
    adapted_template_per_trial = np.zeros(detrended_data.shape)

    for tt in range(detrended_data.shape[0]):
        trial_duration = detrended_data.shape[1] - alignment_P1[tt]
        
        the_template = template_per_trial[tt]

        x = np.arange(1, length_template + 1)
        Xq = np.linspace(1, length_template, trial_duration)

        for xx in range(detrended_data.shape[2]):
            for yy in range(detrended_data.shape[3]):
                adapted_template = interp1d(x, the_template[:, xx, yy], axis=0)(Xq)
                aa = np.hstack((np.zeros(alignment_P1[tt]), adapted_template))
                adapted_template_per_trial[tt, :, xx, yy] = aa

    heart_less_data_pbp = detrended_data - adapted_template_per_trial

    return heart_less_data_pbp, adapted_template_per_trial

