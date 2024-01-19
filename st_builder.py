import datetime, utils
import data_visualization as dv
import numpy as np
import os
import process_vsdi as process
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage import gaussian_filter1d, rotate
from scipy.stats import norm

AREA_MAXIMI_FOR_PEAK = 5

class SpatioTemporalMap:
    def __init__(self, 
                 data,
                 path_session,
                 trajectory_mask,
                 threshold_contour, 
                 rotation_theta,
                 onset_time,
                 condition_name,
                 condition_type = 'ss',
                 retino_pos = None,
                 retino_time = None,
                 is_delay = 30,
                 pixel_spacing = 0.037576,#mm 
                 sampling_rate = 100,
                 high_level = None,
                 low_level = None,
                 rotate_correction_factor = 0,
                 discard_thresh = 1e-7,
                 colors_ret = ['grey'], 
                 storing_path = None):
        self.signal                   = data
        self.path_session             = path_session
        self.storing_path             = storing_path
        self.session_name             = utils.get_session_id_name(self.path_session)
        self.trajectory_mask          = trajectory_mask
        self.rotation_angle           = rotation_theta
        self.rotate_correction_factor = rotate_correction_factor
        self.discard_thresh           = discard_thresh

        self.map, self.masked_data    = get_spatio_temporal_profile(self.signal, 
                                                                    self.trajectory_mask, 
                                                                    self.rotation_angle, 
                                                                    correction_factor = self.rotate_correction_factor, 
                                                                    discard_thresh = self.discard_thresh)
        # Pos and timing of peak
        if (retino_pos is not None) and (retino_time is not None):
            self.retino_pos                    = retino_pos
            self.retino_time                   = retino_time
        else:
            a , b = process.find_highest_sum_area(self.map, AREA_MAXIMI_FOR_PEAK)
            self.retino_pos                    = [a]
            self.retino_time                   = [b]          

        self.threshold_contour        = threshold_contour
        self.pixel_spacing            = pixel_spacing
        self.interstimulus_delay      = is_delay#ms
        self.sampling_rate            = sampling_rate

        # Contours level
        if (high_level is None) and (low_level is None):
            self.high_level   =  np.nanpercentile(self.map, 97.7)
            self.low_level    =  np.nanpercentile(self.map, 15)
        else:
            self.high_level   = high_level 
            self.low_level    = low_level

        self.onset_time            = onset_time
        self.condition_name        = condition_name
        self.condition_type        = condition_type

        # Adjusting color 
        if (self.condition_type == 'ss') and (len(self.retino_pos) == 1):
            self.colors_retinotopy = ['grey']
        else:
            self.colors_retinotopy = colors_ret

    def plot_maps(self):
        if self.storing_path is not None:
            tmp = dv.set_storage_folder(storage_path = self.storing_path, name_analysis = 'STProfiles')
            new_storing_path = os.path.join(tmp, f'STProfile_{self.condition_name}_{self.session_name}')  
        else:
            new_storing_path = None

        if len(self.retino_pos) >1:
            color_peak = 'teal'
            peak_traj = True
        else:
            color_peak = 'grey'
            peak_traj = False

        dv.plot_st(self.map,  
                   self.threshold_contour, 
                   self.trajectory_mask,
                   self.pixel_spacing,
                   retinotopic_pos = self.retino_pos,
                   retinotopic_time = self.retino_time, 
                   map_type = utils.PARULA_MAP,
                   st_title = self.condition_name,
                   onset_time = self.onset_time,
                   colors_retinotopy = self.colors_retinotopy,
                   draw_peak_traj = peak_traj,
                   is_delay = self.interstimulus_delay,#ms
                   sampling_fq = self.sampling_rate,#Hz
                   high_level = self.high_level,
                   color_peak = color_peak,
                   low_level = self.low_level,
                   store_path = new_storing_path)
        return

class SpatioTemporalSession:
    def __init__(self,):
        # self.threshold = 
        pass

def derivative_filter(arr, threshold):
    # Compute the derivative of the array
    derivative = np.diff(arr)

    # Find the indices where the absolute derivative is greater than the threshold
    indices_to_remove = np.where(np.abs(derivative) > threshold)[0] + 1

    # Set the corresponding values to NaN
    arr_filtered = arr.copy()
    arr_filtered[indices_to_remove] = np.nan
    return arr_filtered

def get_spatio_temporal_profile(frames, trajectory_mask, theta, correction_factor = 0, discard_thresh = 1e-7):
    '''
    The method get the frame/frames of the signal, apply a mask, designed on the trajectory of the motion,
    and extract the spatiotemporal profile: depending by the shape of the signal (frames variable of len(shape) == 2 or 3 ) the method
    returns a line or a matrix.
    Input: 
        frames: np.array either 2 or 3 dimensions. It is the signal variable
        trajectory_mask: np.array 2-D. It has to have the same shape of last two dimensions of frames. 
            It represent the spatial trajectory of motion.
        theta: float in rad. Corrective angle for making straight the trajectory mask.
        correction_factor: int, default 0. Sometimes the theta correction of the trajectory mask doesnt work properly.
            This variable allows a fast correction of the theta aberation.
        discard_thresh: float, default 1e-7. Discarding threshold for the nan pixel after masking. 
    Output:
        b: np.array either 1 or 2 dimensions. It is the array containing the spatiotemporal profiles. If the frames input
            was only one frame, then as output b is only a line. Otherwise is a matrix "of lines".
        rotated: np.array either 2 or 3 dimensions. It is the matrix containing the masked signal. 
            Debugging purposes for proper rotation. 
    '''
    # Rad to deg transformation
    theta_deg = (theta*180)/np.pi
    shape_data = frames.shape
    # Check for shape: if datacube goes in
    if len(shape_data) > 2:
        # Loop over the frames and trajectory masking
        profile_1 = np.array([i*trajectory_mask for i in frames])
        # Normalization for rotation
        copia = np.nan_to_num(profile_1, nan = discard_thresh)
        # Rotatation of the masked frames
        rotated = np.array([rotate(i, -theta_deg[0] +correction_factor, reshape=False) for i in copia])
        # Remasking the rotated frames
        rotated[np.where(abs(rotated) <= discard_thresh)] = np.nan
        # Mean over y axis for each rotated frame 
        img_p1 = np.array([np.nanmean(i, axis = 0) for i in rotated])
        b = np.transpose(img_p1)
        
    # If 2d matrix -frame- goes in.
    elif len(shape_data) == 2:
        # Trajectory masking
        copia = frames*trajectory_mask
        # Normalization for rotation
        copia = np.nan_to_num(copia, nan=0)
        # Rotation of the masked frames
        rotated = rotate(copia, -theta_deg[0] +correction_factor, reshape=False)
        # Remasking the rotated frames
        rotated[np.where(abs(rotated) <= discard_thresh)] = np.nan
        # Mean over y axis for the frame 
        b = np.nanmean(rotated, axis=0)
    return b, rotated

def get_linear_expectation(array_of_sequences, global_shift, nonlinear_zeroframe=5):
    """
    Calculate the linear expectation of a sequence of arrays.

    Parameters:
    - array_of_sequences (list of 2D numpy arrays): List containing multiple time sequences.
    - global_shift (int): Number of time steps to shift the sequences globally.
    - nonlinear_zeroframe (int, optional): Number of frames to use for nonlinear zeroing. Default is 5.

    Returns:
    - tmp (2D numpy array): Linear expectation of the input sequences.
    """
    # Get the number of sequences
    n_strokes = len(array_of_sequences)
    
    # Ensure that there is more than one time sequence
    assert n_strokes != 1, 'The input has to be an array with more than one time sequence'
    
    # Calculate the step size for global shifting
    step = int(np.ceil(global_shift/2))
    
    # Copy the input sequences to avoid modifying the original data
    ppp = [np.copy(i) for i in array_of_sequences]
    
    # Calculate the linear expectation by iteratively combining the sequences
    for i in range(len(ppp)-1):
        if i == 0:
            tmp = ppp[i][step:, :] + ppp[i+1][:-step, :]
        else:
            tmp = tmp[step:, :] + ppp[i+1][:-(i+1)*step, :]
    
    # Calculate the mean of the first few frames for nonlinear zeroing
    zero_tmp = np.nanmean(tmp[:nonlinear_zeroframe, :, :], axis=0)
    
    # Zero the result using the calculated mean
    tmp = tmp - zero_tmp
    
    # Return the linear expectation
    return tmp

import numpy as np

def maximi_inda_blob(st_matrix, blob):
    """
    Find the indices of the maximum values in a given matrix multiplied by a binary blob.

    Parameters:
    - st_matrix (numpy.ndarray): The input matrix.
    - blob (numpy.ndarray): The binary blob used for masking.

    Returns:
    - List of tuples: Each tuple contains the maximum values' indices along with their corresponding positions.

    Description:
    This function takes a matrix (st_matrix) and a binary blob. It computes the element-wise product of st_matrix
    and blob and finds the indices of the maximum values along the columns. If a maximum is not detected for a column,
    the corresponding entry in the output array is set to NaN. The result is returned as a list of tuples, where each
    tuple contains the maximum values' indices and their corresponding positions.

    Note:
    - The result includes NaN for positions where no maximum is detected.

    Example Usage:
    st_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    blob = np.array([1, 1, 1])
    result = maximi_inda_blob(st_matrix, blob)
    print(result)
    """
    # Create an array filled with NaN values for the case when no maximum is detected
    zero_array = np.full((blob.shape[1]), np.nan)

    # Find the indices of maximum values after applying the blob
    ty = np.nanargmax(st_matrix * blob, axis=0)

    # Create a mask to handle cases where the maximum is at index 0
    mask_blob = np.ones(ty.shape)
    mask_blob[np.where(ty == 0)[0]] = 0

    # Apply the mask to the indices
    ty = ty * mask_blob

    # Extract valid indices and corresponding positions
    y = ty[np.where(ty != 0)]
    x = np.where(mask_blob == 1)[0]

    # Update the zero_array with valid indices
    zero_array[x] = y

    # Create a list of tuples containing the maximum values' indices and their corresponding positions
    return list(zip(zero_array, np.linspace(0, st_matrix.shape[1]-1, st_matrix.shape[1])))

def rotate_map(profile_1, theta, correction_factor = 0, discard_thresh = 1e-5, kernel = 15):
    # Rad to deg transformation
    theta_deg = (theta*180)/np.pi
    # shape_data = profile_1.shape
    copia = np.nan_to_num(profile_1, nan = discard_thresh)
    # Rotatation of the masked frames
    rotated = np.array(rotate(copia, -theta_deg[0] +correction_factor, reshape=False))
    # Remasking the rotated frames
    rotated = median_filter(rotated, kernel)
    rotated[np.where(abs(rotated) <= discard_thresh*20)] = np.nan
    return rotated

