import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
from scipy import optimize
from scipy.ndimage.filters import convolve, gaussian_filter, median_filter, uniform_filter1d
from scipy.special import erf

from trajectory import get_trajectory


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
    if (len(signal3d_.shape) == 3) or (len(signal3d_.shape) == 1):
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


def deltaf_up_fzero(vsdi_sign, n_frames_zero, deblank = False, blank_sign = None):
    '''F/F0 computation with -or without- demean of n_frames_zero and killing of outlier 
		----------
		vsdi_sign : np.array, with shape nframes, width, height
        n_frames_zero: int, the number of frames taken as zero, aka prestimulus
        demean: bool, switch for demeaning the signal: F-mean(F[0:n_frames_zero])/mean(F[0:n_frames_zero]) if True
                F/mean(F[0:n_frames_zero]) if False 
		Returns
		-------
		df_fz : np.array, with shape nframes, width, height
    '''
    #mean_frames_zero = np.nanmean(vsdi_sign[:n_frames_zero, :, :], axis = 0)
    if len(vsdi_sign.shape) != 3:
        print('Data input not a 3d matrix!')
        return

    mean_frames_zero = np.nanmean(vsdi_sign[:n_frames_zero, :, :], axis = 0)
    #mean_frames_zero[np.where(mean_frames_zero==0)] = np.min(mean_frames_zero)
    # The case for precalculating the blank signal or not deblank at all
    if (deblank and (blank_sign is None)):
        df_fz= (vsdi_sign/mean_frames_zero) 
    # The case for calculating the signal deblanked
    elif deblank and (blank_sign is not None):
        df_fz = ((vsdi_sign/mean_frames_zero)/(blank_sign)) - 1
    # The case without deblank
    elif (not deblank):
        df_fz = (vsdi_sign/mean_frames_zero) -1

    return df_fz

def detection_blob(averaged_zscore, min_lim=80, max_lim = 100, min_2_lim = 97, max_2_lim = 100, std = 15, adaptive_thresh = True, kind = 'zscore'):#From 90 to 99 of min_2_lim
    '''
    Method for automatic detection of blobs, contours and their centroids.
    '''

    #averaged_zscore = np.nan_to_num(averaged_zscore, copy=False, nan=-0.000001, posinf=None, neginf=None)# This could be an issue: using nanmin and divide the results by 10
    # Adaptive thresholding: if true it computes the percentile for thresholding, otherwise the threshold has to be provided
    dim_data = len(averaged_zscore.shape)
    if dim_data == 2:
        # Adaptive threshold for signal profile extraction: if you keep this always same, and the max2 and min2 always at same percentile, it's gonna be fair.
        if adaptive_thresh:
            min_thresh = np.nanpercentile(averaged_zscore, min_lim)
            max_thresh = np.nanpercentile(averaged_zscore, max_lim)
        else:
            min_thresh = min_lim
            max_thresh = max_lim
        # print('get_signal_profile called')
        averaged_zscore = np.nan_to_num(averaged_zscore, nan=np.nanmin(averaged_zscore), neginf=np.nanmin(averaged_zscore[np.where(averaged_zscore != -np.inf)]), posinf=np.nanmax(averaged_zscore[np.where(averaged_zscore != np.inf)]))
        blurred = get_signal_profile(averaged_zscore, min_thresh, max_thresh, std = std)

        if kind == 'zscore':
            # Blob detection
            min_thresh2 = np.nanpercentile(blurred, min_2_lim)
        elif kind == 'df':
            min_thresh2 = 2*np.nanstd(blurred)

        max_thresh2 = np.nanpercentile(blurred, max_2_lim)#100

        contours, centroids, blobs = get_significant_sign(blurred, min_thresh2, max_thresh2)


        return contours, centroids, blobs
    
    elif dim_data == 3:
        # Adaptive threshold for signal profile extraction: if you keep this always same, and the max2 and min2 always at same percentile, it's gonna be fair.
        if adaptive_thresh:
            min_thresh = np.nanpercentile(averaged_zscore, min_lim)
            max_thresh = np.nanpercentile(averaged_zscore, max_lim)
        else:
            min_thresh = min_lim
            max_thresh = max_lim

        # Signal profile extraction over frames
        # print(min_thresh, max_thresh)
        # print('get_signal_profile called')
        averaged_zscore = np.nan_to_num(averaged_zscore, nan=np.nanmin(averaged_zscore), neginf=np.nanmin(averaged_zscore[np.where(averaged_zscore != -np.inf)]), posinf=np.nanmax(averaged_zscore[np.where(averaged_zscore != np.inf)]))
        data = [get_signal_profile(i, min_thresh, max_thresh) for i in averaged_zscore]
        data = np.asarray(data)
        
        if kind == 'zscore':
            # Blob detection
            min_thresh2 = np.nanpercentile(data, min_2_lim)
            # print(min_thresh2)
            # print(min_2_lim)
        elif kind == 'df':
            min_thresh2 = 2*np.nanstd(data)

        max_thresh2 = np.nanpercentile(data, max_2_lim)
        # print('Boundaries for get_significant_sign '+str(min_thresh2) + ' -- '+str(max_thresh2))
        countours_ = list()
        centroids_ = list()
        blobs_ = list()
        
        for i in data:
            # Thresholding and blobs detection
            contours, centroids, blobs = get_significant_sign(i, min_thresh2, max_thresh2)
            blobs_.append(blobs)
            countours_.append(contours)
            centroids_.append(centroids)
            
        return countours_, centroids_, blobs_
    
def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

def find_highest_sum_area(matrix, window_size, start_row=None, end_row=None, start_col=None, end_col=None):
    '''
    Description:
    The find_highest_sum_area method is designed to identify the area within a 2D matrix
    specified by the start and end row and column indices with the highest sum of elements. It 
    employs a sliding window approach to calculate the sum of elements within the specified 
    submatrix and identifies the central position of the area with the maximum sum.

    Parameters:
    matrix (numpy.ndarray): A 2D matrix (numpy array) containing numeric values.
    window_size (int): The size of the moving window or mask used to calculate the sum of elements
    within local regions.
    start_row (int or None): The starting row index of the submatrix. If None, the entire row dimension is considered.
    start_col (int or None): The starting column index of the submatrix. If None, the entire column dimension is considered.
    end_row (int or None): The ending row index of the submatrix. If None, the entire row dimension is considered.
    end_col (int or None): The ending column index of the submatrix. If None, the entire column dimension is considered.

    Return Value:
    max_position (tuple): A tuple containing the coordinates (row, column) of the central position
    within the area with the highest sum of elements.

    '''
    rows, cols = matrix.shape

    if start_row is None:
        start_row = 0
    if start_col is None:
        start_col = 0
    if end_row is None:
        end_row = rows
    if end_col is None:
        end_col = cols

    if start_row >= end_row:
        raise ValueError("start_row must be less than end_row")
    if start_col >= end_col:
        raise ValueError("start_col must be less than end_col")

    max_sum = -np.inf
    max_position = (0, 0)

    # Precompute cumulative sum
    cumsum_matrix = np.nancumsum(np.nancumsum(matrix, axis=0), axis=1)

    for i in range(start_row, min(end_row - window_size + 1, rows)):
        for j in range(start_col, min(end_col - window_size + 1, cols)):
            # Calculate the sum using the cumulative sum
            current_sum = cumsum_matrix[min(i + window_size, rows-1), min(j + window_size, cols-1)] \
                        - cumsum_matrix[min(i, rows - 1), min(j + window_size, cols-1)] \
                        - cumsum_matrix[min(i + window_size, rows-1), min(j, cols - 1)] \
                        + cumsum_matrix[min(i, rows - 1), min(j, cols - 1)]
            if current_sum > max_sum:
                max_sum = current_sum
                max_position = (i + window_size // 2, j + window_size // 2)

    return max_position

def get_best_coordinate(image, coords, radius=3):
    max_avg    = -np.inf
    best_coord = (None, None) 
    
    for x, y in coords:
        # Define neighborhood bounds
        x_min = max(0, x - radius)
        x_max = min(image.shape[0], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(image.shape[1], y + radius + 1)

        # Extract neighborhood
        neighborhood = image[x_min:x_max, y_min:y_max]

        # Compute average ignoring NaNs
        avg = np.nanmean(neighborhood)

        # Update max if needed
        if avg > max_avg:
            max_avg = avg
            best_coord = (x, y)

    return best_coord


def get_centroids(contours):
    # Centroids detection
    centroids = list()
    #conts = list()
    for i in contours:
        #conts.append(np.squeeze(i))
        M = cv.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx, cy))
    return centroids

def get_blobs(blurred, min_thresh2, max_thresh2, smoother_kernel = 10):
    _, blobs = cv.threshold(blurred, min_thresh2, max_thresh2, cv.THRESH_BINARY)
    # Smoother for salt and pepper noise at the edge from the previous filter
    blobs = median_filter(blobs, (smoother_kernel,smoother_kernel))
    # Normalization and binarization
    blobs = blobs/np.nanmax(blobs)
    blobs = blobs.astype(np.uint8)
    return blobs

def get_significant_sign(blurred, min_thresh2, max_thresh2):
    blobs = get_blobs(blurred, min_thresh2, max_thresh2)
    # Contours and centroid detections
    contours, _ = cv.findContours(blobs, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    centroids = get_centroids(contours)
    return contours, centroids, blobs 

def get_signal_profile(averaged_zscore, min_thresh, max_thresh, std = 15):
    # Thresholding of z_score
    _, threshed = cv.threshold(averaged_zscore, min_thresh, max_thresh, cv.THRESH_BINARY)
    # Median filter against salt&pepper noise
    blurred_median = median_filter(threshed, size=(3,3))
    # Gaussian filter for blob individuation
    blurred = gaussian_filter(np.nan_to_num(blurred_median, copy=False, nan=np.nanmin(blurred_median), posinf=None, neginf=None), sigma=std)
    # print(np.nanmin(blurred), np.nanmax(blurred))
    return blurred


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
    # assert len(signal) != 3, 'Datacube required'
    if len(signal.shape) == 3:
        time, space_y, space_x = signal.shape
        fitted_cube = np.empty((time, space_y, space_x))
        for i in range(space_y):
            for j in range(space_x):
                tmp = get_trajectory(np.arange(start, stop, 1), signal[start:stop, i, j], (0, time))
                fitted_cube[:, i, j] = tmp[1]
    elif (len(signal.shape) == 1) or ((len(signal.shape) >10)):
        data_bins = signal.shape[0]
        fitted_cube = np.empty((data_bins))
        tmp = get_trajectory(np.arange(start, stop, 1), signal[start:stop], (0, data_bins))

    return fitted_cube


def manual_thresholding(data, threshold, filter_kernel = 30):
    assert len(data.shape) == 2, 'The data matrix has to be 2D'
    tmp = median_filter(data, (filter_kernel, filter_kernel))
    max_thresh = np.nanpercentile(data, 99)
    contours, centroids, blobs = get_significant_sign(tmp, threshold, max_thresh)
    return contours, centroids, blobs 

def time_course_signal(df_fz, roi_mask):#, hand_made=False):
    """
    Computes the signal in ROI. 
    It recalls initially the mask_roi method, than it computes demeaning of the signal.
    Parameter
    -----------
        self Object
        bnnd_img: numpy.array (70, width of binning, height of binning), the binned signal
    Returns
    -----------
        self.roi_sign: numpy.array (70,1) the signal inside the ROI, represented as a 1D array
    """
    roi_sign = list()
    for i in df_fz:
        masked_array = np.ma.masked_array(i, mask = roi_mask)
        to_app = np.nanmean(masked_array)
        roi_sign.append(to_app)
    return np.array(roi_sign)

def gaussian3d(data , size = 3, std = .65):
    # Define the standard deviations for each dimension (t, y, x)
    sigma_t = std
    sigma_y = std
    sigma_x = std

    # Create a 3D Gaussian kernel
    kernel_size = (size, size, size)  # Adjust the size as needed
    t_kernel = np.linspace(-size/2, size/2, kernel_size[0])
    y_kernel = np.linspace(-size/2, size/2, kernel_size[1])
    x_kernel = np.linspace(-size/2, size/2, kernel_size[2])
    t, y, x = np.meshgrid(t_kernel, y_kernel, x_kernel)
    kernel = np.exp(-(t ** 2 / (2 * sigma_t ** 2) + y ** 2 / (2 * sigma_y ** 2) + x ** 2 / (2 * sigma_x ** 2)))
    kernel /= kernel.sum()  # Normalize the kernel

    # Apply the kernel to your 3D data
    smoothed_data = convolve(data, kernel, mode='reflect')
    return smoothed_data

def gaussian1d(x, A, mu, sigma, alpha = 0, beta = 0):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) * (1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))) * (1 + beta * ((x - mu) / sigma)**4)

def gaussian2d(xy_mesh, A, mu_x, mu_y, sigma_x, sigma_y, alpha_x, alpha_y, kurtosis_x = 0, kurtosis_y = 0):
    x, y = xy_mesh
    exponent_x = -(x - mu_x)**2 / (2 * sigma_x**2)
    exponent_y = -(y - mu_y)**2 / (2 * sigma_y**2)
    erf_term_x = 1 + erf(alpha_x * (x - mu_x) / (np.sqrt(2) * sigma_x))
    erf_term_y = 1 + erf(alpha_y * (y - mu_y) / (np.sqrt(2) * sigma_y))
    kurtosis_term_x = (x - mu_x)**4 / (sigma_x**4)
    kurtosis_term_y = (y - mu_y)**4 / (sigma_y**4)
    return A * np.exp(exponent_x + exponent_y) * erf_term_x * erf_term_y * np.exp(-0.5 * (kurtosis_x * kurtosis_term_x + kurtosis_y * kurtosis_term_y))


def gaussian_fitting(td_mat, ax_to_fit, perc_wind = 3):
    if len(np.shape(td_mat)) > 2:
        print('The matrix to fit has to be two or mono dimensional')
        return
    
    elif len(np.shape(td_mat)) == 2:
        dim = np.shape(td_mat)[ax_to_fit]
        if ax_to_fit == 0:
            proj = np.mean(td_mat, axis = 1)
        else:        
            proj = np.mean(td_mat, axis = 0)
    
    elif len(np.shape(td_mat)) == 1:
        dim = len(td_mat)
        proj = td_mat

    ax = np.linspace(0, 1, dim)
    #print(np.min(proj))
    #proj = proj-proj[0]
    proj = proj-np.min(proj)
    proj = round(uniform_filter1d(proj, size=(len(proj)/100)*perc_wind)) # Moving Average Filter: check the result
    popt,pcov = optimize.curve_fit(gaussian1d, ax, proj)#, bounds=bounds) or ,maxfev = 5000)
    return ax, proj, popt, pcov 

def log_norm(y, mu, sigma):
    return 1/(np.sqrt(2.0*np.pi)*sigma*y)*np.exp(-(np.log(y)-mu)**2/(2.0*sigma*sigma))

def lognorm_fitting(array_to_fit, b= 50):
    # Normalization
    tmp = array_to_fit
    # Histogram computation
    h = np.histogram(tmp, bins=b)
    n = h[1]
    step = (n[1]-n[0])
    nrm = np.sum(h[0]*step)    
    fr = h[0]/nrm
    xx = n - 0.5*step
    ar = np.zeros((fr.shape[0]+1))
    ar[1:] = fr
    # lognormal Fitting
    params, _ = optimize.curve_fit(log_norm, xx, ar)
    mu = params[0]
    sigma = params[1]
    # Median + StdDev
    # Median + StdDev
    return ar, mu, sigma, xx

def lognorm_thresholding(array_to_fit, switch = 'median'):
    array_to_fit = array_to_fit/np.max(array_to_fit)
    tmp, mu, sigma, xx = lognorm_fitting(array_to_fit, b= 50)
    if switch == 'median':
        thresh = np.exp(mu)
    elif switch == 'mean':
        thresh = np.exp(mu + sigma*sigma/2.0)
    thresh_std = (thresh + 2*np.sqrt((np.exp(sigma*sigma)-1)*np.exp(mu+mu+sigma*sigma)))
    select_trials_id = np.where(((array_to_fit)<(thresh_std)))[0].tolist()
    return select_trials_id, (tmp, mu, sigma, xx), array_to_fit.tolist()
    
def sobel_filter(im, k, N):
    (nrows, ncols) = im.shape
    sobelx = cv.Sobel(im,ddepth=cv.CV_64F, dx=1,dy=0,ksize=k) #cv.CV_64F
    sobely = cv.Sobel(im,ddepth=cv.CV_64F, dx=0,dy=1,ksize=k)
    sobel = np.zeros(im.shape)
    if N=='self':
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = sobel/np.amax(sobel)
    else: 
        sobel = np.sqrt(sobelx**2 + sobely**2)/N
    return sobel


def zeta_score(sig_cond, sig_blank, std_blank, full_seq = False, zero_frames = 20):
    #eps = np.nanmin(sig_cond)
    # Security check
    if len(np.shape(sig_cond))<3 or len(np.shape(sig_cond))>4:
        print('The signal has to be 3 or 4 dimensional')
        return
    # Case for average over trials for a condition
    elif len(np.shape(sig_cond))==4:
        # Blank mean and stder computation
        if (sig_blank is None) or (std_blank is None):
            mean_signblnk_overcond = np.nanmean(sig_cond[:, :zero_frames, :, :], axis = 0)
            stder_signblnk_overcond = np.nanstd(sig_cond[:, :zero_frames, :, :], axis = 0)/np.sqrt(np.shape(sig_cond)[0])# Normalization of standard over all the frames, not only the zero_frames        
        else:
            mean_signblnk_overcond = sig_blank
            stder_signblnk_overcond = std_blank#np.std(sig_blank[:, :, :], axis = 0)/np.sqrt(np.shape(sig_blank)[0])

        # Condition mean and stder computation
        mean_sign_overcond = np.nanmean(sig_cond[:, :, :, :], axis = 0)
        # stder_sign_overcond = np.nanstd(sig_cond[:, :, :, :], axis = 0)/np.sqrt(np.shape(sig_cond)[0])

    # Case for single trial analysis: full time sequence analysis    
    elif len(np.shape(sig_cond))==3:
        # Blank mean and stder computation
        if (sig_blank is None) or (std_blank is None):
            mean_signblnk_overcond = np.nanmean(sig_cond[:zero_frames, :, :], axis = 0)
            stder_signblnk_overcond = np.nanstd(sig_cond[:zero_frames, :, :], axis = 0)/np.sqrt(np.shape(sig_cond)[0])# Normalization of standard over all the frames, not only the zero_frames        
        else:
            mean_signblnk_overcond = sig_blank
            stder_signblnk_overcond = std_blank#np.std(sig_blank[:, :, :], axis = 0)/np.sqrt(np.shape(sig_blank)[0])
        
        if full_seq:
            # Condition mean and stder computation
            mean_sign_overcond = sig_cond
            # stder_sign_overcond = 0
        else:        
            # Condition mean and stder computation
            mean_sign_overcond = np.nanmean(sig_cond[ :, :, :], axis = 0)
            # stder_sign_overcond = np.nanstd(sig_cond[ :, :, :], axis = 0)/np.sqrt(np.shape(sig_cond)[0])
    
    # Try to fix the zscore defected for Hip AM3Strokes second session.
    #zscore = np.nan_to_num(np.nan_to_num(mean_sign_overcond-mean_signblnk_overcond)/np.nan_to_num(np.sqrt(stder_signblnk_overcond**2 + stder_sign_overcond**2)))
    #print(mean_sign_overcond.shape, mean_signblnk_overcond.shape)
    A = mean_sign_overcond-mean_signblnk_overcond
    B = stder_signblnk_overcond
    # B = np.sqrt(stder_signblnk_overcond**2 + stder_sign_overcond**2)
    zscore = A/B
    return zscore


def fig_to_img(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def make_video(datacube, contours, centroids = None, global_x= None, global_y= None, timing_onset = None, fps = 3.0, frame_titles = None, title = 'annotated_heatmap_video.avi', dpi = 200):
    (time_steps, height, width) = datacube.shape
    
    if frame_titles is not None:
        assert len(frame_titles) == np.shape(datacube)[0], 'frame_titles and datacube lengths have to be same'

    # Normalize the heatmaps globally for consistent coloring
    global_min = np.nanmin(datacube)
    global_max = np.nanmax(datacube)
    print(global_min, global_max)
    heatmaps_normalized = np.divide((datacube - global_min),(global_max - global_min))
    # Define the output video file parameters
    output_file = title
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for the video
      # Frames per second
        
    new_height = int(height * (dpi / 100))
    new_width = int(width * (dpi / 100))


    # Create a VideoWriter object
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height), isColor=True)
    
#     counter = 0
    colors = ['crimson', 'tomato', 'magenta']
    # Write each frame to the video
    for i in range(time_steps):
        heatmap = heatmaps_normalized[i]

        # Create a Matplotlib figure
        fig = Figure(figsize=(width / 100, height / 100), dpi=dpi)
        ax = fig.add_subplot(111)

        # Plot the heatmap
        cax = ax.imshow(heatmap, cmap=utils.PARULA_MAP, origin='lower', vmax = 1.1, vmin = .15)

        # Add a colorbar
#         fig.colorbar(cax, ax=ax)

        # Add annotations (customize as needed)
        ax.contour(contours[i, :, :], colors='k', linewidths=0.5)
        
        if centroids is not None:
            ax.scatter(centroids[i, 0], centroids[i, 1], color = 'r', marker = 'X')

        if timing_onset is not None:
            for count, t in enumerate(timing_onset):
                if i >= t:
                    ax.hlines(global_x[count],0, heatmap.shape[1], colors=colors[count], linestyles = '-', lw=1.5)    
                    
        
        # Remove axes for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
#         if frame_titles is not None:
#             ax.annotate(frame_titles[i], xy=(20, 20), xytext=(50, datacube.shape[1]-50), 
#                         textcoords='offset points', ha='center', 
#                         fontsize=50, color='k')            
        if frame_titles is not None:
            ax.text(10, datacube.shape[1]-30, frame_titles[i], color='white', fontsize=15, ha='left', va='top')

        # Convert Matplotlib figure to an image
        img = fig_to_img(fig)

        # Convert from RGB to BGR (OpenCV expects BGR)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize the image to the new dimensions
        img_bgr_resized = cv2.resize(img_bgr, (new_width, new_height))

        # Write the frame to the video
        video_writer.write(img_bgr_resized)

    # Release the VideoWriter object
    video_writer.release()


    print(f"Video saved as {output_file}")    
    return

def get_blobs_n_centroids(datacube, manual_thresh):
    blobs, centroids = list(), list()
    for frame in datacube:
        blob = np.zeros(frame.shape, dtype = bool)
        blob[np.where(frame>manual_thresh)] = 1
        # If there is a mask, it looks for maximi inside the blob
        if np.nansum(blob)>0:
            (x_, y_) = find_highest_sum_area(frame*blob, 20)
            centroids.append(np.array([y_, x_]))
        else:
            centroids.append(np.array([np.nan, np.nan]))
        blobs.append(blob)
    return np.array(blobs), np.array(centroids)


def bin_image(data, x_bnnd_size, y_bnnd_size):
    '''
    DESCRIPTION:
    Performs spatial binning on a 3D image dataset by resizing each frame using linear interpolation.

    PARAMETERS:
    data (numpy.ndarray)          : 3D array (Time, Height, Width), representing video frames.
    x_bnnd_size (int)            : Target width after binning.
    y_bnnd_size (int)            : Target height after binning.

    RETURNS:
    b (numpy.ndarray)             : Binned image data with resized frames.

    PROCESS:
    Iterates over all time frames in the dataset.
    Resizes each frame to the specified dimensions using OpenCV's linear interpolation.
    Stores the resized frames in a new array.

    EXAMPLE USAGE:
    binned_data = bin_image(video_data, x_bnnd_size=50, y_bnnd_size=50)    
    '''
    assert len(data.shape) == 3, 'Shape of data matrix wrong'
    time = data.shape[0]
    b = data
    tmp = np.zeros((time, y_bnnd_size, x_bnnd_size))
    for i in range(time):
        tmp[i, :, :] = cv.resize(np.array(b[i, :, :], dtype='float64'), 
                                 (x_bnnd_size, y_bnnd_size), 
                                 interpolation=cv.INTER_LINEAR)
    b = tmp
    return b

def get_binned_data(raw_data, bin = 1):
    '''
    DESCRIPTION:
    Applies spatial binning to a dataset if binning is enabled.

    PARAMETERS:
    raw_data (numpy.ndarray)      : 4D array (Ntrial, Ntime, Ny, Nx), representing video-frame trials.
    bin (int, optional)           : Binning factor. Default is 1 (no binning).

    RETURNS:
    data (numpy.ndarray)          : Binned or original dataset.

    PROCESS:
    Checks if binning is enabled (bin > 1).
    Computes new dimensions based on the binning factor.
    Applies bin_image() function to resize frames.
    Returns the processed dataset.

    EXAMPLE USAGE:
    data = get_binned_data(video_data, bin=2)    
    '''    
    # Binning strategy
    if bin > 1:
        x_new_size = raw_data.shape[-1]//bin
        y_new_size = raw_data.shape[-2]//bin    
        data = np.array([bin_image(i, x_new_size, y_new_size) for i in raw_data])
    else:
        data = raw_data       
    return data


