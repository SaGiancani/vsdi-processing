import cv2 as cv
import numpy as np
from scipy import optimize
from scipy.ndimage.filters import convolve, gaussian_filter, median_filter, uniform_filter1d
from scipy.special import erf

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
    

def find_highest_sum_area(matrix, window_size, start_row=None, end_row=None, start_col=None, end_col=None):
    '''
    Description:
    The find_highest_sum_submatrix method is designed to identify the area within a 2D matrix
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

def gaussian1d(x, A, mu, sigma, alpha):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) * (1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma)))

def gaussian2d(xy_mesh, A, x0, y0, sigma_x, sigma_y, theta, skewness):
    x, y = xy_mesh
    x_diff = x - x0
    y_diff = y - y0
    x_rot = np.cos(theta) * x_diff - np.sin(theta) * y_diff
    y_rot = np.sin(theta) * x_diff + np.cos(theta) * y_diff
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    return A * np.exp(-(a * x_rot**2 + 2 * b * x_rot * y_rot * (1 + skewness) + c * y_rot**2))

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

