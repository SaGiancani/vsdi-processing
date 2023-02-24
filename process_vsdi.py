import cv2 as cv
import numpy as np
from scipy.ndimage.filters import uniform_filter1d, gaussian_filter, median_filter
from scipy import optimize

def deltaf_up_fzero(vsdi_sign, n_frames_zero, deblank = False, blank_sign = None, outlier_tresh = 1000):
    '''F/F0 computation with -or without- demean of n_frames_zero and killing of outlier 
		----------
		vsdi_sign : np.array, with shape nframes, width, height
        n_frames_zero: int, the number of frames taken as zero, aka prestimulus
        demean: bool, switch for demeaning the signal: F-mean(F[0:n_frames_zero])/mean(F[0:n_frames_zero]) if True
                F/mean(F[0:n_frames_zero]) if False 
        outlier_tresh: int, 1000 by default, over this -absolute- threshold the pixel-value is put to 0  
		Returns
		-------
		df_fz : np.array, with shape nframes, width, height
    '''
    #mean_frames_zero = np.nanmean(vsdi_sign[:n_frames_zero, :, :], axis = 0)
    mean_frames_zero = np.nanmean(vsdi_sign[:n_frames_zero, :, :], axis = 0)
    #mean_frames_zero[np.where(mean_frames_zero==0)] = np.min(mean_frames_zero)
    # The case for precalculating the blank signal or not deblank at all
    if (deblank and (blank_sign is None)):
        #t_val = np.nanmean(np.ma.masked_invalid(vsdi_sign))
        #vsdi_sign = np.nan_to_num(vsdi_sign,nan= t_val, posinf = t_val, neginf= t_val) 
        
        df_fz= (vsdi_sign/mean_frames_zero) 
    # The case for calculating the signal deblanked
    elif deblank and (blank_sign is not None):
        #t_val = np.nanmean(np.ma.masked_invalid(vsdi_sign))
        #vsdi_sign = np.nan_to_num(vsdi_sign,nan= t_val, posinf = t_val, neginf= t_val)
        
        #t_val = np.nanmean(np.ma.masked_invalid(blank_sign))
        #blank_sign = np.nan_to_num(blank_sign,nan= t_val, posinf = t_val, neginf= t_val)          
        
        df_fz = ((vsdi_sign/mean_frames_zero)/(blank_sign)) - 1
    # The case without deblank
    elif (not deblank):
        #t_val = np.nanmean(np.ma.masked_invalid(vsdi_sign))
        #vsdi_sign = np.nan_to_num(vsdi_sign,nan= t_val, posinf = t_val, neginf= t_val) 
        
        df_fz = (vsdi_sign/mean_frames_zero) -1
    # Conceptually problematic subtraction, if used in combination with first frame subtraction.         
    #df_fz = df_fz - df_fz[0, :, :] 
    #t_val = np.nanmean(np.ma.masked_invalid(df_fz))
    #df_fz = np.nan_to_num(df_fz,nan= t_val, posinf = t_val, neginf= t_val)
    #df_fz[np.where(np.abs(df_fz)>outlier_tresh)] = 0
    return df_fz

def detection_blob(averaged_zscore, min_lim=80, max_lim = 100, min_2_lim = 99, max_2_lim = 100, std = 15, adaptive_thresh = True):
    averaged_zscore = np.nan_to_num(averaged_zscore, copy=False, nan=np.nanmin(averaged_zscore), posinf=None, neginf=None)# This could be an issue: using nanmin and divide the results by 10
    # Adaptive thresholding: if true it computes the percentile for thresholding, otherwise the threshold has to be provided
    if adaptive_thresh:
        min_thresh = np.nanpercentile(averaged_zscore, min_lim)
        max_thresh = np.nanpercentile(averaged_zscore, max_lim)
    else:
        min_thresh = min_lim
        max_thresh = max_lim
    #print((min_thresh, max_thresh))
    #print(( np.percentile(averaged_zscore, 80),  np.percentile(averaged_zscore, 100)))
    # Thresholding of z_score
    _, threshed = cv.threshold(averaged_zscore, min_thresh, max_thresh, cv.THRESH_BINARY)
    # Median filter against salt&pepper noise
    blurred_median = median_filter(threshed, size=(3,3))
    # Gaussian filter for blob individuation
    blurred = gaussian_filter(np.nan_to_num(blurred_median, copy=False, nan=np.nanmin(blurred_median), posinf=None, neginf=None), sigma=std)# This could be an issue: using nanmin and divide the results by 10
    
    # Blob detection
    # Adaptive thresholding: if true it computes the percentile for thresholding, otherwise the threshold has to be provided
#    if adaptive_thresh:
    min_thresh2 = np.nanpercentile(blurred, min_2_lim)
    max_thresh2 = np.nanpercentile(blurred, max_2_lim)
#    else:
#        min_thresh2 = min_2_lim
#        max_thresh2 = max_2_lim
    #print((min_thresh2, max_thresh2))

    _, blobs = cv.threshold(blurred, min_thresh2, max_thresh2, cv.THRESH_BINARY)
    # Normalization and binarization
    blobs = blobs/np.max(blobs)
    blobs = blobs.astype(np.uint8)
    # Contours and centroid detections
    contours, _ = cv.findContours(blobs, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
            #print((cx, cy))    
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
        to_app = np.nanmean(np.ma.masked_array(i, mask = roi_mask))
        roi_sign.append(to_app)
    return np.array(roi_sign)

def gauss_1d(x, a, mean, stddev):
    return a*np.exp((-(x-mean)**2)/(2*stddev**2))
    #return a * np.exp(-((x - mean) / 4 / stddev)**2)
    #return a * np.exp(-(x - mean)**2 / (2 * stddev**2))

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
    popt,pcov = optimize.curve_fit(gauss_1d, ax, proj)#, bounds=bounds) or ,maxfev = 5000)
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
        stder_sign_overcond = np.nanstd(sig_cond[:, :, :, :], axis = 0)/np.sqrt(np.shape(sig_cond)[0])

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
            stder_sign_overcond = 0
        else:        
            # Condition mean and stder computation
            mean_sign_overcond = np.nanmean(sig_cond[:, :, :, :], axis = 0)
            stder_sign_overcond = np.nanstd(sig_cond[:, :, :, :], axis = 0)/np.sqrt(np.shape(sig_cond)[0])
    
    # Try to fix the zscore defected for Hip AM3Strokes second session.
    #zscore = np.nan_to_num(np.nan_to_num(mean_sign_overcond-mean_signblnk_overcond)/np.nan_to_num(np.sqrt(stder_signblnk_overcond**2 + stder_sign_overcond**2)))
    #print(mean_sign_overcond.shape, mean_signblnk_overcond.shape)
    A = mean_sign_overcond-mean_signblnk_overcond
    B = np.sqrt(stder_signblnk_overcond**2 + stder_sign_overcond**2)
    #zscore = np.true_divide(A,  B, where = (A!=np.nan) | (B!=np.nan))#+eps)
    #zscore = np.nan_to_num(zscore, copy=False, nan=-0.0000001, posinf=10e+07, neginf=-0.0000001)
    zscore = A/B
    return zscore

