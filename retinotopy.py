import cv2 as cv

import json
import numpy as np
import os
import process_vsdi as process

from scipy.ndimage.filters import gaussian_filter
import utils


class Retinotopy:
    def __init__(self, 
                 session_path,
                 cond_name = None,
                 name = None, 
                 session_name = None, 
                 signal = None, 
                 averaged_simple_retino_pos = None, 
                 distribution_centroids = list(),
                 blob = None, 
                 mask = None):

        self.path_session = session_path
        self.cond_name = cond_name
        self.name = name
        self.session_name = session_name
        self.signal = signal
        self.retino_pos = averaged_simple_retino_pos
        self.distribution_positions = distribution_centroids
        self.blob = blob
        self.mask = mask
        self.time_limits = self.get_time_limits()

    def store_retino(self, t):
        tp = [self.path_session, self.cond_name, self.name, self.session_name, self.signal, self.retino_pos, self.distribution_positions, self.blob, self.mask, self.time_limits]
        utils.inputs_save(tp, os.path.join(t,'retino','retinotopy_'+self.cond_name))
        return
    
    def load_retino(self, path):
        tp = utils.inputs_load(path)
        self.path_session = tp[0]
        self.cond_name = tp[1]
        self.name = tp[2]
        self.session_name = tp[3]
        self.signal = tp[4]
        self.retino_pos = tp[5]
        self.distribution_positions = tp[6]
        self.blob = tp[7]
        self.mask = tp[8]
        self.time_limits = tp[9]
        return

    def get_time_limits(self):
        '''
        Reading metadata json file for time limits
        '''
        #sub + '_' + i.split('exp-')[1] + '_'+path_session.split('sess-')[1][0:12]
        tmp = utils.find_thing('json_data.json', self.path_session)
        # If also with find_thing there is no labelConds.txt file, than loaded as name Condition n#
        if len(tmp) == 0:
            self.log.info('Check the json_data.json presence inside the session folder and subfolders')
            return None
        # else, load the labelConds from the alternative path
        else :
            f = open(tmp[0])
            # returns JSON object as a dictionary
            data = json.load(f)
            a = json.loads(data)
            print('Time limits loaded successfully')
            return ((int(a[list(a.keys())[0]]['bottom limit']), int(a[list(a.keys())[0]]['upper limit'])))

    def get_retinotopic_features(FOI, min_lim=80, max_lim = 100, circular_mask_dim = 100, mask_switch = True):
        blurred = gaussian_filter(np.nan_to_num(FOI, copy=False, nan=0.000001, posinf=None, neginf=None), sigma=1)
        _, centroids, blobs = process.detection_blob(blurred, min_lim, max_lim)
        if mask_switch:
            circular_mask = utils.sector_mask(np.shape(blurred), (centroids[0][1], centroids[0][0]), circular_mask_dim, (0,360))
        else:
            circular_mask = None
        return centroids, blobs, circular_mask, blurred

    def centroid_poly(X, Y):
        """
        https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        """
        N = len(X)
        # minimal sanity check
        if not (N == len(Y)): raise ValueError('X and Y must be same length.')
        elif N == 1:
            Cx = X
            Cy = Y
            return int(Cx), int(Cy)
        elif N == 2:
            Cx = sum(X)/len(X)
            Cy = sum(Y)/len(Y)
            return int(Cx), int(Cy)
        elif N>2:
            sum_A, sum_Cx, sum_Cy = 0, 0, 0
            last_iteration = N-1
            # from 0 to N-1
            for i in range(N):
                if i != last_iteration:
                    shoelace = X[i]*Y[i+1] - X[i+1]*Y[i]
                    sum_A  += shoelace
                    sum_Cx += (X[i] + X[i+1]) * shoelace
                    sum_Cy += (Y[i] + Y[i+1]) * shoelace
                else:
                    # N-1 case (last iteration): substitute i+1 -> 0
                    shoelace = X[i]*Y[0] - X[0]*Y[i]
                    sum_A  += shoelace
                    sum_Cx += (X[i] + X[0]) * shoelace
                    sum_Cy += (Y[i] + Y[0]) * shoelace
            A  = 0.5 * sum_A
            factor = 1 / (6*A)
            Cx = factor * sum_Cx
            Cy = factor * sum_Cy
            # returning abs of A is the only difference to
            # the algo from above link
            return int(np.round(Cx)), int(np.round(Cy))#, abs(A)

    def centroid_max(X, Y, data):
        '''
        Pick the point in the matrix data with higher value.
        X and Y are list of x and y coordinates.
        The method returns the coordinates and the value of higher point.
        '''
        max_point = -100000000
        for x, y in zip(X, Y):
            if data[x, y] > max_point:
                index = (x, y)
                max_point = data[x, y]
        return index, max_point


    def single_seq_retinotopy(self,df_f0, 
                            global_centroid,
                            dim_side,
                            start_frame,
                            end_frame,
                            df_confront = None,
                            df_confront_foi = None,
                            df_f0_foi = None,
                            zero_frames = 20,
                            lim_blob_detect = 75,
                            single_frame_analysis = False,
                            time_window = 1):
    
        '''
        The method gets as input:
        df_f0: 3 dimensional matrix
        global_centroid: a tuple with the coordinates of a point
        dim_side: the dimension of the window -square- centered on global_centroid
        start_frame and end_frame: start and end frames to consider for averaging within for obtaining the output
        df_confront: if not None, corresponds to the df_f to subtract to df_f0 -for AM123-AM12 subtraction-
        df_confront_foi: the time window to consider for df_confront: it has to be same dimension of df_f0_foi
        df_f0_foi: the time window to consider for df_f0.
        zero_frames: the number of first frames to consider for computing blank signal and blank standard -for zscore-
        lim_blob_detect: threshold for blob detection for get_retinotopic_features method
        single_frame_analysis: boolean flag: if it's true the method stores one centroid per frame of the zscore. 
                            Useful for trajectory analysis.
        
        It returns:
        (c, d): a tuple with the maximum response centroid detected in the averaged time windowed zscore. It is normalized
                with the global_centroid distance.
        blobs: list of contours in the averaged time windowed zscore
        centroids: list of all the detected centroids in the averaged time windowed zscore
        (a, b): a tuple with the maximum response centroid detected in the averaged time windowed zscore. It is NOT normalized
        ztmp: a 3 dimensional matrix with the zscore computed in the FOI indicated by df_confront_foi and df_f0_foi.
        single_centroids: is a list of tuples with all the centroids found in each frame of the zscore.
        
        If the method has global_centroid indicated, it crops the df_f0 and the df_confront of a square of dim_side of side, centered 
        on the global_centroid. It performs signal blank and standard blank on the cropped df_f0. Than if the df_confront is provided
        it performs the subtraction between the two matrices: if either df_confront_foi and df_f0_foi are provided, it timewindows the
        two signals. If the df_confront is not provided it performs the zscore only on df_f0.
        '''
            
        
        # Considering small portion of the frame, corresponding to a square of dim_side pixel of side, centered on blob centroid
        if global_centroid is not None:
            check_seq =df_f0[:, (global_centroid[1]-(dim_side//2)):(global_centroid[1]+(dim_side//2)), 
                            (global_centroid[0]-(dim_side//2)):(global_centroid[0]+(dim_side//2))]
            
            sig_blank = np.mean(check_seq[:zero_frames, :, :], axis = 0)
            std_blank = np.std(check_seq[:zero_frames, :, :], axis = 0)/np.sqrt(np.shape(check_seq[:, :, :])[0])# Normalization of standard over all the frames, not only the zero_frames        
        
            # Check for presence of df to subtract to df_f0: used for single trial analysis in AMstrokes
            if df_confront is not None:
                df_confront = df_confront[:, (global_centroid[1]-(dim_side//2)):(global_centroid[1]+(dim_side//2)), 
                                (global_centroid[0]-(dim_side//2)):(global_centroid[0]+(dim_side//2))]
        else:
            check_seq = df_f0
            sig_blank = np.mean(check_seq[:zero_frames, :, :], axis = 0)
            std_blank = np.std(check_seq[:zero_frames, :, :], axis = 0)/np.sqrt(np.shape(check_seq[:, :, :])[0])# Normalization of standard over all the frames, not only the zero_frames        
                
        # Check for presence of df to subtract to df_f0: used for single trial analysis in AMstrokes
        if df_confront is not None:
            # FOI for each of the signal elements: either AM or single stroke dF/F0  
            if (df_confront_foi and df_f0_foi) is not None:
                tmp = check_seq[df_f0_foi[0]:df_f0_foi[1], :, :] - df_confront[df_confront_foi[0]:df_confront_foi[1], :, :]
            else:
                tmp = check_seq - df_confront
            ztmp = process.zeta_score(tmp[start_frame:end_frame, :, :], sig_blank, std_blank, full_seq = True)
        else:
            ztmp = process.zeta_score(check_seq[start_frame:end_frame, :, :], sig_blank, std_blank, full_seq = True)
        
        # If want to store information from single frame
        if single_frame_analysis:
            single_centroids = list()
            # Strategy for time windowing
            for i in range(len(ztmp)):
                if time_window==1:
                    print(f'the {i}th frame')
                    tmp_ = ztmp[i, :, :]
                elif time_window >1:
                    if i == 0:
                        print(f'from 0 to {time_window//2}')
                        tmp_ = np.mean(ztmp[i:time_window//2, :, :], axis=0)                    
                    elif i<=time_window//2-1:
                        print(f'from 0 to {time_window//2}')
                        tmp_ = np.mean(ztmp[0:i:time_window//2, :, :], axis=0)

                    elif i>time_window//2-1:
                        try:
                            tmp_ = np.mean(ztmp[i-time_window//2:i+time_window//2, :, :], axis=0)
                            print(f'from {i-time_window//2} to {i+time_window//2}')
                        except:
                            tmp_ = np.mean(ztmp[i-time_window//2:, :, :], axis=0)
                            print(f'from {i-time_window//2} to {len(ztmp)}')

                centroids_singl, _, _, blurred_singl = self.get_retinotopic_features(tmp_, min_lim=lim_blob_detect, mask_switch = False)
                coords_singl = np.array(list(zip(*centroids_singl)))
                # Centroid at maximum response
                (a,b), _ = self.centroid_max(coords_singl[0], coords_singl[1], blurred_singl)
                # Centroid at the centroid of the polygon given by all the points
                #(a,b) = self.centroid_poly(coords_singl[0], coords_singl[1])
                
                # If global_centroid, then normalization of resulting centroid
                if global_centroid is None:
                    c,d = ((a,b))
                else:
                    c, d = ((global_centroid[0]-dim_side//2 + a, global_centroid[1]-dim_side//2 + b))
                single_centroids.append((c, d))
                
        else:
            single_centroids = []

        centroids, blobs, _, blurred = self.get_retinotopic_features(np.mean(ztmp, axis=0), min_lim=lim_blob_detect, mask_switch = False)
        coords = np.array(list(zip(*centroids)))
        (a,b), _ = self.centroid_max(coords[0], coords[1], blurred)
        
        if global_centroid is None:
            c,d = ((a,b))
        else:
            c, d = ((global_centroid[0]-dim_side//2 + a, global_centroid[1]-dim_side//2 + b))
        return (c, d), blurred, blobs, centroids, (a,b), ztmp, single_centroids


    def get_trajectory(xs, ys, limits):
        print(limits[0], limits[1])
        xs_to_fit =  np.round(np.linspace(limits[0], limits[1]-1, limits[1]-limits[0])) 
        ys_to_fit = np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs_to_fit))
        #ys_fitted = [int(round(np.interp(i, xs_to_fit, ys_to_fit))) for i in xs_fitted]
        return xs_to_fit, ys_to_fit

    def get_mask_on_trajectory(self, dims, xs, ys, radius = 2):
        up = int(np.max(xs))
        bottom = int(np.min(xs))
        x, y = self.get_trajectory(xs, ys, (bottom, up))
        xs_fit_lins = np.round(np.linspace(bottom,up,(up-bottom)*2))
        masks_small = [utils.sector_mask(dims, (int(round(np.interp(i, x, y))), i), radius, (0,360)) for i in xs_fit_lins]
        small_mask = sum(masks_small)
        small_mask[np.where(small_mask>1)] = 1
        return small_mask

    def rotate_distribution(xs, ys, theta = None):
        if theta is None:
            theta = -(np.arctan2(np.array([ys[-1]-ys[0]]), np.array([xs[-1] - xs[0]])))
        print(theta)
        # subtracting mean from original coordinates and saving result to X_new and Y_new 
        X_new = xs - np.mean(xs)
        Y_new = ys - np.mean(ys)

        X_apu = [np.cos(theta)*i-np.sin(theta)*j for i, j in zip(X_new, Y_new) ]
        Y_apu = [np.sin(theta)*i+np.cos(theta)*j for i, j in zip(X_new, Y_new) ]

        # adding mean back to rotated coordinates
        return X_apu + np.mean(xs), Y_apu + np.mean(ys), theta
 
        
                
        
