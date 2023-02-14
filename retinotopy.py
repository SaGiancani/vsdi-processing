import argparse, blk_file, datetime, json, os, utils
import cv2 as cv
import data_visualization as dv
import matplotlib.pyplot as plt
import middle_process as md
import numpy as np
import process_vsdi as process

from scipy.ndimage.filters import gaussian_filter

class RetinoSession(md.Session):
        def __init__(self, 
                     path_session, 
                     path_md, 
                     green_name,                   
                     spatial_bin = 3,
                     temporal_bin = 1,
                     zero_frames = None,
                     tolerance = 20,
                     mov_switch=False,
                     deblank_switch=False,
                     conditions_id =None,
                     chunks = 1,
                     strategy = 'mae',
                     logs_switch =False,  
                     base_report_name= 'BaseReport.csv',
                     base_head_dim = 19, 
                     logger = None, 
                     condid = None, 
                     store_switch = False, 
                     data_vis_switch = True, 
                     end_frame = None,
                     single_stroke_label = 'pos',
                     multiple_stroke_label = 'am',
                     **kwargs):
            #path_session, logs_switch = False, deblank_switch = False

            super(RetinoSession, self).__init__(path_session, 
                             spatial_bin = 3,
                             temporal_bin = 1,
                             zero_frames = None,
                             tolerance = 20,
                             mov_switch=False,
                             deblank_switch=False,
                             conditions_id =None,
                             chunks = 1,
                             strategy = 'mae',
                             logs_switch =False,  
                             base_report_name= 'BaseReport.csv',
                             base_head_dim = 19, 
                             logger = None, 
                             condid = None, 
                             store_switch = False, 
                             data_vis_switch = True, 
                             end_frame = None, 
                             **kwargs)

            if logger is None:
                self.log = utils.setup_custom_logger('myapp')
            else:
                self.log = logger        

            self.cond_names = None
            self.header = super().get_session_header(path_session, spatial_bin, temporal_bin, tolerance, mov_switch, deblank_switch, conditions_id, chunks, strategy, logs_switch)
            print(self.header)
            self.all_blks = md.get_all_blks(self.header['path_session'], sort = True) # all the blks, sorted by creation date -written on the filename-.

            if len(self.all_blks) == 0:
                print('Check the path: no blks found')
            
            self.single_stroke_label = single_stroke_label
            self.multiple_stroke_label = multiple_stroke_label
            print(self.single_stroke_label, self.multiple_stroke_label)
            self.path_session = path_session
            self.path_md = path_md

            # Corresponding single stroke for each AM condition
            self.retino_pos_am = get_conditions_correspondance(self.path_session)
            print(self.retino_pos_am)
            # All the conditions    
            self.cond_dict = super().get_condition_name()
            self.cond_names = list(self.cond_dict.values())
            # Extract blank condition id
            self.blank_id = super().get_blank_id(cond_id=condid)
            # Store all conditions
            self.cond_dict_all = self.cond_dict
            # Pick only inserted conditions and corresponding single positions
            self.cond_dict = self.get_conditions_intersect()
            # Name condition extraction
            self.cond_names = list(self.cond_dict.values())
            print(self.cond_dict)
            print(f'Only picked conditions: {self.cond_dict}')
            print(f'All session conditions: {self.cond_dict_all}')

            # Blank condition loading            
            self.blank_condition = self.get_blank()
            self.id_name = self.get_session_id_name()
            print('Session ID name: ' + self.id_name)
            self.green = self.get_green(green_name)
            self.mask = self.get_mask()
    
        # def get_condition_name(self):
        #     self.cond_dict = super().get_condition_name()
        #     print(self.single_stroke_label, self.multiple_stroke_label)
        #     # Two dictionaries, for type of conditions -pos or am-
        #     single_pos_conds = self.get_conditions_pos()
        #     am_conds = self.get_conditions_am()

        #     # Start from the single stroke conditions for storing and afterward showing the positions in AM conditions
        #     return {**single_pos_conds, **am_conds}                           

        def get_conditions_pos(self):
            return {k: v for k,v in self.cond_dict.items() if self.single_stroke_label.lower() in v.lower()}

        def get_conditions_am(self):
            return {k: v for k,v in self.cond_dict.items() if (self.single_stroke_label.lower() not in v.lower()) and (v.lower() != 'blank')}

        def get_conditions_intersect(self):
            conditions_id = self.header['conditions_id']
            print(f'The picked ID conditions are: {conditions_id}')
            # Start from the single stroke conditions for storing and afterward showing the positions in AM conditions
            am_conds = self.get_conditions_am()
            single_conds = self.get_conditions_pos()
            conds_full = {**single_conds, **am_conds}

            # Intersect the set of all the conditions with the picked one in the parser
            if conditions_id is not None:
                # Manual insert of condition id by key number
                conds = {k: v for k,v in conds_full.items() if k in conditions_id}
                # Taking the picked condition names
                conds_names = list(conds.values())
                # Taking the am conditions ONLY
                am_tmp = list(set(conds_names).intersection(set(am_conds.values())))
                # Taking the single stroke conditions that make the AM
                cond_t_list =  [j for v in am_tmp for j in self.retino_pos_am[v]]
                # Considering a sum of single stroke that make the picked AMs and unifying them to the one immediately picked -w/o repetition- 
                tmp = list(conds.values()) + cond_t_list
                all_considered_conds = list(set(tmp))
                # Rebuild dictionary with id as key and condition name as value
                conds = {k: v for k,v in conds_full.items() if v in all_considered_conds}
            else:
                conds = conds_full
            print(f'Conditions picked: {conds}')
            return conds
        
        
        def get_mask(self):
            # Loading handmade mask
            try:
                mask = np.load(os.path.join(self.path_session, 'derivatives','handmade_mask.npy'))
                (y_size, x_size) = self.blank_signal.averaged_df[0, :,:].shape
                x_bnnd_size = x_size
                y_bnnd_size = y_size
                mask = mask[0:y_bnnd_size , 0:x_bnnd_size ]

            except:
                print('No mask present in derivatives folder for session ' + self.id_name)
                mask = None
            return mask

        def get_session_id_name(self):                
            # Session names extraction
            sub_name, experiment_name, session_name = get_session_metainfo(self.path_session)
            id_name = sub_name + experiment_name + session_name
            return id_name

        def get_green(self, green_name):
            try:
                # Loading green
                green_path = utils.find_thing(green_name, self.path_session)
                green = cv.imread(green_path[0], cv.IMREAD_UNCHANGED)

                # Resizing green
                (y_size, x_size) = self.blank_signal.averaged_df[0, :,:].shape
                x_bnnd_size = x_size
                y_bnnd_size = y_size
                tmp = cv.resize(np.array(green, dtype='float64'), (x_bnnd_size, y_bnnd_size), interpolation=cv.INTER_LINEAR)
                green_ = np.copy(tmp)
                return green_

            except:
                print('No green '+ green_name+ ' present in rawdata folder for session ' + self.id_name)   
                return None

        def get_blank(self):
            #conds = list(session.cond_dict.keys())
            path_md_files = os.path.join(self.path_md,'md_data')

            # Blank condition loading
            cd_blank = md.Condition()
            try:
                cd_blank.load_cond(os.path.join(path_md_files, 'md_data_blank'))
                self.blank_condition = cd_blank
                print('Blank condition loaded succesfully!')
                #mean_blank = np.nanmean(cd_blank.averaged_df[:20, :,:], axis=0)
                #std_blank = np.nanstd(cd_blank.averaged_df[:20, :,:], axis=0)/np.sqrt(np.shape(cd_blank.averaged_df)[0]) 
            except:
                print('Blank condition not found')
                # In case of absence of the blank condition, it processes and stores it
                self.storage_switch = True
                self.visualization_switch = False
                # It is gonna get the blank signal automatically
                print('Processing blank signal')
                _ = self.get_signal(self.blank_id)
                self.storage_switch = False
                cd_blank.load_cond(os.path.join(path_md_files, 'md_data_blank'))
                self.blank_condition = cd_blank
                print('Blank condition loaded succesfully!')
            return 

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
                 mask = None,
                 green = None,
                 maps = None,
                 mask_tc = None,
                 tc = None,
                 averaged_tc = None,
                 df = None,
                 stroke_type = 'single stroke'):

        self.path_session = session_path
        self.cond_name = cond_name
        self.name = name
        self.session_name = session_name
        self.signal = signal
        self.retino_pos = averaged_simple_retino_pos
        self.distribution_positions = distribution_centroids
        self.blob = blob
        self.mask = mask
        if stroke_type is not None:
            self.time_limits = self.get_time_limits(stroke_type)
        self.green = green
        self.map = maps
        self.tc_mask = mask_tc
        self.time_courses = tc
        self.average_time_course = averaged_tc
        self.df_fz = df


    def store_retino(self, t):
        tp = [self.path_session,
              self.cond_name, 
              self.name, 
              self.session_name, 
              self.signal, 
              self.retino_pos, 
              self.distribution_positions, 
              self.blob, 
              self.mask, 
              self.time_limits, 
              self.green, 
              self.map, 
              self.tc_mask,
              self.time_courses,
              self.average_time_course,
              self.df_fz]
        storage_path = os.path.join(t, 'retino')
        tmp = dv.set_storage_folder(name_analysis = os.path.join(storage_path,))
        utils.inputs_save(tp, os.path.join(tmp,'retinotopy_'+self.cond_name))
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
        self.green = tp[10]
        self.map = tp[11]
        self.tc_mask = tp[12]
        self.time_courses = tp[13]
        self.average_time_course = tp[14]
        self.df_fz = tp[15]

        return


    def get_time_limits(self, stroke_type):
        '''
        Reading metadata json file for time limits
        '''
        #sub + '_' + i.split('exp-')[1] + '_'+path_session.split('sess-')[1][0:12]
        tmp = utils.find_thing('json_data.json', self.path_session)
        # If also with find_thing there is no labelConds.txt file, than loaded as name Condition n#
        if len(tmp) == 0:
            print('Check the json_data.json presence inside the session folder and subfolders')
            return None
        # else, load the labelConds from the alternative path
        else :
            f = open(tmp[0])
            # returns JSON object as a dictionary
            data = json.load(f)
            a = json.loads(data)
            print('Time limits loaded successfully')
            return ((int(a[list(a.keys())[0]][stroke_type]['bottom limit']), int(a[list(a.keys())[0]][stroke_type]['upper limit'])))
            #return ((int(a[list(a.keys())[0]]['bottom limit']), int(a[list(a.keys())[0]]['upper limit'])))


    def get_retinotopic_features(self, FOI, min_lim=90, max_lim = 100, circular_mask_dim = 100, mask_switch = True, adaptive_thresh = True):
        num_for_nan = np.nanmin(FOI)/10
        #num_for_nan = -33e-10
        print(f'Minimum limit {min_lim}, maximum limit {max_lim}')
        blurred = gaussian_filter(np.nan_to_num(FOI, copy=False, nan=num_for_nan, posinf=None, neginf=None), sigma=1)
        _, centroids, blobs = process.detection_blob(blurred, min_lim, max_lim, adaptive_thresh=adaptive_thresh)
        if mask_switch:
            circular_mask = utils.sector_mask(np.shape(blurred), (centroids[0][1], centroids[0][0]), circular_mask_dim, (0,360))
        else:
            circular_mask = None
        return centroids, blobs, circular_mask, blurred


    def centroid_poly(self, X, Y):
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


    def single_seq_retinotopy(self,df_f0, 
                                global_centroid,
                                dim_side,
                                start_frame,
                                end_frame,
                                df_confront = None,
                                df_confront_foi = None,
                                df_f0_foi = None,
                                zero_frames = 20,
                                lim_blob_detect = 80,
                                single_frame_analysis = False,
                                time_window = 1,
                                sig_blank = None,
                                std_blank = None):
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
        if (global_centroid is not None) and \
           (global_centroid[1] + dim_side//2<df_f0.shape[-2]) and\
           (global_centroid[1] - dim_side//2>0) and\
           (global_centroid[0] + dim_side//2<df_f0.shape[-1]) and\
           (global_centroid[0] - dim_side//2>0):
            check_seq =df_f0[:, (global_centroid[1]-(dim_side//2)):(global_centroid[1]+(dim_side//2)), 
                            (global_centroid[0]-(dim_side//2)):(global_centroid[0]+(dim_side//2))]

            # Handling the case in which blank signal is provided or not
            if (sig_blank is None) and (std_blank is None):
                sig_blank = np.nanmean(check_seq[:zero_frames, :, :], axis = 0)
                std_blank = np.nanstd(check_seq[:zero_frames, :, :], axis = 0)/np.sqrt(np.shape(check_seq[:, :, :])[0])# Normalization of standard over all the frames, not only the zero_frames
            else:
                sig_blank = sig_blank[(global_centroid[1]-(dim_side//2)):(global_centroid[1]+(dim_side//2)),
                                      (global_centroid[0]-(dim_side//2)):(global_centroid[0]+(dim_side//2))]
                std_blank = std_blank[(global_centroid[1]-(dim_side//2)):(global_centroid[1]+(dim_side//2)),
                                      (global_centroid[0]-(dim_side//2)):(global_centroid[0]+(dim_side//2))]               
        
            # Check for presence of df to subtract to df_f0: used for single trial analysis in AMstrokes
            if df_confront is not None:
                df_confront = df_confront[:, (global_centroid[1]-(dim_side//2)):(global_centroid[1]+(dim_side//2)), 
                                (global_centroid[0]-(dim_side//2)):(global_centroid[0]+(dim_side//2))]
            flag_adjust_centroid = True
        else:
            check_seq = df_f0
            # Handling the case in which blank signal is provided or not
            if (sig_blank is None) and (std_blank is None):
                sig_blank = np.nanmean(check_seq[:zero_frames, :, :], axis = 0)
                std_blank = np.nanstd(check_seq[:zero_frames, :, :], axis = 0)/np.sqrt(np.shape(check_seq[:, :, :])[0])# Normalization of standard over all the frames, not only the zero_frames        
            flag_adjust_centroid = False
                
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
        
        # Thresholding values
        lim_inf = np.nanpercentile(np.nanmean(ztmp, axis=0), lim_blob_detect)
        lim_sup = np.nanpercentile(np.nanmean(ztmp, axis=0), 100)
        #print(lim_inf, lim_sup)

        # If want to store information from single frame
        if single_frame_analysis:
            single_centroids = list()
            # Strategy for time windowing
            for i in range(len(ztmp)):
                if time_window==1:
                    #print(f'the {i}th frame')
                    tmp_ = ztmp[i, :, :]
                elif time_window >1:
                    if i == 0:
                        #print(f'from 0 to {time_window//2}')
                        tmp_ = np.nanmean(ztmp[i:time_window//2, :, :], axis=0)                    
                    elif i<=time_window//2-1:
                        #print(f'from 0 to {time_window//2}')
                        tmp_ = np.nanmean(ztmp[0:i:time_window//2, :, :], axis=0)

                    elif i>time_window//2-1:
                        try:
                            tmp_ = np.nanmean(ztmp[i-time_window//2:i+time_window//2, :, :], axis=0)
                            #print(f'from {i-time_window//2} to {i+time_window//2}')
                        except:
                            tmp_ = np.nanmean(ztmp[i-time_window//2:, :, :], axis=0)
                            #print(f'from {i-time_window//2} to {len(ztmp)}')
                centroids_singl, _, _, blurred_singl = self.get_retinotopic_features(tmp_, min_lim=lim_blob_detect, max_lim = 100, mask_switch = False)
                coords_singl = np.array(list(zip(*centroids_singl)))
                if (coords_singl is not None) and (len(coords_singl)>0) :
                    # Centroid at maximum response
                    (a,b), _ = centroid_max(coords_singl[0], coords_singl[1], blurred_singl)
                else:
                    print(len(coords_singl))
                    (a,b) = (np.nan, np.nan)
                # Centroid at the centroid of the polygon given by all the points
                #(a,b) = centroid_poly(coords_singl[0], coords_singl[1])
                
                # If global_centroid, then normalization of resulting centroid
                if global_centroid is None:
                    c,d = ((a,b))
                else:
                    c, d = ((global_centroid[0]-dim_side//2 + a, global_centroid[1]-dim_side//2 + b))
                single_centroids.append((c, d))
                
        else:
            single_centroids = []

        centroids, blobs, _, blurred = self.get_retinotopic_features(np.nanmean(ztmp, axis=0), min_lim=lim_inf, max_lim = lim_sup, mask_switch = False, adaptive_thresh=False)
        coords = np.array(list(zip(*centroids)))
        if (coords is not None) and (len(coords)>0) :
            (a,b), _ = centroid_max(coords[0], coords[1], blurred)
        else:
            (a,b) = (np.nan, np.nan)
        # Problematic if: global_centroid could be not None and still not need to adjust the c, d values. TO TEST
        if global_centroid is None or (not flag_adjust_centroid):
            c,d = ((a,b))
        else:
            c, d = ((global_centroid[0]-dim_side//2 + a, global_centroid[1]-dim_side//2 + b))
        return (c, d), blurred, blobs, centroids, (a,b), ztmp, single_centroids

def get_conditions_correspondance(path):
    '''
    Reading metadata json file for conditions positions
    '''
    tmp = utils.find_thing('json_data.json', path)
    # If also with find_thing there is no json file, than return None and a warning#
    if len(tmp) == 0:
        print('Check the json_data.json presence inside the session folder and subfolders')
        return None
    # else, load the json
    else :
        f = open(tmp[0])
        # returns JSON object as a dictionary
        data = json.load(f)
        a = json.loads(data)
        print('Positions condition metadata loaded successfully')
        return a[list(a.keys())[0]]['pos metadata']
        
def get_mask_on_trajectory(dims, xs, ys, radius = 2):
    up = int(np.max(xs))
    bottom = int(np.min(xs))
    x, y = get_trajectory(xs, ys, (bottom, up))
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
 

def get_trajectory(xs, ys, limits):
    print(limits[0], limits[1])
    xs_to_fit =  np.round(np.linspace(limits[0], limits[1]-1, limits[1]-limits[0])) 
    ys_to_fit = np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs_to_fit))
    #ys_fitted = [int(round(np.interp(i, xs_to_fit, ys_to_fit))) for i in xs_fitted]
    return xs_to_fit, ys_to_fit


def centroid_max(X, Y, data):
    '''
    Pick the point in the matrix data with higher value.
    X and Y are list of x and y coordinates.
    The method returns the coordinates and the value of higher point.
    '''
    max_point = -np.inf
    for i, (x, y) in enumerate(zip(X, Y)):
        #print(data[y, x])
        if data[y, x] >= max_point:
            index = (x, y)
            max_point = data[y, x]
        elif i == 0:
            print('Something wrong with the centroid_max method')
    return index, max_point


def get_session_metainfo(path_session):
    experiment_name = path_session.split('exp-')[1].split('sub-')[0][:-1]  
    session_name = path_session.split('sess-')[1][0:12]
    sub_name = path_session.split('sub-')[1].split('sess-')[0][:-1]
    return sub_name, experiment_name, session_name


def get_retinotopy(name_cond, 
                   path_md_files,
                   mask,
                   green,
                   time_limits,
                   window_dimension,
                   mean_blank = None,
                   std_blank = None,
                   tc_window_dimension = 10,
                   zero_frames = 20,
                   retino_features = True,
                   kind_stroke = None ):

    start_time = datetime.datetime.now().replace(microsecond=0)
    
    PATH = path_md_files.split('derivatives')[0]
    #print(PATH)
    TIME_LIMITS_SINGLE = time_limits
    DIM_WINDOW = window_dimension
    DIM_TC_MASK = tc_window_dimension
    v = name_cond

    # Extracting title info
    sub_name, experiment_name, session_name = get_session_metainfo(path_md_files)
    name_exp = sub_name+experiment_name+session_name
    
    # Condition instance
    cd = md.Condition()
    cd.load_cond(os.path.join(path_md_files, 'md_data_'+v))
    print('Condition ' + v + ' loaded!')
    
    # dF/F0 of only autoselected trials 
    df = md.get_selected(cd.df_fz, cd.autoselection)
    # if blank does not provided:
    if (mean_blank is None) and (std_blank is None):
            mean_blank = np.nanmean(cd.averaged_df[:zero_frames, :, :], axis = 0)
            std_blank = np.nanstd(cd.averaged_df[:zero_frames, :, :], axis = 0)/np.sqrt(np.shape(cd.averaged_df)[0])

    z_s = process.zeta_score(cd.averaged_df, mean_blank, std_blank, full_seq = True)
    
    # Instance retinotopy object: single stroke
    single_stroke = Retinotopy(PATH,
                               cond_name = v,
                               name = name_exp + '_cond_' +v, 
                               session_name = name_exp,
                               signal = z_s,
                               mask = mask,
                               green = green,
                               stroke_type = kind_stroke)

    if (TIME_LIMITS_SINGLE is not None) and (kind_stroke is None):
        single_stroke.time_limits = TIME_LIMITS_SINGLE

    if retino_features:

        # Blob and centroids extraction_ averaged 
        _, blurred, blobs, centroids, _, _, _ = single_stroke.single_seq_retinotopy(cd.averaged_df, None, None,
                                                                                    single_stroke.time_limits[0],
                                                                                    single_stroke.time_limits[1])
        
        print(f'Centroids found: {centroids}')

        # Storing values
        single_stroke.blob = blobs
        single_stroke.retino_pos = centroids[0]
        blurred[~mask] = np.NAN
        single_stroke.map = blurred

        # Single trial centroids distribution
        print(f'Shape of signal for single trial extracting centroids: {df.shape}\n')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        print(f'Centroids and dimension of windows: {(single_stroke.retino_pos, DIM_WINDOW)}\n')
        print(single_stroke.retino_pos, DIM_WINDOW, single_stroke.time_limits[0],single_stroke.time_limits[1])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        pos_single_trials_data = [single_stroke.single_seq_retinotopy(i, 
                                                                    single_stroke.retino_pos,
                                                                    DIM_WINDOW, 
                                                                    single_stroke.time_limits[0],
                                                                    single_stroke.time_limits[1],
                                                                    sig_blank = mean_blank,
                                                                    std_blank = std_blank) for i in df]    

        # Storing distribution of points
        pos_centroids = list(list(zip(*pos_single_trials_data))[0])
        single_stroke.distribution_positions = list(zip(*pos_centroids))
        (y_bnnd_size, x_bnnd_size) = df.shape[-2:]
        # Masking for time course extraction
        single_stroke.tc_mask = utils.sector_mask((y_bnnd_size, x_bnnd_size),
                                                    (single_stroke.retino_pos[1], single_stroke.retino_pos[0]),
                                                    DIM_TC_MASK, 
                                                    (0,360))

        # Time courses making
        single_stroke.time_courses = np.array([process.time_course_signal(np.nan_to_num(w, copy = False, nan=0.0000001, posinf=None, neginf=None), abs(single_stroke.tc_mask - 1)) for w in df])
        single_stroke.average_time_course = np.nanmean(single_stroke.time_courses, axis = 0)
    
    single_stroke.df_fz = df
    print('Condition ' +v + ' elaborated in '+ str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!')

    return single_stroke

def single_trial_detection(retino_object, dim_window, time_window_inference, df_conf, time_limits_first, time_limits_second, id_name, sub = 'No Wallace or Bretzel'):
    
    #if (sub == 'Bretzel') or (sub == 'Wallace'):
    if (sub == 'Bretzel') or (sub == 'Wallace') or (sub=='Ziggy'):
        dim_window = None
        centroids_ = None
    else:
        centroids_ = retino_object.retino_pos

    print(f'Shape of signal for single trial extracting centroids: {retino_object.df_fz.shape}\n')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    print(f'Centroids and dimension of windows: {(centroids_, dim_window)}\n')
    print(centroids_, dim_window, time_window_inference[0], time_window_inference[1], time_limits_second, time_limits_first)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    pos_single_trials_data = [retino_object.single_seq_retinotopy(i, 
                                                                  centroids_,
                                                                  dim_window, 
                                                                  time_window_inference[0],
                                                                  time_window_inference[1],
                                                                  df_confront = df_conf,
                                                                  df_confront_foi = time_limits_second,
                                                                  df_f0_foi = time_limits_first,
                                                                  single_frame_analysis=True,
                                                                  time_window=3) for i in retino_object.df_fz]    
    
    # Storing distribution of points
    pos_centroids = list(list(zip(*pos_single_trials_data))[0])
    retino_object.distribution_positions = list(zip(*pos_centroids))
    print('Centroids found!\n')               
    #print('The dots for single trial analysis are:\n')
    #print(single_stroke.distribution_positions)
    # blurrs_pos_single_trials = list(list(zip(*pos_single_trials_data))[1])
    # blobs_pos_single_trials = list(list(zip(*pos_single_trials_data))[2])
    # #pos_centroids_single_trials = list(list(zip(*pos_single_trials_data))[3])
    # not_normlzd_pos_centroids = list(list(zip(*pos_single_trials_data))[4])
    #z_scores_pos_single_trials = list(list(zip(*pos_single_trials_data))[5])
    #centroids_pos_single_frame = list(list(zip(*pos_single_trials_data))[6])
                    
    return retino_object

def subtraction_among_conditions(path_session, 
                                 first, second, 
                                 time_limits_first, 
                                 time_limits_second, 
                                 id_name, 
                                 name,
                                 session_name, 
                                 mask, 
                                 traject_mask, 
                                 time_courses, 
                                 green, 
                                 df, 
                                 params, 
                                 name_params, 
                                 time_window_inference, 
                                 multiple_stroke_123,
                                 stroke_type = 'multiple stroke', sub = 'No Wallace or Bretzel', single_trial_analysis = True, dim_window = 50):
    
    #First
    _, _, _, _, _, z_123_shrinked, _ = multiple_stroke_123.single_seq_retinotopy(first, None, None, time_limits_first[0], time_limits_first[1])
    sign = z_123_shrinked

    if (second is not None) and (time_limits_second is not None):
        #AM12
        _, _, _, _, _, z_12_shrinked, _ = multiple_stroke_123.single_seq_retinotopy(second, None, None, time_limits_second[0], time_limits_second[1])
        sign = z_123_shrinked-z_12_shrinked

    # AM123 - AM12
    pos_inferred_averaged = Retinotopy(path_session, 
                                        cond_name = id_name + name, 
                                        name = id_name + name,
                                        signal = sign,
                                        averaged_simple_retino_pos = None, 
                                        session_name = session_name, 
                                        distribution_centroids = list(),
                                        blob = None, 
                                        mask = mask,
                                        maps = None,
                                        mask_tc = traject_mask,
                                        tc = time_courses,
                                        averaged_tc = np.nanmean(time_courses, axis=0),
                                        df = df,
                                        green = green, 
                                        stroke_type = stroke_type)

    pos_inferred_averaged.time_limits = ((time_limits_first[0], time_limits_first[1]))
    if (sub != 'Wallace' ) and (sub !='Bretzel') and (sub!='Ziggy'):
        FOI = np.mean(pos_inferred_averaged.signal, axis=0)*pos_inferred_averaged.mask
    else:
        FOI = np.mean(pos_inferred_averaged.signal, axis=0)

    # Find retinotopic position in averaged signal over 15 frames
    centroids, blobs, _, blurred = pos_inferred_averaged.get_retinotopic_features(FOI, mask_switch = False)
    min_bord = np.percentile(blurred, 15)
    max_bord = np.percentile(blurred, 98)
    #coords_singl = np.array(list(zip(*centroids)))
    #(a,b), _ = retino.centroid_max(coords_singl[0], coords_singl[1], blurred)
    pos_inferred_averaged.retino_pos = centroids[0]
    pos_inferred_averaged.blob = blobs
    
    #if (sub != 'Wallace' ) and (sub !='Bretzel'):
    if (sub != 'Wallace' ) and (sub !='Bretzel') and (sub!='Ziggy'):
        blurred[~pos_inferred_averaged.mask] = np.NAN
    pos_inferred_averaged.map = blurred
    
    if single_trial_analysis:
    #        if (second is None) or (time_limits_second is None):
            #retino_object, pos_centroids, blurrs_pos_single_trials, blobs_pos_single_trials, pos_centroids_single_trials, not_normlzd_pos_centroids, z_scores_pos_single_trials, centroids_pos_single_frame 
            #print(dim_window, time_window_inference, second, pos_inferred_averaged.time_limits, time_limits_second,id_name, sub)
    #            time_limits_first = None
    #        else:
        pos_inferred_averaged = single_trial_detection(pos_inferred_averaged, dim_window, time_window_inference, second, pos_inferred_averaged.time_limits, time_limits_second, id_name, sub = sub)
    else:
        pos_inferred_averaged.distribution_positions = list()
    
    # Storing parameters
    params[name_params].append((min_bord, max_bord)) #heatmaps limits
    params[name_params].append(pos_inferred_averaged.blob) #blob contours
    params[name_params].append(centroids) #averaged retinotopic position
    params[name_params].append(pos_inferred_averaged.map) #averaged zscore
    params[name_params].append(pos_inferred_averaged.distribution_positions)#pos3_inferred_averaged.distribution_positions) #single trial centroids distribution
    params[name_params].append(np.arange(pos_inferred_averaged.time_limits[0]-7,pos_inferred_averaged.time_limits[1]+7,1).astype(int)) #xlimits for timecourse plot
    params[name_params].append(pos_inferred_averaged.tc_mask) #mask
    params[name_params].append(pos_inferred_averaged.time_courses) #average timecourse
    params[name_params].append(pos_inferred_averaged.average_time_course) #average timecourse 8

    # Build the 'retino_inferred' folder for the session 
    # storage_path = os.path.join(NAME_ANALYSIS, id_name, 'retino_inferred')
    # tmp = dv.set_storage_folder(name_analysis = os.path.join(storage_path,))
    # # Storing retinotopic objects
    # # Deleting df_fz and time courses for sparing storage volume
    # pos_inferred_averaged.df_fz = None
    # pos_inferred_averaged.time_courses = None
    # pos_inferred_averaged.store_retino(os.path.join(dv.STORAGE_PATH, NAME_ANALYSIS, tmp))

    return params, pos_inferred_averaged 


def set_retinotopy_session(path_md, green_name, single_stroke_label, conditions_id=None):
    # Extract path session from md data folder path
    path_session = path_md.split('derivatives')[0]

    # Extract metainformation related to condition positions
    #RETINO_POS_AM 
    retino_pos_am = get_conditions_correspondance(path_session)

    # Useful variable extraction from session object
    session = md.Session(path_session, logs_switch = False, deblank_switch = False)
    
    #conds = list(session.cond_dict.keys())
    path_md_files = os.path.join(path_md,'md_data')
    
    # Session names extraction
    sub_name, experiment_name, session_name = get_session_metainfo(path_session)
    id_name = sub_name + experiment_name + session_name

    # Blank condition loading
    cd_blank = md.Condition()
    cd_blank.load_cond(os.path.join(path_md_files, 'md_data_blank'))
    mean_blank = np.nanmean(cd_blank.averaged_df[:20, :,:], axis=0)
    std_blank = np.nanstd(cd_blank.averaged_df[:20, :,:], axis=0)/np.sqrt(np.shape(cd_blank.averaged_df)[0])
    
    # Loading green
    try:
        green_path = utils.find_thing(green_name, path_session)
        green = cv.imread(green_path[0], cv.IMREAD_UNCHANGED)

        # Resizing green
        (y_size, x_size) = cd_blank.averaged_df[0, :,:].shape
        x_bnnd_size = x_size
        y_bnnd_size = y_size
        tmp = cv.resize(np.array(green, dtype='float64'), (x_bnnd_size, y_bnnd_size), interpolation=cv.INTER_LINEAR)
        green_ = np.copy(tmp)
    except:
        print('No green '+ green_name+ ' present in rawdata folder for session ' + id_name)    

    # Loading handmade mask
    try:
        mask = np.load(os.path.join(path_session, 'derivatives','handmade_mask.npy'))
        mask = mask[0:y_bnnd_size, 0:x_bnnd_size]

    except:
        print('No mask present in derivatives folder for session ' + id_name)
        mask = None

    # Two dictionaries, for type of conditions -pos or am-
    single_pos_conds = {k: v for k,v in session.cond_dict.items() if single_stroke_label.lower() in v.lower()}
    am_conds = {k: v for k,v in session.cond_dict.items() if (single_stroke_label.lower() not in v.lower()) and (v.lower() != 'blank')}

    # Start from the single stroke conditions for storing and afterward showing the positions in AM conditions
    conds_full = {**single_pos_conds, **am_conds}

    # Intersect the set of all the conditions with the picked one in the parser
    if conditions_id is not None:
        # Manual insert of condition id by key number
        conds = {k: v for k,v in conds_full.items() if k in conditions_id}
        # Taking the picked condition names
        conds_names = list(conds.values())
        # Taking the am conditions ONLY
        am_tmp = list(set(conds_names).intersection(set(am_conds.values())))
        # Taking the single stroke conditions that make the AM
        cond_t_list =  [retino_pos_am[v] for v in am_tmp]
        # Considering a sum of single stroke that make the picked AMs and unifying them to the one immediately picked -w/o repetition- 
        all_considered_conds = list(set(list(conds.values()) + cond_t_list))
        # Rebuild dictionary with id as key and condition name as value
        conds = {k: v for k,v in conds_full.items() if v in all_considered_conds}

    else:
        conds = conds_full
    
    print(conds)

    return conds, green_, mask, mean_blank, std_blank, id_name, path_md_files, retino_pos_am
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Launching retinotopy analysis pipeline')

    parser.add_argument('--path_md', 
                        dest='path_md',
                        type=str,
                        required=True,
                        help='The middle process datafolder path')

    parser.add_argument('--green_name', 
                        dest='green_name',
                        type=str,
                        default = 'green01.bmp',
                        required=False)  

    parser.add_argument('--ss_label', 
                        dest='single_stroke_label',
                        type=str,
                        default = 'pos',
                        required=False)  

    parser.add_argument('--am_label', 
                        dest='apparent_motion_label',
                        type=str,
                        default = 'am',
                        required=False)  
    parser.add_argument('--cid', 
                        action='append', 
                        dest='conditions_id',
                        default=None,
                        type=int,
                        help='Conditions to analyze: None by default -all the conditions-')   

    start_process_time = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args()

    print(args)

    # Store time boundaries
    #time_limits_single = ((args.bottom_time_window, args.upper_time_window))

    # Instantiate variables
    global_centroid, masks, blobs_pos = list(), list(), list()
    tcs_single_pos, tcs_single_pos_avrg, names_cd = list(), list(), list()
    tcs_ams, tcs_ams_avrg, names_cd_ams= list(), list(), list()
    centroids_singlepos, tc_masks = list(), list()

    retino_pos_ = dict()    

    # Instance of the retinotopy session
    path_session = args.path_md.split('derivatives')[0]

    retino_session = RetinoSession(path_session, 
                                   args.path_md, 
                                   args.green_name, 
                                   conditions_id=args.conditions_id, 
                                   single_stroke_label=args.single_stroke_label, 
                                   multiple_stroke_label=args.apparent_motion_label) 
    
    # conds, green_, mask, mean_blank, std_blank, ID_NAME, path_md_files, RETINO_POS_AM = set_retinotopy_session(args.path_md, args.green_name, args.single_stroke_label, args.conditions_id)


    # for k,v in conds.items():
    #     print(v + '\n')
    #     start_cond_time = datetime.datetime.now().replace(microsecond=0)
        
    #     # Creation/Check of existence data folder for filtered data
    #     tmp_filt = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis =  os.path.join(args.name_analysis, ID_NAME, v, 'filtered'))

    #     # Switch for the retino object
    #     switch_cond = False
    #     if args.single_stroke_label in v:
    #         switch_cond = True
    #         k_s = 'single stroke'
    #     else:
    #         k_s = 'multiple stroke'

    #     # Instance retino object   
    #     retino_obj = get_retinotopy(v,
    #                                 path_md_files,
    #                                 mask,
    #                                 green_,
    #                                 time_limits_single,
    #                                 args.dim_crop_window,
    #                                 mean_blank = mean_blank,
    #                                 std_blank = std_blank,
    #                                 tc_window_dimension = args.dim_tc_wind,
    #                                 zero_frames = args.zero_frames,
    #                                 retino_features = switch_cond,
    #                                 kind_stroke = k_s)
    #     maps = retino_obj.signal
    #     print(f'Time limits for retinotopy detection: {retino_obj.time_limits}')

    #     if args.retino_extraction:
    #         retino_pos_[v] = retino_obj.retino_pos
    #         # Colorcoding for retinotopic positions
    #         if switch_cond:
    #             print(retino_pos_)
    #             indeces_colors = [y for y, kj in enumerate(retino_pos_.keys()) if kj==v][0]
    #             colrs = [dv.COLORS_7[indeces_colors]]
    #             print(colrs)
    #             g_centers = [retino_obj.retino_pos]
    #         else:
    #             indeces_colors =[list(retino_pos_.keys()).index(i) for i in RETINO_POS_AM[v]]
    #             colrs =  [dv.COLORS_7[i] for i in indeces_colors]        
    #             g_centers = [retino_pos_[i] for i in RETINO_POS_AM[v]]
    #     else:
    #         colrs, g_centers = [], []

    #     if args.data_vis_switch:
    #         dv.whole_time_sequence(maps, 
    #                                 mask = mask,
    #                                 name='z_sequence_'+ v + ID_NAME, 
    #                                 max=80, min=20, 
    #                                 global_cntrds = g_centers,
    #                                 colors_centr = colrs,
    #                                 name_analysis_= os.path.join(args.name_analysis, ID_NAME, v))


    #     if args.data_vis_switch and args.retino_extraction:
    #         if switch_cond:
    #             # Plotting retinotopic positions over averaged maps
    #             min_bord = np.nanpercentile(retino_obj.map, 15)
    #             max_bord = np.nanpercentile(retino_obj.map, 98)

    #             fig, ax = plt.subplots(1,1, figsize=(9,7), dpi=300)
    #             ax.contour(retino_obj.blob, 4, colors='k', linestyles = 'dotted')
    #             pc = ax.pcolormesh(retino_obj.map, vmin=min_bord,vmax=max_bord, cmap=utils.PARULA_MAP)
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             fig.colorbar(pc, shrink=1, ax=ax)
    #             ax.scatter(retino_obj.retino_pos[0],retino_obj.retino_pos[1],color='r', marker = '+', s=150)
    #             ax.scatter(retino_obj.distribution_positions[0],retino_obj.distribution_positions[1], color=dv.COLORS_7[list(retino_pos_.keys()).index(v)], marker = '.', s=150)
    #             ax.vlines(retino_obj.retino_pos[0], 0, retino_obj.map.shape[0], color = dv.COLORS_7[list(retino_pos_.keys()).index(v)], lw= 3, ls='--', alpha=1)
    #             ax.set_title(ID_NAME + ' condition: ' + v )

    #             # Storing picture
    #             tmp = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis = os.path.join(args.name_analysis, ID_NAME, v))
    #             plt.savefig(os.path.join(tmp, 'averagedheatmap_' +v+ '.svg'))
    #             print('averagedheatmap_' +v+ '.svg'+ ' stored successfully!')
    #             plt.savefig(os.path.join(tmp, 'averagedheatmap_' +v+ '.png'))
    #             plt.close('all')

    #             # Variables for plotting timecourses and averaged heatmap
    #             tcs_single_pos.append(retino_obj.time_courses)
    #             tcs_single_pos_avrg.append(retino_obj.average_time_course)
    #             names_cd.append(v)
    #             centroids_singlepos.append([retino_obj.retino_pos])
    #             tc_masks.append(retino_obj.tc_mask)

    #         elif (not switch_cond) and args.retino_extraction:
    #             # Time courses for AM conds 
    #             xs = list(list(zip(*g_centers))[0])
    #             ys = list(list(zip(*g_centers))[1])
    #             a, b = get_trajectory(xs, ys, (75, retino_obj.signal.shape[-1] - 75))
    #             small_mask = get_mask_on_trajectory((retino_obj.signal.shape[-2], retino_obj.signal.shape[-1]), a, b, radius = 15)
                
    #             # Time courses making
    #             retino_obj.time_courses = np.array([process.time_course_signal(np.nan_to_num(w, copy = False, nan=0.0000001, posinf=None, neginf=None), abs(small_mask - 1)) for w in retino_obj.df_fz])
    #             retino_obj.average_time_course = np.nanmean(retino_obj.time_courses, axis = 0)
                
    #             # Variables for plotting timecourses and averaged heatmap
    #             tcs_ams.append(retino_obj.time_courses)
    #             tcs_ams_avrg.append(retino_obj.average_time_course)
    #             names_cd_ams.append(v)
    #             #app_ = v.split('AM_')[1]
    #             #names_cd_ams.append(app_.replace('nbStrokes', 'AM'))
        
    #         elif (not switch_cond):# and (not args.retino_extraction):
    #             circ_mask = blk_file.circular_mask_roi(np.shape(mean_blank)[-1], np.shape(mean_blank)[-2])
    #             retino_obj.time_courses = np.array([process.time_course_signal(i, circ_mask) for i in retino_obj.df_fz])
    #             retino_obj.average_time_course = np.nanmean(retino_obj.time_courses, axis = 0)

    #     #check_presence = utils.find_thing('filtered_df_' + v + '.npy', os.path.join(dv.STORAGE_PATH, ''))
    


        
