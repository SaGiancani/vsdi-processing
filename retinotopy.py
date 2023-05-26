import argparse, blk_file, datetime, json, os, utils
import cv2 as cv
import data_visualization as dv
import matplotlib.pyplot as plt
import middle_process as md
import numpy as np
import process_vsdi as process

from scipy.ndimage.filters import gaussian_filter

COLORS_STROKE_WITHIN_AM = ['turquoise', 'teal']

NAME_RETINO_ANALYSIS = 'AMnbStrokes_Retinotopic_Analysis'

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
                     time_course_window_dim = 10,
                     window_dim = 150,
                     acquisition_fq = 100,#Hz
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
            # All blks names loaded
            self.all_blks = md.get_all_blks(self.header['path_session'], sort = True) # all the blks, sorted by creation date -written on the filename-.
            # A blk loaded for useful hyperparameters
            blk = blk_file.BlkFile(os.path.join(self.header['path_session'],'rawdata', self.all_blks[np.random.randint(len(self.all_blks)-1)]), 
                                self.header['spatial_bin'], 
                                self.header['temporal_bin'])
            self.header['n_frames'] = blk.header['nframesperstim']
            self.header['original_height'] = blk.header['frameheight']
            self.header['original_width'] = blk.header['framewidth']
            # Setting key frames
            # End
            if end_frame is None:
                self.header['ending_frame'] = int(round(self.header['n_frames']*0.9))
            else:
                self.header['ending_frame'] = end_frame
            # Start
            if zero_frames is None:
                self.header['zero_frames'] = int(round(self.header['n_frames']*0.2))
            else:
                self.header['zero_frames'] = zero_frames
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
            # Separated dictionaries, for AM and single pos conditions
            self.cond_pos = self.get_conditions_pos()
            self.cond_am = self.get_conditions_am()
            # Pick only inserted conditions and corresponding single positions
            self.cond_dict = self.get_conditions_intersect()
            # Name condition extraction
            self.cond_names = list(self.cond_dict.values())
            print(str(self.cond_dict) + '\n')
            print(f'Only picked conditions: {self.cond_dict}\n')
            print(f'All session conditions: {self.cond_dict_all}\n')
            self.acquisition_frequency = acquisition_fq

            # Metadata stimulus
            self.stimulus_metadata = get_stimulus_metadata(self.path_session) 

            # Blank condition loading
            # TO NOTICE: deblank_switch add roi_signals, df_fz, auto_selected, conditions, counter_blank and overwrites the session_blks
            self.time_course_blank = None
            self.f_f0_blank = None
            self.stde_f_f0_blank = None           
            self.blank_condition = None            
            self.get_blank()
            self.mean_blank = self.blank_condition.averaged_df
            self.std_blank = np.nanstd(self.mean_blank, axis=0)/np.sqrt(np.shape(self.mean_blank)[0])

            self.id_name = self.get_session_id_name()
            print('Session ID name: ' + self.id_name)
            self.green = self.get_green(green_name)
            self.mask = self.get_mask()

            # Single centroid mask dimension
            self.tc_window_dimension =  time_course_window_dim
            self.window_dimension = window_dim

            self.visualization_switch = data_vis_switch
            self.storage_switch = store_switch
    
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
            am_conds = self.cond_am
            single_conds = self.cond_pos
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
                (y_size, x_size) = self.blank_condition.averaged_df[0, :,:].shape
                x_bnnd_size = x_size
                y_bnnd_size = y_size
                mask = mask[0:y_bnnd_size , 0:x_bnnd_size ]
                print('Mask loaded succesfully!')

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
                (y_size, x_size) = self.blank_condition.averaged_df[0, :,:].shape
                x_bnnd_size = x_size
                y_bnnd_size = y_size

                tmp = cv.resize(np.array(green, dtype='float64'), (x_bnnd_size, y_bnnd_size), interpolation=cv.INTER_LINEAR)
                green_ = np.copy(tmp)
                print('Green '+green_name+' loaded succesfully!')
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
                print(os.path.join(path_md_files, 'md_data_blank'))
                cd_blank.load_cond(os.path.join(path_md_files, 'md_data_blank'))
                self.blank_condition = cd_blank
                print('Blank condition loaded succesfully!\n')
                #mean_blank = np.nanmean(cd_blank.averaged_df[:20, :,:], axis=0)
                #std_blank = np.nanstd(cd_blank.averaged_df[:20, :,:], axis=0)/np.sqrt(np.shape(cd_blank.averaged_df)[0]) 
            except:
                print('Blank condition not found in ' + path_md_files+ '\n')
                # In case of absence of the blank condition, it processes and stores it
                self.storage_switch = True
                self.visualization_switch = False
                # It is gonna get the blank signal automatically
                print('Processing blank signal\n')
                _ = self.get_signal(self.blank_id)
                self.storage_switch = False
                cd_blank.load_cond(os.path.join(path_md_files, 'md_data_blank'))
                self.blank_condition = cd_blank
                print('Blank condition loaded succesfully!\n')
            return 
        

        def get_retinotopy(self,
                           name_cond, 
                           time_limits, 
                           retinotopic_path_folder, 
                           dict_retino):
            print('Start processing retinotopy analysis for condition ' + name_cond )
            start_time = datetime.datetime.now().replace(microsecond=0)

            # Condition instance
            cd = md.Condition()
            # Loading or building the condition
            try:
                cd.load_cond(os.path.join(self.path_md, 'md_data','md_data_'+name_cond))
                print('Condition ' + name_cond + ' loaded!\n')

            except:
                print('Condition ' + name_cond + ' not found\n')
                self.storage_switch = True
                self.visualization_switch = False
                # It is gonna get the blank signal automatically
                print('Processing ' + name_cond + ' signal\n')
                id_cond = [k for k, v in self.cond_dict_all.items() if v == name_cond][0]
                _ = self.get_signal(id_cond)
                self.storage_switch = False
                cd.load_cond(os.path.join(self.path_md, 'md_data','md_data_'+name_cond))
                print('Condition ' + name_cond + ' loaded!\n')
            colrs = []
            # Single stroke condition
            if name_cond in list(self.cond_pos.values()):
                # Try to check if retino_cond already exists
                try:
                    retino_cond = Retinotopy(self.path_session)
                    retino_cond.load_retino(os.path.join(retinotopic_path_folder, self.id_name, name_cond, 'retino'))                    
                # If does not, it build it
                except:
                    retino_cond = self.get_stroke_retinotopy(name_cond, time_limits, cd, stroke_number = None, str_type = 'single stroke')
                    # Store single stroke condition
                    dict_retino[name_cond] = retino_cond
                    # Extract visualization utility variables
                    indeces_colors = [list(self.cond_pos.values()).index(name_cond)][0]
                    colrs.append(dv.COLORS_7[indeces_colors])
    #                g_centers = [retino_cond.retino_pos]
                    # If true, store pictures
                    if self.visualization_switch:
                        self.plot_stuff(retinotopic_path_folder, name_cond, colrs, dict_retino)
                    # If true store variables
                    if self.storage_switch:
                        retino_cond.store_retino(os.path.join(retinotopic_path_folder, self.id_name, name_cond))
            
            # Multiple stroke condition
            elif name_cond in list(self.cond_am.values()):
                # Storing variable
                dict_retino[name_cond] = dict()
                for i, j in enumerate(self.retino_pos_am[name_cond]):
                    print('The stroke ' +j+f' is the number {i}')
                    retino_cond = self.get_stroke_retinotopy(name_cond, time_limits, cd, stroke_number = i, str_type = 'multiple stroke')
                    # Store single stroke within AM
                    dict_retino[name_cond][j] = retino_cond
                    # Extract visualization utility variables
                    #indeces_colors =[list(dict_retino.keys()).index(k) for k in self.retino_pos_am[name_cond]]
                    #g_centers = [dict_retino[k] for k in self.retino_pos_am[name_cond]]
                    indeces_colors =[list(self.cond_pos.values()).index(j)][0]
                    colrs.append(dv.COLORS_7[indeces_colors])
#                    g_centers.append(dict_retino[j].retino_pos)
                    # If true store variables
                    if self.storage_switch:
                        retino_cond.store_retino(os.path.join(retinotopic_path_folder, self.id_name, name_cond, name_cond +'-'+j))
                # If true, store pictures
                if self.visualization_switch:
                    self.plot_stuff(retinotopic_path_folder, name_cond, colrs, dict_retino)
            print('End processing retinotopy analysis for condition ' + name_cond )
            print('Condition ' +name_cond + ' elaborated in '+ str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!\n')            
            return dict_retino
        

        def get_retino_session(self):
            start_time = datetime.datetime.now().replace(microsecond=0)
            # Create Retinotopic Analysis folder path
            retinotopic_path_folder = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis = os.path.join(NAME_RETINO_ANALYSIS))
            print('Retino session for data session ' + self.id_name + ' start to process...\n')                                            
            # Storing variable
            dict_retino = dict()
            for cond_id, cond_name in self.cond_dict.items():
                dict_retino = self.get_retinotopy(cond_name, None, retinotopic_path_folder, dict_retino)
            print('Retino session elaborated in '+ str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!\n')                                
            return


        def get_stroke_retinotopy(self,
                                  name_cond,
                                  time_limits, 
                                  cd,
                                  stroke_number = None,
                                  str_type = 'single stroke'):

            start_time = datetime.datetime.now().replace(microsecond=0)

            if str_type == 'multiple stroke':
                a = self.stimulus_metadata['pos metadata']
                space_step = a[name_cond]['inter stimulus space']
                starting_time = a[name_cond]['start'] #In frames
                time_step = np.ceil((1/self.stimulus_metadata['speed'])*space_step*self.acquisition_frequency)
                time_step = int(time_step) # In frames                  
                print(f'The interstimulus space is {space_step}, for a starting time of {starting_time}\n')                                     
                print(f'Frame step between the appearance of one stroke and the other: {time_step}')  

            # dF/F0 of only autoselected trials 
            df = md.get_selected(cd.df_fz, cd.autoselection)
            avr_df = np.mean(df, axis = 0)

            # COUNTERCHECK THIS BLANK 
            mean_blank = np.nanmean(self.mean_blank, axis = 0)
            z_s = process.zeta_score(avr_df, mean_blank, self.std_blank, full_seq = True)

            # Instance retinotopy object: single stroke
            r = Retinotopy(self.path_session,
                           cond_name = name_cond,
                           name = self.id_name + '_cond_' +name_cond, 
                           session_name = self.id_name,
                           signal = z_s,
                           mask = self.mask,
                           green = self.green,
                           stroke_type = str_type)

            if (time_limits is not None):
                r.time_limits = time_limits                                 

            #z_s = process.zeta_score(cd_pos3.averaged_df, None, None, full_seq = True)
            # Blob and centroids extraction
            if str_type == 'multiple stroke':
                begin_time = r.time_limits[0]+ starting_time+stroke_number*time_step # stimulus onset time  + actual onset w/o grey frames + number of the stroke*time of occurrence of the stroke
                end_time = r.time_limits[0]+ starting_time+stroke_number*time_step+time_step # stimulus onset time  + actual onset w/o grey frames + number of the stroke*inter stimulus time + end time appearance of the stroke
                foi = ((0, time_step))
            else:
                begin_time = r.time_limits[0]
                end_time = r.time_limits[1]
                foi = None

            print(f'Begin and end frames are: {(begin_time, end_time)}')

            _, blurred, blobs, centroids, norm_centroids, z_s, _ = r.single_seq_retinotopy(avr_df, 
                                                                                           None, None,
                                                                                           begin_time,
                                                                                           end_time,
                                                                                           sig_blank = mean_blank,
                                                                                           std_blank = self.std_blank,
                                                                                           lim_blob_detect = 70)

            r.blob = blobs
            r.retino_pos = centroids[0]
            print(f'Retinotopic averaged position at: {r.retino_pos}')
            #min_bord = np.nanpercentile(blurred, 15)
            #max_bord = np.nanpercentile(blurred, 98)
            blurred[~r.mask] = np.NAN
            r.map = blurred

            #dv.whole_time_sequence(z_s, mask = multiple_stroke.mask, max=85, min=15, n_columns=3, global_cntrds = [multiple_stroke.retino_pos], colors_centr = ['magenta'])


            print('Condition ' +name_cond + ' elaborated in '+ str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!')


            print(f'Shape of signal for single trial extracting centroids: {df.shape}\n')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            print(f'Centroids and dimension of windows: {(r.retino_pos, self.window_dimension)}\n')
            print(r.retino_pos, self.window_dimension, begin_time, end_time)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            pos_single_trials_data = [r.single_seq_retinotopy(i, 
                                                              r.retino_pos,
                                                              self.window_dimension, 
                                                              begin_time,
                                                              end_time,
                                                              df_f0_foi = foi,
                                                              sig_blank = mean_blank,
                                                              std_blank = self.std_blank,
                                                              lim_blob_detect = 70) for i in df] 

            # Storing distribution of points
            pos_centroids = list(list(zip(*pos_single_trials_data))[0])
            r.distribution_positions = list(zip(*pos_centroids))
            return r

        def plot_stuff(self, retinotopic_path_folder, name_cond, colrs, dict_retino):
            if name_cond in list(self.cond_pos.values()):
                col_distr = colrs
                dv.whole_time_sequence(dict_retino[name_cond].signal, 
                                    mask = dict_retino[name_cond].mask,
                                    name='z_sequence_'+ name_cond + self.id_name, 
                                    max=80, min=20,
                                    handle_lims_blobs = ((97.72, 100)),
                                    #significant_thresh = np.percentile(dict_retino[name_cond].signal, 97.72), 
                                    global_cntrds = [dict_retino[name_cond].retino_pos],
                                    colors_centr = colrs,
                                    name_analysis_= os.path.join(retinotopic_path_folder, self.id_name, name_cond))
                # Parameters for heatmap plotting
                min_bord = np.nanpercentile(dict_retino[name_cond].map, 15)
                max_bord = np.nanpercentile(dict_retino[name_cond].map, 98)
                # Averaged hetmap plot
                dv.plot_averaged_map(name_cond, dict_retino[name_cond], dict_retino[name_cond].map, dict_retino[name_cond].retino_pos, min_bord, max_bord, colrs, self.id_name, colrs, name_analysis_ = os.path.join(self.id_name, name_cond, 'RetinotopicPositions'), store_path = retinotopic_path_folder)
            elif name_cond in list(self.cond_am.values()):
                if len(list(self.retino_pos_am[name_cond])) <3:
                    col_distr = COLORS_STROKE_WITHIN_AM[0]
                else:
                    col_distr = COLORS_STROKE_WITHIN_AM[1]
                for c, name_pos in enumerate(list(self.retino_pos_am[name_cond])):
                    # Parameters for heatmap plotting
                    min_bord = np.nanpercentile(dict_retino[name_cond][name_pos].map, 15)
                    max_bord = np.nanpercentile(dict_retino[name_cond][name_pos].map, 98)
                    # Averaged hetmap plot
                    dv.plot_averaged_map(name_cond+name_pos, dict_retino[name_cond][name_pos], dict_retino[name_cond][name_pos].map, dict_retino[name_pos].retino_pos, min_bord, max_bord, [colrs[c]], self.id_name, col_distr, name_analysis_ = os.path.join(self.id_name, name_cond, 'RetinotopicPositions'), store_path = retinotopic_path_folder)
                # Zscore
                dv.whole_time_sequence(dict_retino[name_cond][name_pos].signal, 
                                       mask = dict_retino[name_cond][name_pos].mask,
                                       name='z_sequence_'+ name_cond + self.id_name, 
                                       max=80, min=20,
                                       handle_lims_blobs = ((97.72, 100)),
                                       #significant_thresh = np.percentile(dict_retino[name_cond][name_pos].signal, 97.72), 
                                       global_cntrds = [dict_retino[name_pos].retino_pos for name_pos in list(dict_retino[name_cond].keys())],
                                       colors_centr = colrs,
                                       name_analysis_= os.path.join(retinotopic_path_folder, self.id_name, name_cond))
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


    def get_retinotopic_features(self, FOI, min_lim=90, max_lim = 100, circular_mask_dim = 100, mask_switch = True, adaptive_thresh = True, thresh_gaus = 97.72):# MODIFIED HERE 22/03/23
        num_for_nan = np.nanmin(FOI)/10
        #num_for_nan = -33e-10
        print(f'Minimum limit {min_lim}, maximum limit {max_lim}')
        #averaged_df1 = np.nan_to_num(averaged_df1, nan=np.nanmin(averaged_df1), neginf=np.nanmin(averaged_df1[np.where(averaged_df1 != -np.inf)]), posinf=np.nanmax(averaged_df1[np.where(averaged_df1 != np.inf)]))
        blurred = gaussian_filter(np.nan_to_num(FOI, copy=False, nan=num_for_nan, posinf=None, neginf=None), sigma=1)
        _, centroids, blobs = process.detection_blob(blurred, min_lim, max_lim, min_2_lim = thresh_gaus, adaptive_thresh=adaptive_thresh)
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
                                std_blank = None,
                                single_frame_thresh = 97,
                                all_frame_thres = 97):
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

        # Full frame analysis, no crop
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
        mean_ztmp = np.nanmean(ztmp, axis=0)
        lim_inf = np.nanpercentile(mean_ztmp[np.where((mean_ztmp != -np.inf) | (mean_ztmp != np.inf))], lim_blob_detect)
        lim_sup = np.nanpercentile(mean_ztmp[np.where((mean_ztmp != -np.inf) | (mean_ztmp != np.inf))], 100)
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
                centroids_singl, _, _, blurred_singl = self.get_retinotopic_features(tmp_, min_lim=lim_blob_detect, max_lim = 100, mask_switch = False, thresh_gaus=single_frame_thresh)
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

        centroids, blobs, _, blurred = self.get_retinotopic_features(np.nanmean(ztmp, axis=0), min_lim=lim_inf, max_lim = lim_sup, mask_switch = False, adaptive_thresh=False, thresh_gaus=all_frame_thres)
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

def get_assess_centroid(centroids, mask):
    '''
    Assess position of the centroids: if within the mask, then it is considered
    '''
    return [i for i in centroids if mask[i[1],i[0]]]


def get_stimulus_metadata(path):
    '''
    Reading stimulus metadata json file
    '''
    tmp = utils.find_thing('json_data.json', path)
    if len(tmp) == 0:
        print('Check the json_data.json presence inside the session folder and subfolders')
        return None
    # else, load the json
    else :
        f = open(tmp[0])
        # returns JSON object as a dictionary
        data = json.load(f)
        a = json.loads(data)
    return  a[list(a.keys())[0]]

def get_conditions_correspondance(path):
    '''
    Reading metadata json file for conditions positions
    '''
    a = get_stimulus_metadata(path)
    # Build a dictionary with am conditions as keys and corresponding single stroke position as lists
    return {i: j['conditions'] for i, j in list(a['pos metadata'].items())}

        
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

def get_trajectory_mask(points_in_space, frame_dimension, extremities = (0,0)):
    '''
    points_in_space: list of tuples with coordinates (x,y).
    frame_dimension: tuple with shape dimension
    extremities: tuple with amount to subtract from the frame_dimension
    '''
    xs = list(list(zip(*points_in_space))[0])
    ys = list(list(zip(*points_in_space))[1])
    a, b = get_trajectory(xs, ys, (0 + extremities[0], frame_dimension[1]- extremities[1]))
    traject_mask = get_mask_on_trajectory(frame_dimension, a, b, radius = 15)
    return traject_mask

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
                        
    parser.add_argument('--tcwd_dim', 
                        dest='tcwd',
                        type=int,
                        default = 10,
                        required=False,
                        help='Time course window dimension -pixels radius-') 

    parser.add_argument('--wd_dim', 
                        dest='wd',
                        type=int,
                        default = 150,
                        required=False,
                        help='Window dimension for single stroke centroid detection -pixels side of a square-') 

    parser.add_argument('--vis', 
                        dest='data_vis_switch', 
                        action='store_true')
    parser.add_argument('--no-vis', 
                        dest='data_vis_switch', 
                        action='store_false')
    parser.set_defaults(data_vis_switch=False)  
    
    parser.add_argument('--store', 
                        dest='store_switch',
                        action='store_true')
    parser.add_argument('--no-store', 
                        dest='store_switch', 
                        action='store_false')
    parser.set_defaults(store_switch=False)   

    start_process_time = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args()

    print(args)

    # Store time boundaries
    #time_limits_single = ((args.bottom_time_window, args.upper_time_window))

    # Session path extraction
    path_session = args.path_md.split('derivatives')[0]
    
    # Instance of the retinotopy session
    retino_session = RetinoSession(path_session, 
                                   args.path_md, 
                                   args.green_name, 
                                   conditions_id=args.conditions_id, 
                                   single_stroke_label=args.single_stroke_label, 
                                   multiple_stroke_label=args.apparent_motion_label,
                                   time_course_window_dim=args.tcwd,
                                   window_dim=args.wd,
                                   store_switch=args.store_switch,
                                   data_vis_switch=args.data_vis_switch) 
    
    retino_session.get_retino_session()