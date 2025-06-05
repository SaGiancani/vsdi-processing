import argparse, datetime, utils
import data_visualization as dv
import numpy as np
import os
import process_vsdi as process
from middle_process import get_classic_signal, Condition 
import retinotopy 
from scipy.ndimage.filters import median_filter
from scipy.ndimage import rotate, zoom
import trajectory as trj 
import utils

AREA_MAXIMI_FOR_PEAK = 5

class SpatioTemporalMap:
    def __init__(self, 
                 path_session,
                 trajectory_mask = None,
                 rotation_theta = None,
                 onset_time = None,
                 condition_name = None,
                 data = None,
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
                 bounds_for_max_seek = (None, None, None, None), 
                 logger = None,
                 storing_path = None):
        self.signal                   = data                           #MODIFY THIS. It could get the full matrix, and then run the get_spatio_temporal_profile iteratively on single trials and on average across trials
        self.avrg_signal              = None        
        self.path_session             = path_session
        self.storing_path             = storing_path
        self.session_name             = utils.get_session_id_name(self.path_session)
        self.trajectory_mask          = trajectory_mask
        self.rotation_angle           = rotation_theta
        self.rotate_correction_factor = rotate_correction_factor
        self.discard_thresh           = discard_thresh
        self.logger                   = logger
        
        if self.signal is not None:
            # Filtering of average across trials
            if len(self.signal.shape) == 4:
                tmp_signal                          = np.nanmean(self.signal, axis = 0)
                tmp                                 = np.array([get_spatio_temporal_profile(i,
                                                                                            self.trajectory_mask, 
                                                                                            self.rotation_angle, 
                                                                                            correction_factor = self.rotate_correction_factor, 
                                                                                            discard_thresh = self.discard_thresh) for i in self.signal])
                self.maps, self.masked_data_trials  = np.array(list(zip(*tmp))[0]), np.array(list(zip(*tmp))[1])           
                
            elif len(self.signal.shape) == 3:
                tmp_signal                          = self.signal
                self.maps, self.masked_data_trials  = None, None
            
            else:
                utils.stampa('Something wrong with signal shape', logger=self.logger)
            
            tmp_signal                          = np.array([median_filter(i, size=(5,5)) for i in tmp_signal])
            self.avrg_signal                    = process.gaussian3d(tmp_signal, std = 1.5, size = 5)
            self.map, self.masked_data          = get_spatio_temporal_profile(tmp_signal, 
                                                                              self.trajectory_mask, 
                                                                              self.rotation_angle, 
                                                                              correction_factor = self.rotate_correction_factor, 
                                                                              discard_thresh = self.discard_thresh)
            self.signal                         = None 
        else:
            self.map, self.masked_data          = None, None
            self.maps, self.masked_data_trials  = None, None

        # Pos and timing of peak
        if (retino_pos is not None) and (retino_time is not None):
            self.retino_pos      = retino_pos
            self.retino_time     = retino_time
        else:
            if self.map is not None:
                (a, b), _ = process.find_highest_sum_area(self.map, AREA_MAXIMI_FOR_PEAK, *bounds_for_max_seek)
                self.retino_pos  = [a]
                self.retino_time = [b]    
            else:
                self.retino_pos  = None
                self.retino_time = None    

        self.pixel_spacing            = pixel_spacing
        self.interstimulus_delay      = is_delay#ms
        self.sampling_rate            = sampling_rate

        # Contours level
        if (high_level is None) and (low_level is None) and (self.map is not None):
            self.high_level   =  np.nanpercentile(self.map, 97.7)
            self.low_level    =  np.nanpercentile(self.map, 15)
        else:
            self.high_level   = high_level 
            self.low_level    = low_level

        self.onset_time            = onset_time
        self.condition_name        = condition_name
        self.condition_type        = condition_type

        # Adjusting color 
        if self.retino_pos is not None:
            if (self.condition_type == 'ss') and (len(self.retino_pos) == 1):
                self.colors_retinotopy = ['grey']
            else:
                self.colors_retinotopy = colors_ret

    def visualize_maps(self, 
                       colors_retinotopy, 
                       threshold_contour, 
                       high_level  = None, 
                       low_level   = None, 
                       retino_pos  = None,
                       retino_time = None, 
                       color_mappa = utils.PARULA_MAP):
        
        if self.storing_path is not None:
            tmp = dv.set_storage_folder(storage_path = self.storing_path, name_analysis = 'STProfiles')
            new_storing_path = os.path.join(tmp, f'STProfile_{self.condition_name}_{self.session_name}')  
        else:
            new_storing_path = None

        if high_level is None:
            high_level = self.high_level
        elif low_level is None:
            low_level = self.low_level

        if retino_pos is None:
            retino_pos = self.retino_pos
        if retino_time is None:
            retino_time = self.retino_time
        
        if len(retino_pos) >1:
            color_peak = 'teal'
            peak_traj = True
        else:
            color_peak = 'w'
            peak_traj = False
        
        utils.stampa(f'Onset time {self.onset_time}', logger = self.logger)
        plot_st(self.map, 
                threshold_contour, 
                self.trajectory_mask,
                self.pixel_spacing,
                retinotopic_pos  = retino_pos,
                retinotopic_time = retino_time, 
                map_type   = color_mappa,
                st_title   = self.condition_name,
                onset_time = self.onset_time,
                colors_retinotopy = colors_retinotopy,
                draw_peak_traj    = peak_traj,
                is_delay    = self.interstimulus_delay,#ms
                sampling_fq = self.sampling_rate,#Hz
                high_level  = high_level,
                color_peak  = color_peak,
                low_level   = low_level,
                store_path  = new_storing_path)
        
        tmp = dv.set_storage_folder(storage_path  = tmp, 
                                    name_analysis = 'single_trials')
        utils.stampa(f'Store maps single trials at {tmp}', logger = self.logger)
        if self.maps is not None:
            for n, i in enumerate(self.maps):
                plot_st(i, 
                        threshold_contour, 
                        self.trajectory_mask,
                        self.pixel_spacing,
                        retinotopic_pos  = retino_pos,
                        retinotopic_time = retino_time, 
                        map_type   = color_mappa,
                        st_title   = f'{self.condition_name}_trial{n}',
                        onset_time = self.onset_time,
                        colors_retinotopy = colors_retinotopy,
                        draw_peak_traj    = peak_traj,
                        is_delay    = self.interstimulus_delay,#ms
                        sampling_fq = self.sampling_rate,#Hz
                        high_level  = high_level,
                        color_peak  = color_peak,
                        low_level   = low_level,
                        store_path  = os.path.join(tmp, f'STProfile_{self.condition_name}_{n}_{self.session_name}'))
                        
        return
    
    def store_stmap(self, t):
        tp = [self.signal,
              self.path_session, 
              self.storing_path, 
              self.session_name, 
              self.trajectory_mask, 
              self.rotation_angle, 
              self.rotate_correction_factor, 
              self.discard_thresh, 
              self.map, 
              self.masked_data, 
              self.retino_pos, 
              self.retino_time, 
              self.pixel_spacing,
              self.interstimulus_delay,
              self.sampling_rate,
              self.high_level,
              self.low_level,
              self.onset_time,
              self.condition_name,
              self.condition_type,
              self.colors_retinotopy,
              self.maps, 
              self.masked_data_trials,
              self.avrg_signal]
        
        storage_path = os.path.join(t, 'spatiotemporal_profile')
        tmp = dv.set_storage_folder(name_analysis = os.path.join(storage_path,))
        utils.inputs_save(tp, os.path.join(tmp,'st_map_'+self.condition_name))
        return

    def load_stmap(self, path):
        tp = utils.inputs_load(path)
        self.signal                     = tp[0]
        self.path_session               = tp[1]
        self.storing_path               = tp[2]
        self.session_name               = tp[3]
        self.trajectory_mask            = tp[4]
        self.rotation_angle             = tp[5]
        self.rotate_correction_factor   = tp[6]
        self.discard_thresh             = tp[7]
        self.map                        = tp[8]
        self.masked_data                = tp[9]
        self.retino_pos                 = tp[10]
        self.retino_time                = tp[11]
        self.pixel_spacing              = tp[12]
        self.interstimulus_delay        = tp[13]
        self.sampling_rate              = tp[14]
        self.high_level                 = tp[15]
        self.low_level                  = tp[16]
        self.onset_time                 = tp[17]
        self.condition_name             = tp[18]
        self.condition_type             = tp[19]
        self.colors_retinotopy          = tp[20]
        self.maps                       = tp[21]
        self.masked_data_trials         = tp[22]    
        self.avrg_signal                = tp[23]    
        return

class SpatioTemporalSession:
    def __init__(self,
                 path_session,
                 logger          = None,  
                 store_switch    = False,
                 vis_switch      = True, 
                 conditions_id   = None, 
                 green_name      = '',
                 single_stroke_label   = 'pos',
                 multiple_stroke_label = 'am',
                 acquisition_fq  = 100, #Hz
                 optical_ratio   = 85/50, #Optical magnification
                 cortical_dim    = 14.5, #mm 
                 denoise_flag    = False,
                 retin_fold_path = None,
                 zmaps_flag      = False,
                 load_data       = True,
                 **kwargs):

        self.acquisition_frequency = acquisition_fq #Hz
        self.time_bin              = (1/self.acquisition_frequency)*1000 #ms

        self.denoise_switch        = denoise_flag
        self.vis_switch            = vis_switch
        self.store_switch          = store_switch
        self.zmaps_flag            = zmaps_flag
        if retin_fold_path is None:
            self.retin_folder      = os.path.join(dv.STORAGE_PATH, utils.NAME_RETINO_ANALYSIS)
        else:
            self.retin_folder      = retin_fold_path  

        self.storing_folder        = dv.set_storage_folder(storage_path  = dv.STORAGE_PATH, 
                                                           name_analysis = utils.NAME_SPACETIME_ANALYSIS)
        self.path_to_derivatives   = path_session
        self.path_session          = path_session.split('derivatives')[0]
        if logger is None:
            self.log = utils.setup_custom_logger('myapp')
        else:
            self.log = logger     
        self.green                 = utils.get_green(green_name, self.path_session, log=self.log)

        self.single_stroke_label   = single_stroke_label
        self.multiple_stroke_label = multiple_stroke_label

        # Create an instance of RetinoSession instead of inheriting
        self.retino_session = retinotopy.RetinoSession(path_session=self.path_session,
                                                       logger=self.log,
                                                       path_md=self.path_to_derivatives,
                                                       green_name=green_name,
                                                       single_stroke_label=self.single_stroke_label,
                                                       multiple_stroke_label=self.multiple_stroke_label,
                                                       conditions_id=conditions_id,
                                                       store_switch=self.store_switch,
                                                       data_vis_switch=self.vis_switch,
                                                       denoise_flag=self.denoise_switch ,
                                                       **kwargs)
        
        self.ny, self.nx       = self.retino_session.std_blank.shape 
        self.cond_dict         = self.retino_session.cond_dict
        self.cond_pos          = self.retino_session.cond_pos
        self.cond_am           = self.retino_session.cond_am
        self.cond_names        = self.retino_session.cond_names
        self.retino_pos_am     = self.retino_session.retino_pos_am
        self.color_pos         = {i: dv.COLORS_7[n]  for n, i in enumerate(list(self.retino_session.cond_pos.values()))}

        self.stimulus_metadata       = self.retino_session.stimulus_metadata
        self.stimulus_speed          = self.stimulus_metadata['speed']
        self.timing_single_stroke    = (self.stimulus_metadata['single stroke']['bottom limit'], self.stimulus_metadata['single stroke']['upper limit'])
        self.timing_am_sequence      = (self.stimulus_metadata['multiple stroke']['bottom limit'], self.stimulus_metadata['multiple stroke']['upper limit'])
       
        if self.zmaps_flag:
            self.id_name           = f'{self.retino_session.id_name}_z'
            if  load_data:
                # Safety check: load a md file for checking a coherent dimensionality between the retinotopic centroids loaded and the actual data for st maps
                cd_x       = Condition() 
                conds_list = os.listdir(os.path.join(self.path_to_derivatives, 'md_data'))
                conds_list = [os.path.join(self.path_to_derivatives, 'md_data', i) for i in conds_list if 'md_data_' in i]     
                cd_x.load_cond(conds_list[0].split('.pickle')[0])  
                _, _, y2check, _ = cd_x.df_fz.shape

                bin_tmp = int(np.ceil(y2check/self.ny))
                utils.stampa(f'Relative bin to correct: {bin_tmp}', logger = self.log)

                self.dict_zeta   = get_classic_signal(self.path_to_derivatives, 
                                                      np.nanmin([self.timing_single_stroke[0], self.timing_am_sequence[0]]), 
                                                      bin_value = bin_tmp, 
                                                      denoise_flag = self.denoise_switch,
                                                      log = self.log)
        else:
            self.id_name           = self.retino_session.id_name
            self.dict_zeta         = None

        self.time_sequence           = self.retino_session.header['n_frames']
        self.time_ss                 = np.linspace(-(self.timing_single_stroke[0]-1)*self.time_bin, 
                                                   (self.time_sequence-(self.timing_single_stroke[0]))*self.time_bin, 
                                                   self.time_sequence)
        self.time_am                 = np.linspace(-(self.timing_am_sequence[0]-1)*self.time_bin, 
                                                   (self.time_sequence-(self.timing_am_sequence[0]))*self.time_bin, 
                                                   self.time_sequence)

        # Get original data shape 
        try:
            self.original_frame_shape = self.green.shape
        except:
            self.original_frame_shape = (1312, 1312)

        self.spatial_bin        = np.nanmax(self.original_frame_shape)/np.nanmax([self.ny, self.nx])  #Import green and import an md file and check the difference in frame shape
        self.pixel_spacing      = self.spatial_bin*(cortical_dim*(optical_ratio))/np.nanmax(self.original_frame_shape)        
        utils.stampa(f'Spatial bin: {self.spatial_bin}\n', logger = self.log)  
        utils.stampa(f'Cortical dim: {cortical_dim}\n', logger = self.log)  
        utils.stampa(f'Optical ratio: {optical_ratio}\n', logger = self.log)  
        utils.stampa(f'Original frame shape: {self.original_frame_shape}\n', logger = self.log)  
        utils.stampa(f'Pixel Spacing: {self.pixel_spacing}\n', logger = self.log)  
 
        self.single_pos         = retinotopy.get_retinotopic_single_pos(self.retin_folder, 
                                                                        list(self.retino_session.cond_pos.values()), 
                                                                        self.path_session, 
                                                                        denoise_flag = self.denoise_switch)
        utils.stampa(f'{self.single_pos}', logger = self.log)  
        self.trajectory_mask     = trj.get_trajectory_mask(self.single_pos, (self.ny, self.nx), extremities = (0,0))        
        _, _, self.orient_traj   = trj.rotate_distribution(list(list(zip(*self.single_pos))[0]), 
                                                          list(list(zip(*self.single_pos))[1])) #in rad
        self.data_dictionary     = {}
        self.data_pos_frame      = {}

    def get_session(self):
        utils.stampa(f'Start processing spatiotemporal profile analysis \n', logger=self.log)
        start_time = datetime.datetime.now().replace(microsecond=0)     
        dict_ss    = {}   
        starting_time  = int(np.ceil(0.06/(1/self.acquisition_frequency))) #60ms of synaptic delay injected

        # Single stroke conditions
        utils.stampa(f'Start processing single stroke conditions\n', logger=self.log)
        for cond_id, cond_name in self.cond_pos.items():
            averaged_signal    = self.get_spatiotemporal_maps(cond_name, synaptic_latency = starting_time) 
            dict_ss[cond_name] = averaged_signal

        # Apparent motion conditions and corresponding linear predictions with subtraction
        utils.stampa(f'Start processing apparent motion conditions and corresponding linear prediction with subtractions\n', logger=self.log)
        for cond_id, cond_name in self.cond_am.items():
            _                  = self.get_spatiotemporal_maps(cond_name, synaptic_latency = starting_time, single_pos_cds = [dict_ss[i] for i in self.retino_pos_am[cond_name]]) 
        utils.stampa(f'Start processing subtractions among conditions -not linear prediction-')
        dict_subtrs = utils.get_conds_for_sub(self.path_session)
        for c1, c2 in dict_subtrs.items():
            cd1       = self.data_dictionary[c1]
            cd2       = self.data_dictionary[c2]
            colors    = [self.color_pos[i] for i in self.retino_pos_am[c1]]
            positions = [self.data_pos_frame[ss][0] for ss in self.retino_pos_am[c1]]
            times     = [self.data_pos_frame[ss][1] - (self.timing_single_stroke[0] - self.timing_am_sequence[0]) for ss in self.retino_pos_am[c1]]
            
            if (cd1.interstimulus_delay == cd2.interstimulus_delay) or (c2 in list(self.cond_pos.values())):  
                # Subtraction between maps goes here
                self.get_subtraction_condition(cd2, 
                                               cd1, 
                                               cd1.interstimulus_delay, 
                                               colors, 
                                               positions,
                                               times)                
            else:
                utils.stampa(f'{c1}-{c2} skipped!', logger=self.log)

        meta_dict = {'id_name': self.id_name,
                     'acquisition_freq': self.acquisition_frequency,
                     'denoise_flag': self.denoise_switch,
                     'pixel_spacing': self.pixel_spacing, 
                     'storing_folder': self.storing_folder,
                     'data_folder': self.path_to_derivatives,
                     'zscore_flag': self.zmaps_flag, 
                     'start_time_am': self.timing_am_sequence,
                     'start_time_ss': self.timing_single_stroke, 
                     'stimulus_speed': self.stimulus_speed}
        utils.write_parse(meta_dict, os.path.join(self.storing_folder, self.id_name))
        utils.stampa(f'End processing spatiotemporal profile analysis', logger=self.log)   
        utils.stampa(f'Analysis elaborated in {str(datetime.datetime.now().replace(microsecond=0)-start_time)}!\n', logger=self.log)                 
        return
    
    def get_spatiotemporal_maps(self, name_cond, synaptic_latency = 6, single_pos_cds = None):
        utils.stampa(f'Get spatiotemporal profiles for condition {name_cond} \n', logger=self.log)
        start_time = datetime.datetime.now().replace(microsecond=0)

        if self.zmaps_flag:
            signal      = self.dict_zeta[name_cond]
            avrg_signal = np.nanmean(self.dict_zeta[name_cond], axis = 0)

        else:
            cd          = self.retino_session.get_data_to_process(name_cond)
            signal      = cd.df_fz
            avrg_signal = cd.averaged_df

        st_map_cd, positions, times, colors, start_time_cd, ISinterval, cd_type_flag = self.get_condition_map(signal, name_cond, synaptic_latency = synaptic_latency)
        min_level = np.nanpercentile(st_map_cd.maps, 15)
        max_level = np.nanpercentile(st_map_cd.maps, 95)
        thresh    = np.nanpercentile(st_map_cd.maps, 60)

        # Linear prediction
        if (single_pos_cds is not None) and (name_cond in list(self.cond_am.values())):
            st_map_linear_pred, new_times = self.get_linear_predicted_maps(name_cond, 
                                                                           self.timing_single_stroke[0] - synaptic_latency, 
                                                                           colors, times, positions, 
                                                                           (max_level, min_level, thresh),
                                                                           ISinterval = ISinterval, 
                                                                           single_pos_cds = single_pos_cds, 
                                                                           cd_type_flag = cd_type_flag)
            
            # Subtraction between maps goes here
            self.get_subtraction_condition(st_map_linear_pred, 
                                           st_map_cd, 
                                           ISinterval, 
                                           colors, positions, 
                                           new_times)     

        self.data_dictionary[name_cond] = st_map_cd               
        utils.stampa(f'End processing spatiotemporal profiles for condition {name_cond}', logger=self.log)
        utils.stampa(f'Condition {name_cond} elaborated in {datetime.datetime.now().replace(microsecond=0)-start_time}!\n', logger=self.log)                     
        return avrg_signal
    
    def get_subtraction_condition(self, map_linear_pred, map_cond, ISinterval, colors, positions, times, sanity_switch = True):

        # Sanity check in dimensions
        tmp_map       = map_cond.avrg_signal
        tmp_linear    = map_linear_pred.avrg_signal

        utils.stampa(f'AM sequence map shape {tmp_map.shape}, Linear prediction map shape {tmp_linear.shape}', logger = self.log)
        utils.stampa(f'Onset time Map cd {map_cond.onset_time}, Onset time Map linear {map_linear_pred.onset_time}', logger = self.log)
        utils.stampa(f'AM sequence map reshape as linear map {tmp_map.shape}', logger = self.log)

        # First alignment: mismatch due to the linear summation that makes you lose as many frames as the time stepping between dots        
        # Second alignment: mismatch due to temporal misalignment of AM and ss condition
        # Considering the Onset times (the first vertical line for onset stimulus) the realignment is automatic and takes in account both aspects
        misalign_time_cds   = map_cond.onset_time - map_linear_pred.onset_time
        new_onset_time      = np.nanmin([map_cond.onset_time, map_linear_pred.onset_time]) 
        if misalign_time_cds < 0:
            tmp_linear = tmp_linear[abs(misalign_time_cds):, :, :]
        elif misalign_time_cds > 0:
            tmp_map    = tmp_map[misalign_time_cds:, :, :]            

        # Sanity check on dimensionality and dropping out last frames
        name_cond_sub = f'{map_cond.condition_name} - {map_linear_pred.condition_name}'
        terminal_correction = tmp_map.shape[0] - tmp_linear.shape[0]

        utils.stampa(f'AM sequence map reshape misalign {tmp_map.shape}', logger = self.log)
        utils.stampa(f'Linear sequence map reshape misalign {tmp_linear.shape}', logger = self.log)

        # Drop out of the last frames for shape coherency
        if terminal_correction > 0:
            tmp_map       = tmp_map[:(tmp_linear.shape[0]), :, :]
        else:
            tmp_linear    = tmp_linear[:(tmp_map.shape[0]), :, :]

        utils.stampa(f'AM sequence map reshape terminal {tmp_map.shape}', logger = self.log)
        utils.stampa(f'Linear sequence map reshape terminal {tmp_linear.shape}', logger = self.log)

        subtraction   =  tmp_map - tmp_linear                    

        if sanity_switch:
            sanity_tmp_map, _          = get_spatio_temporal_profile(tmp_map, 
                                                                     map_cond.trajectory_mask, 
                                                                     map_cond.rotation_angle, 
                                                                     correction_factor = map_cond.rotate_correction_factor, 
                                                                     discard_thresh = map_cond.discard_thresh)
            dv.plot_averaged_map(f'Subtraction_SanityCheck_{map_cond.condition_name}', None, None, None, 
                                 sanity_tmp_map, None, 
                                 np.nanpercentile(tmp_map, 15), 
                                 np.nanpercentile(tmp_map, 95), 
                                 None, 
                                 f'{self.id_name}', 
                                 None, 
                                 name_analysis_ = os.path.join(self.storing_folder, self.id_name, name_cond_sub), 
                                 store_path = '')        
            sanity_tmp_lin, _          = get_spatio_temporal_profile(tmp_linear, 
                                                                     map_linear_pred.trajectory_mask, 
                                                                     map_linear_pred.rotation_angle, 
                                                                     correction_factor = map_linear_pred.rotate_correction_factor, 
                                                                     discard_thresh = map_linear_pred.discard_thresh)
            dv.plot_averaged_map(f'Subtraction_SanityCheck_{map_linear_pred.condition_name}', None, None, None, 
                                 sanity_tmp_lin, None, 
                                 np.nanpercentile(tmp_map, 15), 
                                 np.nanpercentile(tmp_map, 95), 
                                 None, 
                                 f'{self.id_name}', 
                                 None, 
                                 name_analysis_ = os.path.join(self.storing_folder, self.id_name, name_cond_sub), 
                                 store_path = '')                        

        st_map_cd = SpatioTemporalMap(self.path_session, 
                                      trajectory_mask = self.trajectory_mask,
                                      rotation_theta  = self.orient_traj,
                                      onset_time      = new_onset_time,
                                      condition_name  = name_cond_sub,
                                      data            = subtraction,
                                      condition_type  = 'am',
                                      is_delay        = ISinterval,
                                      pixel_spacing   = self.pixel_spacing,#mm 
                                      sampling_rate   = self.acquisition_frequency, 
                                      storing_path    = os.path.join(self.storing_folder, self.id_name, name_cond_sub), 
                                      logger          = self.log)    
        utils.stampa(f'Data shape {st_map_cd.masked_data.shape}', logger=self.log)
          
        if self.vis_switch:
            dv.whole_time_sequence(st_map_cd.avrg_signal, 
                                   mask = np.ones((st_map_cd.avrg_signal.shape[-2], st_map_cd.avrg_signal.shape[-1]), dtype = bool),
                                   name=f'time_sequence_subtraction_{name_cond_sub}_{self.id_name}', 
                                   max = 97, min = 15,
                                   ext = 'png',
                                   name_analysis_ = os.path.join(self.storing_folder, self.id_name, name_cond_sub), 
                                   store_path = '')
                   
            st_map_cd.visualize_maps(colors, np.nanpercentile(map_cond.maps, 60), 
                                     retino_pos = positions, 
                                     retino_time = times, # This could be bugged if cd2/linear pred onset time > cd1/am seq onset time 
                                     high_level = np.nanpercentile(map_cond.maps, 80), 
                                     low_level  = -np.nanpercentile(map_cond.maps, 80))
        
        if self.store_switch:
            st_map_cd.store_stmap(os.path.join(self.storing_folder, self.id_name, name_cond_sub))                
        return

    def get_condition_map(self, signal, name_cond, synaptic_latency = 6):
        # Single stroke condition
        if name_cond in list(self.cond_pos.values()):
            cd_type_flag  = 'ss'
            ISinterval    = 30 # does not matter
            start_time_cd = self.timing_single_stroke[0]
            positions, times = None, None
            colors        = ['w']

        # Multiple stroke condition
        elif name_cond in list(self.cond_am.values()):
            # Try to check if retino_cond already exists
            cd_type_flag  = 'am'
            ISspacing     = self.stimulus_metadata['pos metadata'][name_cond]['inter stimulus space'] #in dva
            ISinterval    = int(np.ceil((ISspacing/self.stimulus_speed)*1000))
            start_time    = self.stimulus_metadata['pos metadata'][name_cond]['start'] 
            start_time_cd = self.timing_am_sequence[0] - start_time           
            positions     = [self.data_pos_frame[ss][0] for ss in self.retino_pos_am[name_cond]] 
            times         = [self.data_pos_frame[ss][1] - (self.timing_single_stroke[0] - self.timing_am_sequence[0]) - start_time for ss in self.retino_pos_am[name_cond]] 
            utils.stampa(f'InterStimulus spacing: {ISspacing}, ISI: {ISinterval}, Starting time {start_time_cd}, MultiAM starting frame {start_time}', logger = self.log)
            colors        = [self.color_pos[i] for i in self.retino_pos_am[name_cond]]
        
        start_time_cd -= synaptic_latency

        try:
            st_map_cd = SpatioTemporalMap(self.path_session, condition_type = cd_type_flag, logger = self.log)
            tmp_name =  os.path.join(self.id_name, name_cond, 'spatiotemporal_profile', f'st_map_{name_cond}') 
            utils.stampa(f'{tmp_name} loaded!', logger = self.log)
            st_map_cd.load_stmap(tmp_name)    
        # If does not, it build it
        except:            
            st_map_cd = SpatioTemporalMap(self.path_session, 
                                          trajectory_mask = self.trajectory_mask,
                                          rotation_theta  = self.orient_traj,
                                          onset_time      = start_time_cd,
                                          condition_name  = name_cond,
                                          data            = signal,
                                          condition_type  = cd_type_flag,
                                          is_delay        = ISinterval,
                                          pixel_spacing   = self.pixel_spacing,#mm 
                                          sampling_rate   = self.acquisition_frequency, 
                                          storing_path    = os.path.join(self.storing_folder, self.id_name, name_cond), 
                                          logger          = self.log)
            utils.stampa(f'{name_cond} elaborated!', logger = self.log)

        if self.vis_switch:
            st_map_cd.visualize_maps(colors, np.nanpercentile(st_map_cd.maps, 70), 
                                     retino_pos = positions, retino_time = times,
                                     high_level = np.nanpercentile(st_map_cd.maps, 95), 
                                     low_level = np.nanpercentile(st_map_cd.maps, 15))
            utils.stampa(f'Data shape {st_map_cd.masked_data.shape}', logger=self.log)

            # Sanity check on rotation
            dv.plot_averaged_map(f'{name_cond}_SanityCheck', None, None, None, 
                                 st_map_cd.masked_data[st_map_cd.masked_data.shape[0]//2, :, :], None, 
                                 np.nanpercentile(st_map_cd.masked_data, 15), 
                                 np.nanpercentile(st_map_cd.masked_data, 95), 
                                 None, 
                                 f'{self.id_name}', 
                                 None, 
                                 name_analysis_ = os.path.join(self.storing_folder, self.id_name, name_cond), 
                                 store_path = '')    
        
        # If true store variables
        if self.store_switch:
            st_map_cd.store_stmap(os.path.join(self.storing_folder, self.id_name, name_cond))                            

        if name_cond in list(self.cond_pos.values()):
            self.data_pos_frame[name_cond] = [st_map_cd.retino_pos[0], st_map_cd.retino_time[0]]    
            utils.stampa(f'Update to spatio-temporal dictionary: {self.data_pos_frame}', logger=self.log)

        return st_map_cd, positions, times, colors, start_time_cd, ISinterval, cd_type_flag

    def get_linear_predicted_maps(self, name_cond, start_time_cd, colors, times, positions, thresholds, ISinterval = None, single_pos_cds = None, cd_type_flag = 'am'):
        name_cond_pred          = ''

        for i in self.retino_pos_am[name_cond]:
            name_cond_pred     += i 

        try:
            st_map_cd = SpatioTemporalMap(self.path_session, condition_type = cd_type_flag, logger = self.log)
            tmp_name  = os.path.join(self.storing_folder, self.id_name, name_cond_pred, 'spatiotemporal_profile', f'st_map_{name_cond_pred}') 
            st_map_cd.load_stmap(tmp_name)   
            utils.stampa(f'Linear prediction {tmp_name} loaded!', logger = self.log)

        # If does not, it build it
        except:          
            time_step          = int(np.ceil(ISinterval/self.time_bin))
            linear_prediction  = get_linear_expectation(single_pos_cds, 
                                                        time_step, 
                                                        nonlinear_zeroframe = start_time_cd-time_step,
                                                        log = self.log)
            # Filtering linear_prediction
            filtered_pred      = linear_prediction            
            time_slide         = (single_pos_cds[0].shape[0] - linear_prediction.shape[0])

            utils.stampa(f'Time bins to remove: {time_slide}, Time step: {time_step}, Starting time {start_time_cd}', logger = self.log)                                                                                  
            st_map_cd = SpatioTemporalMap(self.path_session, 
                                          trajectory_mask = self.trajectory_mask,
                                          rotation_theta  = self.orient_traj,
                                          onset_time      = start_time_cd - time_slide,
                                          condition_name  = name_cond_pred,
                                          data            = filtered_pred,
                                          condition_type  = cd_type_flag,
                                          is_delay        = ISinterval, 
                                          pixel_spacing   = self.pixel_spacing,#mm 
                                          sampling_rate   = self.acquisition_frequency, 
                                          storing_path    = os.path.join(self.storing_folder, self.id_name, name_cond_pred), 
                                          logger          = self.log)
            utils.stampa(f'Linear prediction {name_cond_pred} elaborated!', logger = self.log)

        time_slide         = (single_pos_cds[0].shape[0] - st_map_cd.avrg_signal.shape[0])
        new_times = np.array(times) - time_slide + (self.timing_single_stroke[0] - self.timing_am_sequence[0])
        # Visualize linear prediction
        if self.vis_switch:
            dv.whole_time_sequence(st_map_cd.avrg_signal, 
                                   mask = np.ones((st_map_cd.avrg_signal.shape[-2], st_map_cd.avrg_signal.shape[-1]), dtype = bool),
                                   name = f'time_sequence_subtraction_{st_map_cd.condition_name}_{self.id_name}', 
                                   max = 97, min = 15,
                                   ext = 'png',
                                   name_analysis_ = os.path.join(self.storing_folder, self.id_name, st_map_cd.condition_name), 
                                   store_path = '')    

            st_map_cd.visualize_maps(colors, thresholds[2], 
                                     retino_pos = positions, 
                                     retino_time = new_times,
                                     high_level = thresholds[0], 
                                     low_level = thresholds[1])
            utils.stampa(f'Data shape of linear prediction sequence {st_map_cd.masked_data.shape}', logger=self.log)  
        
        if self.store_switch:
            st_map_cd.store_stmap(os.path.join(self.storing_folder, self.id_name, st_map_cd.condition_name))                

        return st_map_cd, new_times
    

    def get_maximi_single_pos(self):
        path_session = self.path_session
        map_folder   = os.path.join(self.storing_folder, self.id_name)
        map_folder   = utils.normalize_path_os(map_folder)    
        
        for k,v in self.cond_pos.items():
            utils.stampa(f'Single pos cond: {v}', logger = self.log)
            st_map = SpatioTemporalMap(path_session)
            st_map.load_stmap(os.path.join(map_folder, v, 'spatiotemporal_profile', f'st_map_{v}'))
            utils.stampa(f'(pos, time) : ({st_map.retino_pos, st_map.retino_time})', logger = self.log)
            self.data_pos_frame[v] = [st_map.retino_pos[0], st_map.retino_time[0]]      
        return 

    def get_nonlinear_pred_maps(self): 
        path_session = self.path_session
        map_folder   = os.path.join(self.storing_folder, self.id_name)
        map_folder   = utils.normalize_path_os(map_folder)    

        dict_sub_lin = {} 
        peak_dots    = {}
        timing_frame = (1/self.acquisition_frequency)*1000

        for k,v in self.retino_pos_am.items():

            start_time = self.stimulus_metadata['pos metadata'][k]['start'] 
            name_cond_pred = ''
            positions  = [self.data_pos_frame[ss][0] for ss in self.retino_pos_am[k]] 
            times      = [self.data_pos_frame[ss][1] - (self.timing_single_stroke[0] - self.timing_am_sequence[0]) - start_time for ss in self.retino_pos_am[k]]       

            st_map_sub = SpatioTemporalMap(path_session)

            for n, i in enumerate(v):
                name_cond_pred  += i 

            # Checking the slide to add to the peaks
            st_map_sub.load_stmap(os.path.join(map_folder, name_cond_pred, 'spatiotemporal_profile', f'st_map_{name_cond_pred}'))   
            tmp_lin = st_map_sub.map.shape[1]    
            st_map_sub.load_stmap(os.path.join(map_folder, k, 'spatiotemporal_profile', f'st_map_{k}'))   
            tmp_cd  = st_map_sub.map.shape[1]
            corrective_slide = tmp_cd - tmp_lin
            
            # Linear prediction name
            sub_cond = f'{k} - {name_cond_pred}'    
            utils.stampa(f'Subtraction condition: {sub_cond}', logger=self.log)            
            utils.stampa(f'Time linear pred and AM: {tmp_lin, tmp_cd}', logger=self.log)            
            
            st_map_sub.load_stmap(os.path.join(map_folder, sub_cond, 'spatiotemporal_profile', f'st_map_{sub_cond}'))   


            for n, i in enumerate(v):
                isi_frames   = int(np.ceil(st_map_sub.interstimulus_delay/timing_frame))
                times[n]     = times[n] + n*isi_frames - corrective_slide       

            dict_sub_lin[sub_cond] = st_map_sub
            peak_dots[sub_cond]    = list(zip(times, positions))

        return dict_sub_lin, peak_dots


    def get_subtraction_maps(self):
        # Instance folder paths
        path_session = self.path_session
        map_folder = os.path.join(self.storing_folder, self.id_name)
        map_folder = utils.normalize_path_os(map_folder)    
        # Load dict for possible cond subtractions 
        dict_subtrs  = utils.get_conds_for_sub(path_session)
        
        # Instance variables
        dict_sub     = {} 
        peak_dots    = {}
        timing_frame = (1/self.acquisition_frequency)*1000


        for c1, c2 in dict_subtrs.items():
            name_cond_sub = f'{c1} - {c2}'
            st_map_sub = SpatioTemporalMap(path_session)
            st_map_sub.load_stmap(os.path.join(map_folder, name_cond_sub, 'spatiotemporal_profile', f'st_map_{name_cond_sub}'))    

            # colors    = [st_session.color_pos[i] for i in st_session.retino_pos_am[c1]]
            positions = [self.data_pos_frame[ss][0] for ss in self.retino_pos_am[c1]]
            times     = [self.data_pos_frame[ss][1] - (self.timing_single_stroke[0] - self.timing_am_sequence[0]) for ss in self.retino_pos_am[c1]]

            for n, _ in enumerate(self.retino_pos_am[c1]):
                isi_frames   = int(np.ceil(st_map_sub.interstimulus_delay/timing_frame))
                times[n]     = times[n] + n*isi_frames     

            dict_sub[name_cond_sub]  = st_map_sub
            peak_dots[name_cond_sub] = list(zip(times, positions))
        return dict_sub, peak_dots


    def get_maps_last_dot(self, dict_sub, peak_dots, list_dots=[2, 3], list_space=[.5, 1], list_direction=[-1, 1], matrix_dict = None, peaks_dict = None, baseline = None):   
        # Create the nested dictionary with direction as the innermost level
        directions = get_directions(self.data_pos_frame, self.retino_pos_am) 

        if matrix_dict is None:
            matrix_dict = {outer: {middle: {inner: [] for inner in list_direction} 
                                for middle in list_space} for outer in list_dots}
        if peaks_dict is None:            
            peaks_dict  = {outer: {middle: {inner: [] for inner in list_direction} 
                                for middle in list_space} for outer in list_dots}

        if baseline is None:
            baseline    = {outer: {middle: {inner: [] for inner in list_direction} 
                                for middle in list_space} for outer in list_dots}
            
        for k, v in dict_sub.items():
            tmp = k.split(' -')[0]

            n_dots  = len(self.stimulus_metadata['pos metadata'][tmp]['conditions'])
            spacing = self.stimulus_metadata['pos metadata'][tmp]['inter stimulus space']
            
            direction    = directions[tmp] 
            tmp_coord    = peak_dots[k][-1]         
            tmp_map      = dict_sub[k].map[:, tmp_coord[0]:]
            base_frames  = np.nanmin([self.timing_am_sequence[0], self.timing_single_stroke[0]])
            tmp_baseline = dict_sub[k].map[:, :(base_frames)]

            # Normalization of maps' scales across sessions
            tmp_map, scale  = resample_spatiotemporal_map(tmp_map, self.time_bin, self.pixel_spacing)
            tmp_baseline, _ = resample_spatiotemporal_map(tmp_baseline, self.time_bin, self.pixel_spacing)
            
            peaks_dict[n_dots][spacing][direction].append(tmp_coord[1]*scale[0])                  
            matrix_dict[n_dots][spacing][direction].append(tmp_map)
            baseline[n_dots][spacing][direction].append(tmp_baseline)
            
        return matrix_dict, peaks_dict, baseline


def derivative_filter(arr, threshold):
    # Compute the derivative of the array
    derivative = np.diff(arr)

    # Find the indices where the absolute derivative is greater than the threshold
    indices_to_remove = np.where(np.abs(derivative) > threshold)[0] + 1

    # Set the corresponding values to NaN
    arr_filtered = arr.copy()
    arr_filtered[indices_to_remove] = np.nan
    return arr_filtered

def get_directions(data_pos_frame, retino_pos_am):
    return {k: np.sign(data_pos_frame[v[-1]][0] - data_pos_frame[v[0]][0]) for k, v in retino_pos_am.items()}    

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

def get_linear_expectation(array_of_sequences, stepping, nonlinear_zeroframe = 5 , log = None):
    """
    Calculate the linear expectation of a sequence of arrays.

    Parameters:
    - array_of_sequences (list of 2D numpy arrays): List containing multiple time sequences.
    - stepping (int): Number of time steps for shifting the from one stroke's sequence to the next.
    - nonlinear_zeroframe (int, optional): Number of frames to use for nonlinear zeroing. Default is 5.

    Returns:
    - tmp (2D numpy array): Linear expectation of the input sequences.
    """
    # Get the number of sequences
    n_strokes = len(array_of_sequences)
    
    # Ensure that there is more than one time sequence
    assert n_strokes != 1, 'The input has to be an array with more than one time sequence'
    
    # Calculate the step size for global shifting
    # step = int(np.ceil(global_shift/2))
    step = int(stepping)
    
    utils.stampa(f'N° strokes {n_strokes}\n', logger=log)
    utils.stampa(f'Global shift {stepping}\n', logger=log)

    # Copy the input sequences to avoid modifying the original data
    ppp = [np.copy(i) for i in array_of_sequences]
    
    # Calculate the linear expectation by iteratively combining the sequences
    for i in range(len(ppp)-1):
        if i == 0:
            tmp = ppp[i][step:, :] + ppp[i+1][:-step, :]
            utils.stampa(f'Index first stroke: {i}\n', logger=log)
            utils.stampa(f'Index second stroke: {i+1}\n', logger=log)
            utils.stampa(f'Shape temporary matrix {tmp.shape}\n', logger=log)
        else:
            utils.stampa(f'Index {i+1}-th stroke: {i+1}\n', logger=log)            
            tmp = tmp[step:, :] + ppp[i+1][:-(i+1)*step, :]
            utils.stampa(f'Shape temporary matrix {tmp.shape}\n', logger=log)

    
    # Calculate the mean of the first few frames for nonlinear zeroing
    zero_tmp = np.nanmean(tmp[:nonlinear_zeroframe, :, :], axis=0)
    
    # Zero the result using the calculated mean
    tmp = tmp - zero_tmp
    
    # Return the linear expectation
    return tmp

import numpy as np

def maximi_inda_blob(st_matrix, blob, activity_mask = None):
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

    # Further mask is applied: constraining activity area
    if activity_mask is not None:
        active_mask = np.ones(st_matrix.shape)
        active_mask[np.where(ty == 0)[0]] = 0
        active_mask[activity_mask[0]:activity_mask[1], :] = 0
        ty = ty * active_mask

    # Extract valid indices and corresponding positions
    y = ty[np.where(ty != 0)]
    x = np.where(mask_blob == 1)[0]

    # Update the zero_array with valid indices
    zero_array[x] = y

    # Create a list of tuples containing the maximum values' indices and their corresponding positions
    return list(zip(zero_array, np.linspace(0, st_matrix.shape[1]-1, st_matrix.shape[1])))

def resample_spatiotemporal_map(matrix, dt_original, dx_original, dt_target = 10, dx_target = .1):
    """
    Resample a 2D spatiotemporal matrix to target resolutions.
    Assumes:
    - Axis 0: Time (rows)
    - Axis 1: Space (columns)
    """
    # Compute scale factors (time, space)
    scale_t = dt_original / dt_target  # Temporal scaling
    scale_x = dx_original / dx_target  # Spatial scaling
    
    # Resample with per-axis scaling (order=1 for linear interpolation)
    resampled = zoom(matrix, (scale_x, scale_t), order=1)
    return resampled, (scale_x, scale_t)

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

def get_threshold(data_maps, zero_of_cond, start_n_end=False, percentile = 99, full_seq = False):
    '''
    data_maps: n°conditions, space, time
    
    '''
    if not full_seq:
        if start_n_end:
            threshold_contour = np.nanpercentile(np.append(data_maps[:, :, :zero_of_cond].ravel(),
                                                        data_maps[:, :, -zero_of_cond:].ravel()), percentile)  
        else:
            threshold_contour = np.nanpercentile(data_maps[:, :, :zero_of_cond].ravel(), percentile)      
    else:
        if start_n_end:
            threshold_contour = np.nanpercentile(np.append(data_maps[:, :zero_of_cond, :].ravel(),
                                                        data_maps[:, -zero_of_cond:, :].ravel()), percentile)  
        else:
            threshold_contour = np.nanpercentile(data_maps[:, :zero_of_cond, :].ravel(), percentile)              
    return threshold_contour

import matplotlib.pyplot as plt

def plot_st(profilemap,  
            threshold_contour, 
            traj_mask,
            pixel_spacing,
            retinotopic_pos = None,
            retinotopic_time = None, 
            map_type = utils.PARULA_MAP,
            st_title = None,
            onset_time = 4,
            colors_retinotopy = ['crimson', 'tomato', 'magenta'],
            draw_peak_traj = True,
            is_delay = 30,#ms
            sampling_fq = 100,#Hz
            high_level = 5,
            color_peak = 'teal',
            low_level = -1,
            visualize_figure = False,
            store_path = None):
    
    # Safety checks
#     if (retinotopic_pos is not None) and (retinotopic_time is not None):
#         assert len(retinotopic_pos) == len(colors_retinotopy), 'Mismatch in retinotopic positions numbers and colors available'
    space, time  = profilemap.shape
    timing_frame = (1/sampling_fq)*1000
    assert (is_delay/timing_frame)>1, 'Something weird: sampling frequency and timing of a frame incompatible'
    isi_frames   = int(np.ceil(is_delay/timing_frame))

    # Plot colormap
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    fig.set_facecolor('white')
    pc_ = ax.pcolormesh(profilemap, cmap= map_type, vmax = high_level, vmin=low_level)
    
    # Plot intensity contour
    blobs = np.zeros(profilemap.shape, dtype = bool)
#     blobs[np.where(median_filter(profilemap, size=(5,5))>=threshold_contour)] = 1
    blobs[np.where(profilemap>=threshold_contour)] = 1
    ax.contour(blobs, 4, colors='k', alpha = .5, levels=[1])
    
    blobs_ = np.copy(blobs)
    blobs_[np.where(median_filter(profilemap, size=(5,5))>=threshold_contour)] = 1
    
    # Draw peak's trajectory
    if draw_peak_traj:
        a = maximi_inda_blob(profilemap, blobs_)
        ax.scatter(list(list(zip(*a))[1]), list(list(zip(*a))[0]), marker = '.', color = 'k')
        ax.plot(list(list(zip(*a))[1]), list(list(zip(*a))[0]), ls = '-', color = 'k', alpha = .3)
    
    if (retinotopic_pos is not None) and (retinotopic_time is not None):
        number_strokes = len(retinotopic_pos)
        # Plot timelines and retinotopic positions
        for n in range(number_strokes):
            print(f'stroke\'s peak {retinotopic_time[n]+n*isi_frames} coordinate')
            ax.scatter(retinotopic_time[n]+n*isi_frames, retinotopic_pos[n], marker = 'o', color = colors_retinotopy[n], s= 100)

    else:
        colors_retinotopy = [color_peak]
        number_strokes = 1
        
    for n in range(number_strokes):
        plt.vlines(onset_time+n*isi_frames, 
                   np.where(traj_mask != 0)[1].min(), 
                   np.where(traj_mask != 0)[1].max(), 
                   color = colors_retinotopy[n], ls ='--', lw=2)
    
    # Plot highest spot
    if len(retinotopic_pos)>1:
        (a, b), _ = process.find_highest_sum_area(profilemap*blobs_, 5, None, None, onset_time, 45)
        ax.scatter(b,a, marker = 'o', color = color_peak, s= 100)
        print(a, b)
    else:
        a = retinotopic_pos 
        b = retinotopic_time             

    # Custom axis
    strokes_onset_times = [onset_time+i*isi_frames for i in range(number_strokes)]
    strokes_onset_times.sort()
    print(strokes_onset_times)
    start_time_instants = [0] + strokes_onset_times
    tmp = start_time_instants + list(np.linspace(start_time_instants[-1], time, (2+(time-start_time_instants[-1])//10)))

    print(tmp)
    ax.set_xticks(tmp)
    labels_ = [item.get_text() for item in ax.get_xticklabels()]
    # x_tmp = np.arange((zero_of_cond-12), (zero_of_cond+30+12), len(tmp))
    list_x = list()
    for i, x in zip(labels_, tmp):
        list_x.append(f'{int((x-(onset_time))*timing_frame)}')
    ax.set_xticklabels(list_x, fontsize = 12)
    ax.set_xlabel('Time - ms', fontsize = 15)

    tmp_y = np.linspace(0, space-10, 9) 
    ax.set_yticks(tmp_y)
    labels_ = [item.get_text() for item in ax.get_yticklabels()]
    list_y  = list()
    for y in np.linspace(0, (pixel_spacing*space) , 9):
        list_y.append(f'{y:.1f}')
    ax.set_yticklabels(list_y, fontsize = 12)
    ax.set_ylabel('Space - mm', fontsize = 15)
    ax.set_ylim((np.where(traj_mask != 0)[1].min(), np.where(traj_mask != 0)[1].max()))
    fig.colorbar(pc_) 
                   
    if st_title is not None:
        plt.title(st_title, fontsize = 15)
        if store_path is not None:
            # plt.savefig(os.path.join(store_path+ '.pdf'), format = 'pdf', dpi =500)
            plt.savefig(os.path.join(store_path+ '.png'), format = 'png', dpi =500)

    if visualize_figure:
        plt.show()
    plt.close('all')
    return (a,b)

# Example for script launching sbatch Desktop/runpy_giancani.sh st_builder.py --path /envau/work/neopto/DATA_AnDO/exp-AM3_BEHAV+VSDI/sub-Hip/sess-20210108-001/derivatives/spcbin3_timebin1_zerofrms6_strategymae_n_chunk1_movFalse_dtrendFalse_deblankTrue/ --store --vis --denoised --zmaps

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Launching spatio-temporal profile analysis pipeline')

    parser.add_argument('--path', 
                        dest='path',
                        type=str,
                        required=True,
                        help='The middle process datafolder path: point at the folder inside derivatives')
    
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
    
    parser.add_argument('--green_name', 
                        dest='green_name',
                        type=str,
                        default = 'green01.bmp',
                        required=False)  
    
    parser.add_argument('--cid', 
                        action='append', 
                        dest='conditions_id',
                        default=None,
                        type=int,
                        help='Conditions to analyze: None by default -all the conditions-')   
    
    parser.add_argument('--fq', 
                        dest='sampling_fq',
                        default=100,
                        type=int,
                        required=False,
                        help='Acquisition sampling fq: to recover in the lab books. Either 100 or 110Hz usually')  
    
    parser.add_argument('--opt_magn', 
                        dest='optical_magnification',
                        default=50/50,
                        type=float,
                        required=False,
                        help='Optical magnification as ratio of the objectives focal length')  
    
    parser.add_argument('--brain_mm', 
                        dest='recorded_diameter',
                        default=16, #mm
                        type=float,
                        required=False,
                        help='Dimension of the optical recording chamber -the diameter in mm-')  
    
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

    parser.add_argument('--denoised', 
                        dest='denoised_switch',
                        action='store_true')
    parser.add_argument('--no-denoised', 
                        dest='denoised_switch', 
                        action='store_false')
    parser.set_defaults(denoised_switch=False)   

    parser.add_argument('--zmaps', 
                        dest='zmaps_flag', 
                        action='store_true')
    parser.add_argument('--no-zmaps', 
                        dest='zmaps_flag', 
                        action='store_false')
    parser.set_defaults(zmaps_flag=False)  

    
    start_process_time = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args()

    log = utils.setup_custom_logger('myapp')
    utils.stampa(f'{args}', logger = log)            

    # Instance of the retinotopy session
    st_session   = SpatioTemporalSession(args.path, 
                                         logger          = log,
                                         store_switch    = args.store_switch,
                                         vis_switch      = args.data_vis_switch, 
                                         conditions_id   = args.conditions_id, 
                                         acquisition_fq  = args.sampling_fq, #Hz
                                         optical_ratio   = args.optical_magnification, #Optical magnification
                                         cortical_dim    = args.recorded_diameter, #mm 
                                         denoise_flag    = args.denoised_switch,                                         
                                         green_name      = args.green_name,
                                         zmaps_flag      = args.zmaps_flag,                                         
                                         single_stroke_label   = args.single_stroke_label, 
                                         multiple_stroke_label = args.apparent_motion_label) 
    st_session.get_session()