import argparse, datetime
from collections import defaultdict
import data_visualization as dv
import middle_process as md
import numpy as np
import os
import process_vsdi as process
import retinotopy as retino
from scipy.ndimage import median_filter, gaussian_filter
import warnings, utils
            
class ActiveCortexSession:
    def __init__(self,
                 path_session,
                 threshold       = 2,
                 logger          = None,  
                 store_flag      = False,
                 vis_switch      = True, 
                 acquisition_fq  = 100, #Hz
                 optical_ratio   = 85/50, #Optical magnification
                 cortical_dim    = 14.5, #mm 
                 denoise_flag    = False,
                 pretrigger_zero = False,
                 spatial_filter  = True,
                 retino_path     = None,
                 blank_name      = 'blank',
                 trial_metadata_flag = True,
                 zero_frames     = 10,
                 green_name      = '',
                 filter_kernel   = 5,
                 gaussian_kernel = 1,
                 **kwargs):

        if logger is None:
            self.log = utils.setup_custom_logger('myapp')
        else:
            self.log = logger     

        self.threshold             = threshold
        self.path_to_derivatives   = path_session
        self.path_session          = path_session.split('derivatives')[0]

        self.acquisition_frequency = acquisition_fq #Hz
        self.time_bin              = (1/self.acquisition_frequency)*1000 #ms
        self.green                 = utils.get_green(green_name, self.path_session, log=self.log)
        self.blank_name            = blank_name 
        # Get original data shape 
        try:
            self.original_frame_shape = self.green.shape
        except:
            self.original_frame_shape = (1312, 1312)

        self.denoise_switch        = denoise_flag
        self.vis_switch            = vis_switch
        self.store_switch          = store_flag
        self.pretriggerz_flag      = pretrigger_zero
        self.filter_flag           = spatial_filter # Does not work for zmaps_flag
        self.filter_kernel         = filter_kernel # Does not work for zmaps_flag
        self.gaussian_kernel       = gaussian_kernel # Does not work for zmaps_flag
        
        if retino_path is None:
            self.retin_folder      = dv.STORAGE_PATH
        else:
            self.retin_folder      = retino_path  

        self.storing_folder        = dv.set_storage_folder(storage_path  = dv.STORAGE_PATH, 
                                                           name_analysis = utils.NAME_ACTIVECORTEX_SURFACE)


        self.stimulus_metadata     = utils.get_stimulus_metadata(self.path_session) 
        self.stimulus_speed        = self.stimulus_metadata['speed']
        self.timing_single_stroke  = (self.stimulus_metadata['single stroke']['bottom limit'], self.stimulus_metadata['single stroke']['upper limit'])
        self.timing_am_sequence    = (self.stimulus_metadata['multiple stroke']['bottom limit'], self.stimulus_metadata['multiple stroke']['upper limit'])

        self.data_loader         = retino.RetinoLoaderManager(self.path_session,  flag_denoise=self.denoise_switch, storage_path=self.retin_folder, pretrigger_flag = self.pretriggerz_flag)
        utils.stampa(f'Retinotopy data loaded successfully!', logger=self.log)

        self.id_name             = self.data_loader.id_name 
        self.retino_data         = self.data_loader.data  # dict of condition -> RetinotopicObject(s)
        self.time_window_per_cd  = self.get_time_window()
        self.peaks_distribution  = self.get_peaks_distribution()
        utils.stampa(f'{self.time_window_per_cd}', logger=self.log)

        self.cond_am = list(self.data_loader.cond_am)   
        self.cond_pos = list(self.data_loader.cond_pos)
        self.list_conds = self.cond_am + self.cond_pos + [blank_name]
        self.retino_pos_am = utils.get_conditions_correspondance(self.path_session)
        utils.stampa(f'Dictionary of conditions: {self.retino_pos_am}', logger = self.log)    
        utils.stampa(f'{self.list_conds}', logger=self.log)

        self.data, self.dict_autoselection = get_md_files(self.path_to_derivatives, self.list_conds, behavior_flag = trial_metadata_flag, get_md_data = not self.denoise_switch)
        
        if self.denoise_switch:
            # Get denoised selected trials
            directory_path       = os.path.join(self.path_to_derivatives, 'denoised')    
            files_denoise        = os.listdir(directory_path)
            self.data            = utils.load_all_files(directory_path, files_denoise, particle = 'rem_', log = self.log)
        else:
            blank_data =  np.array([process.deltaf_up_fzero(i, zero_frames, deblank = True, blank_sign = None) for i in self.data[blank_name]])
            average_blank = np.nanmean(blank_data, axis = 0)
            dict_values = {}
            for k, v in self.data.values():
                p_dffz      =  np.array([process.deltaf_up_fzero(i,
                                                                 zero_frames, 
                                                                 deblank = True, 
                                                                 blank_sign = average_blank) for i in v])
                dict_values[k] = p_dffz
            self.data = dict_values
        utils.stampa(f'VSDI data loaded successfully!', logger=self.log)

        _, self.nt, self.ny, self.nx = self.data[blank_name].shape
        self.spatial_bin     = np.nanmax(self.original_frame_shape)/np.nanmax([self.ny, self.nx])  #Import green and import an md file and check the difference in frame shape
        self.pixel_spacing   = self.spatial_bin*(cortical_dim*(optical_ratio))/np.nanmax(self.original_frame_shape)    

        tmp_blnk     = self.data[self.blank_name]
        if self.filter_flag:
            tmp_blnk = median_filter(tmp_blnk, size = (1, 1, args.median_kernel, args.median_kernel))
            tmp_blnk = gaussian_filter(tmp_blnk, sigma = (0, args.gaussian_kernel, args.gaussian_kernel, args.gaussian_kernel)) 
        
        self.blank_cd        = tmp_blnk
        self.regular_conds_acs = self.get_conditions()
        self.subtraction_acs   = self.get_subtractions()

    def get_peaks_distribution(self):
        peaks_dict = {}
        for k, v in self.retino_data.items():
            peaks_dict[k] = v[-1].distribution_positions
        return peaks_dict

    def get_time_window(self):
        time_dict = {}
        for cd_am in self.data_loader.cond_am:
            metadata_cd_am   = self.stimulus_metadata['pos metadata'][cd_am]
            time_window      = self.retino_data[cd_am][-1].time_limits
            time_zero        = time_window[0]
            space_step       = metadata_cd_am['inter stimulus space']
            starting_time    = metadata_cd_am['start']  # in frames
            time_step        = int(np.ceil((1 / self.stimulus_metadata['speed']) * space_step * self.acquisition_frequency))
            begin_time       = time_zero + starting_time + len(self.retino_data[cd_am]) * time_step  + time_step 
            end_time         = begin_time + time_step
            time_dict[cd_am] = (begin_time, end_time)

        for cd_pos in self.data_loader.cond_pos:
            time_window = self.retino_data[cd_am][-1].time_limits
            #         # Inject an arbitrary delay (60ms) for maximum signal-to-noise ratio
            delay      = int(np.ceil(0.06 / (1 / self.acquisition_frequency))) # This delay is added to an already present 60ms synaptic delay
            begin_time = time_window[0] + delay
            end_time   = time_window[1] + delay
            time_dict[cd_pos] = (begin_time, end_time)
        return time_dict

    def get_conditions(self, 
                       synaptic_latency = 60, #in ms
                       keep_blob_nan = True,
                       compute_behavior = True):

        median_kernel  = self.filter_kernel
        gaussian_sigma = self.gaussian_kernel
        conditions = {}
        # Only for AM conds
        for cond_name in self.list_conds:
            if cond_name == self.blank_name:
                continue  
            utils.stampa(f"[{cond_name}]: beginning of process...", logger=self.log)

            # safety checks
            if cond_name not in self.data:
                utils.stampa(f"[build_conditions] skipping {cond_name}: no data in self.data", logger=self.log)
                continue
            
            synaptic_latency = int(np.ceil(synaptic_latency/self.time_bin)) # In frames
            if cond_name in self.cond_am:
                start_time    = self.stimulus_metadata['pos metadata'][cond_name]['start'] 
                onset_time = self.timing_am_sequence[0] - start_time - synaptic_latency

            if cond_name in self.cond_pos:
                onset_time = self.timing_single_stroke[0] - synaptic_latency

            data = self.data[cond_name]
            tw = self.time_window_per_cd.get(cond_name, None)
            behavior_dict = self.dict_autoselection.get(cond_name, {})
            peaks = self.peaks_distribution.get(cond_name, None)
            tmp_print = len(behavior_dict['autoselection'])
            utils.stampa(f'Lenght behavior list: {tmp_print} and length peaks distribution {len(peaks[0])}', logger=self.log)
            ac = ActiveCortex(cond_name=cond_name,
                              data=data,
                              pixel_spacing=self.pixel_spacing,
                              time_window=tw,
                              onset_time=onset_time,
                              peaks_distribution=peaks,
                              blank_cd = self.blank_cd,
                              statistical_threshold=self.threshold,
                              behavior_dict=behavior_dict,
                              logger=self.log)
            # run pipeline
            ac.run_full_pipeline(median_kernel=median_kernel,
                                 gaussian_sigma=gaussian_sigma,
                                 threshold=self.threshold,
                                 keep_blob_nan=keep_blob_nan,
                                 compute_behavior=compute_behavior)

            ac.maps         = ac.compute_behavior_maps()
            # Compute the time courses
            ac.time_courses = ac.compute_blob_timecourse()        
            conditions[cond_name] = ac
        return conditions

    def build_conditions(self):
        """
        Build an ActiveCortex object for each condition in self.list_conds.
        Returns a dict: cond_name -> ActiveCortex instance.
        """
        conditions = self.regular_conds_acs
        # Only for AM conds
        for cond_name in self.list_conds:
            if cond_name == self.blank_name:
                continue  
            ac   = conditions[cond_name]
            maps = ac.maps
            if self.vis_switch:
                cmaps    = maps['map']
                peaks    = maps['peaks']
                blobs    = maps['blob']
                min_bord = np.nanpercentile(ac.zscored_data, 3)
                max_bord = np.nanpercentile(ac.zscored_data, 97)

                for key in cmaps.keys():
                    mappa = cmaps[key]
                    picco = peaks[key]
                    blob  = blobs[key]

                    if mappa is None or np.all(np.isnan(mappa)):
                        continue
                    len_cd = ac.time_courses['n_trials'][key]
                    dv.plot_averaged_map(f"{ac.cond_name}_{key}",
                                         blob,
                                         None,
                                         picco,
                                         mappa,
                                         None,
                                         min_bord, max_bord,
                                         'k',
                                         f'{self.id_name} - trials: {len_cd}',
                                         'k',
                                         name_analysis_=os.path.join(self.id_name, ac.cond_name, 'SurfaceMap'),
                                         store_path=self.storing_folder)
                ac.maps  = cmaps
                ac.peaks = peaks
                ac.blobs = blobs

                time_series_info = [ac.onset_time, self.time_bin, ac.filtered_data.shape[1]] #zero, time_interval, time_bins
                plot_blob_timecourse(ac.time_courses, time_series_info, y_lim = (min_bord, max_bord), store_pic=True,
                                     name_cond = ac.cond_name, title_plot = f'Blob time course {ac.cond_name}', 
                                     name_analysis_ = os.path.join(self.id_name, ac.cond_name),
                                     store_path = self.storing_folder)
            
            if self.store_switch:
                ac.store_activecortex(os.path.join(self.storing_folder, self.id_name, ac.cond_name))
        return 

    def build_sub_conditions(self):

        if self.subtraction_acs is None:
            self.subtraction_acs = self.get_subtractions()

        if self.vis_switch:
            for sub_cond_name, values in self.subtraction_acs.items():
                ac = values[0]
                mappa = ac.map
                picco = ac.peaks
                blob  = ac.blob_binary  
                min_bord = np.nanpercentile(mappa, 10)
                max_bord = np.nanpercentile(mappa, 97)

                if mappa is None or np.all(np.isnan(mappa)):
                    continue
                dv.plot_averaged_map(f"{sub_cond_name}",
                                        blob,
                                        None,
                                        picco,
                                        mappa,
                                        None,
                                        min_bord, max_bord,
                                        'k',
                                        f'{self.id_name}',
                                        'k',
                                        name_analysis_=os.path.join(self.id_name, ac.cond_name, 'SurfaceMap'),
                                        store_path=self.storing_folder)
                
            if self.store_switch:
                ac.store_activecortex(os.path.join(self.storing_folder, self.id_name, ac.cond_name))
        return

    def get_subtractions(self):
        
        # All possible subtraction dictionary building
        default_time_window = 20
        single_pos          = list(set([i for v in self.retino_pos_am.values() for i in v]))
        dict_components_    = self.retino_pos_am
        for i in single_pos:
            dict_components_[i] = [i]
        
        dict_subs   = utils.find_subsets(dict_components_)     

        # METHOD BUILT ON THE LINES ABOVE. CHECK UTILS
        utils.stampa(f'Dictionary of subtractions: {dict_subs}', logger = self.log)                                                               
        dict_subtrs = dict()

        for first_cond, second_cond in dict_subs.items():

            behavior_dict = self.dict_autoselection.get(first_cond, {})
            peaks = self.peaks_distribution.get(first_cond, None)

            # Provide the control retinotopic position in case of glitch in peak detection

            first_cd  = self.data[first_cond]  
            second_cd = self.data[second_cond]  
            
            t10, t11  = ((self.stimulus_metadata['multiple stroke']['bottom limit'], 
                          self.stimulus_metadata['multiple stroke']['bottom limit'] + default_time_window))
            
            if second_cd in self.cond_pos:
                t20, t21  = ((self.stimulus_metadata['single stroke']['bottom limit'], 
                              self.stimulus_metadata['single stroke']['bottom limit'] + default_time_window))
            else:
                t20, t21  = ((self.stimulus_metadata['multiple stroke']['bottom limit'], 
                              self.stimulus_metadata['multiple stroke']['bottom limit'] + default_time_window))
            
            name_subtrcts = f'{first_cond}-{second_cond}'

            a             = self.stimulus_metadata['pos metadata']
            space_step    = a[first_cond]['inter stimulus space']
            time_stepping = int(np.ceil((1/self.stimulus_metadata['speed'])*(space_step*(len(dict_components_[first_cond])-1))*self.acquisition_frequency)) 
            frames_start  = time_stepping + a[first_cond]['start']
            utils.stampa(f'Name sub {name_subtrcts}, space stepping {space_step}', logger = self.log)   

            # Time window twice the regular stroke stepping
            frames_end            = frames_start + 2*time_stepping
            utils.stampa(f'Frame start {frames_start} and end {frames_end}', logger = self.log)                                                               

            ac           = ActiveCortex(None, None, None,
                                        blank_cd = self.blank_cd,
                                        pixel_spacing = self.pixel_spacing,
                                        peaks_distribution = peaks,
                                        behavior_dict = behavior_dict, 
                                        logger = self.log)
            
            ac.cond_name         = name_subtrcts
            first_filtered_data  = ac.apply_spatial_filter(first_cd, median_kernel = self.filter_kernel, gaussian_sigma = self.gaussian_kernel)
            second_filtered_data = ac.apply_spatial_filter(second_cd, median_kernel = self.filter_kernel, gaussian_sigma = self.gaussian_kernel)
            zscored_first_data   = ac.compute_zscore(np.nanmean(first_filtered_data, axis = 0))
            zscored_second_data  = ac.compute_zscore(np.nanmean(second_filtered_data, axis = 0)) 
            sub_conds            = zscored_first_data[t10:t11, :, :] - zscored_second_data[t20:t21, :, :] 
            ac.map               = ac.compute_map(sub_conds, time_window = (frames_start,frames_end))
            ac.blob_binary, ac.blob_values = ac.compute_blob(ac.map, threshold=self.threshold)

            dict_subtrs[f'{first_cond}-{second_cond}'] = ((ac, first_cd, second_cd))     
        return dict_subtrs


class ActiveCortex:
    """
    Condition-level object that merges retinotopy & trial data.

    Expected data shape: (n_trials, n_timepoints, ny, nx)

    Attributes created/updated by methods:
      - filtered_data: same shape as raw_data after median+gaussian spatial filtering
      - zscored_data: same shape after z-scoring (via md.get_zscore)
      - map: 2D map (ny, nx) averaging across trials and time-window
      - time_window_used: (t0, t1) used for the map (python-style slice [t0:t1])
      - blob_binary: boolean mask of map > threshold
      - blob_values: map values where blob_binary True, NaN (or 0) elsewhere
      - behavior: dict with 'blk_names', 'autoselection' (bool array),
                  'corrects' (bool array), 'intersection' (bool array)
    """

    def __init__(self,
                 cond_name,
                 data,
                 time_window,
                 pixel_spacing = .07,
                 peaks_distribution = None,
                 blank_cd = None,
                 onset_time = 20, # In frames
                 statistical_threshold = 2.5,
                 behavior_dict = None,
                 logger = None):
        self.cond_name = cond_name
        self.raw_data = np.asarray(data, dtype=np.float32) if data is not None else None # keep original copy
        self.time_window = time_window  # expected (begin_frame, end_frame)
        self.pixel_spacing = pixel_spacing
        self.peaks_distribution = peaks_distribution
        self.blank_cd = blank_cd
        self.statistical_threshold = statistical_threshold
        self.onset_time = onset_time
        self.behavior_dict = behavior_dict or {}
        self.logger = logger

        # placeholders
        self.active_surface = None
        self.filtered_data = None
        self.zscored_data = None
        self.zscore_cd = None
        self.map = None
        self.time_window_used = None
        self.blob_binary = None
        self.blob_values = None
        self.behavior = None
        self.maps = None
        self.peaks = None
        self.blobs = None
        self.time_courses = None

    def store_activecortex(self, t):
        tp = [self.cond_name, 
              self.time_window, 
              self.peaks_distribution, 
              self.statistical_threshold, 
              self.onset_time,
              self.behavior_dict, 
              self.map, 
              self.zscore_cd,
              self.maps,
              self.blobs,
              self.time_window_used, 
              self.blob_binary, 
              self.blob_values, 
              self.time_courses,
              self.behavior,
              self.pixel_spacing,
              self.active_surface]
        storage_path = os.path.join(t, 'activecortex')
        tmp = dv.set_storage_folder(name_analysis = os.path.join(storage_path,))
        utils.inputs_save(tp, os.path.join(tmp,'ac_'+self.cond_name))
        return

    def load_activecortex(self, path):
        normalized_path = utils.normalize_path_os(path)
        tp = utils.inputs_load(normalized_path)

        self.cond_name             = tp[0]
        self.time_window           = tp[1]
        self.peaks_distribution    = tp[2] 
        self.statistical_threshold = tp[3]
        self.onset_time       = tp[4]
        self.behavior_dict    = tp[5]
        self.map              = tp[6]
        self.zscore_cd        = tp[7]
        self.maps             = tp[8]
        self.blobs            = tp[9]
        self.time_window_used = tp[10] 
        self.blob_binary      = tp[11]
        self.blob_values      = tp[12]
        self.time_courses     = tp[13]
        self.behavior         = tp[14]
        self.pixel_spacing    = tp[15]
        self.active_surface   = tp[16]

        return

    # --- logging helper ---
    def _log(self, msg):
        # prefer utils.stampa (keeps consistent with your code base)
        try:
            utils.stampa(msg, logger=self.logger)
        except Exception:
            # fallback
            if self.logger and hasattr(self.logger, 'info'):
                self.logger.info(msg)
            else:
                print(msg)

    # --- filtering ---
    def apply_spatial_filter(self, raw_data, median_kernel: int = 3, gaussian_sigma: float = 1.0):
        """
        Apply median filter (spatial) then gaussian filter (spatial) on last two dims.
        median_kernel: integer kernel size (will be forced odd if even).
        gaussian_sigma: sigma in pixels for gaussian filter (if <=0 gaussian is skipped).
        """
        if median_kernel is None:
            median_kernel = 3
        if median_kernel < 1:
            raise ValueError("median_kernel must be >= 1")

        # median filter kernels are usually odd — force odd and inform user
        if median_kernel % 2 == 0:
            median_kernel += 1
            self._log(f"[{self.cond_name}] median_kernel even -> bumped to odd: {median_kernel}")

        # convert to float32 for filtering
        arr = np.asarray(raw_data, dtype=np.float32)

        # apply median filter only on last two axes (ny, nx) without mixing trials/time:
        # median_filter accepts a tuple 'size' that matches array ndim
        if len(arr.shape) == 3:
            size  = (1, median_kernel, median_kernel)
            sigma = (float(gaussian_sigma), float(gaussian_sigma), float(gaussian_sigma))
        elif len(arr.shape) == 4:
            size  = (1, 1, median_kernel, median_kernel)
            sigma = (0, float(gaussian_sigma), float(gaussian_sigma), float(gaussian_sigma))

        arr = median_filter(arr, size=size, mode='reflect')
        arr = gaussian_filter(arr, sigma=sigma, mode='reflect')

        return arr
    
    # --- zscore using your existing md.get_zscore ---
    def compute_zscore(self, raw_data = None, blank_cd = None):
        """
        Calls md.get_zscore(filtered_data, mean_blank_forz, std_blank, logger)
        Requires mean_blank_forz and std_blank to be provided at init (precomputed).
        """

        if raw_data is None:
            raw_data = self.raw_data
        if self.blank_cd is None and blank_cd is None:
            raise RuntimeError(f"[{self.cond_name}] blank condition must be provided to compute zscore.")
        
        if blank_cd is None:
            blank_cd = self.blank_cd

        self._log(f"[{self.cond_name}] computing z-score using provided blank mean/std")
        if len(raw_data.shape) == 3:
            tmp_blnk = np.nanmean(blank_cd, axis = 0)
            z_cond   = process.zeta_score(raw_data, 
                                          np.nanmean(tmp_blnk, axis = 0), 
                                          np.nanstd(tmp_blnk, axis = 0)/np.sqrt(tmp_blnk.shape[0]), 
                                          full_seq=True)
            z_cond   = z_cond -  np.nanmean(z_cond)

        elif len(raw_data.shape) == 4:
            z_cond = np.array([process.zeta_score(j, 
                                                  np.nanmean(blank_cd, axis = 0), 
                                                  np.nanstd(blank_cd, axis = 0)/np.sqrt(blank_cd.shape[0]), full_seq=True) for j in raw_data])
            tmp_mean = np.nanmean(z_cond, axis = (-3, -2, -1))
            z_cond = z_cond - tmp_mean[:, np.newaxis, np.newaxis, np.newaxis]
        return z_cond

    # --- map computation ---
    def compute_map(self, zscored_data, time_window = None):
        """
        Average z-scored data across (trials, frames in time window) to form a 2D map.
        If time_window is None, uses self.time_window, if that is None uses entire time axis.
        Sets self.time_window_used to the actual slice (t0, t1).
        """
        if zscored_data is None:
            raise RuntimeError(f"[{self.cond_name}] zscored_data is None — run compute_zscore first.")

        nt = zscored_data.shape[0]
        tw = time_window if time_window is not None else self.time_window
        if tw is None:
            t0, t1 = 0, nt
        else:
            t0, t1 = int(tw[0]), int(tw[1])

        # clip to bounds
        t0 = max(0, t0)
        t1 = min(nt, t1)
        if t1 <= t0:
            raise ValueError(f"[{self.cond_name}] invalid time window after clipping: {(t0, t1)}")

        sel = zscored_data[t0:t1, :, :]
        # mean across trials and time (axis 0 and 1)
        with np.errstate(invalid='ignore'):
            self._log(f"Map over frames {t0}:{t1} (shape of selected data {sel.shape})")
            map = np.nanmean(sel, axis=0)
        self.time_window_used = (t0, t1)
        self._log(f"[{self.cond_name}] computed map over frames {t0}:{t1} (shape {map.shape})")
        return map

    # --- blob (threshold mask) ---    
    def compute_blob(self, map_, threshold = None, keep_values_nan=True):
        """
        Compute blob masks from maps:
          - blob_binary: boolean mask (map > threshold)
          - blob_values: map values where mask True; NaN (or 0) elsewhere
        If threshold is None uses self.statistical_threshold.
        """
        thr = self.statistical_threshold if threshold is None else float(threshold)
        self.statistical_threshold = thr

        # mask: ignore NaNs
        mask = np.isfinite(map_) & (map_ > thr)
        blob_binary = mask
        if keep_values_nan:
            vals = np.where(mask, map_, np.nan)
        else:
            vals = np.where(mask, map_, 0.0)
        blob_values = vals

        n_pixels = np.sum(mask)
        self._log(f"[{self.cond_name}] blob computed with threshold={thr} -> {int(n_pixels)} pixels selected")
        self.active_surface = process.get_active_surface(mask, self.pixel_spacing)
        self._log(f"[{self.cond_name}] surface in mm^2: {process.get_active_surface(mask, self.pixel_spacing):.3f}/{process.get_active_surface(np.ones((mask.shape)), self.pixel_spacing):.3f}")
        return blob_binary, blob_values


    # --- behavior extraction ---
    def extract_behavior(self):
        """
        Build intersection between 'corrects' and 'autoselection' while preserving blk_names order.
        Expects behavior_dict keys: 'blk_names', 'autoselection' (list of 0/1), 'corrects' (list/array of bools).
        Stores in self.behavior = {
            'blk_names': [...],
            'autoselection': np.bool_,
            'corrects': np.bool_,
            'intersection': np.bool_
        }
        """
        d = self.behavior_dict or {}
        blk_names = d.get('blk_names', None)
        autoselection = d.get('autoselection', None)
        corrects = d.get('corrects', None)

        if blk_names is None:
            # If blk_names missing, we still try to proceed but warn
            warnings.warn(f"[{self.cond_name}] behavior_dict has no 'blk_names'; behavior arrays may be unaligned.")
            blk_names = []

        if autoselection is None:
            raise KeyError(f"[{self.cond_name}] behavior_dict must contain 'autoselection' (list of 0/1).")
        if corrects is None:
            raise KeyError(f"[{self.cond_name}] behavior_dict must contain 'corrects' (list/array of bools).")

        autoselect_arr = np.asarray(autoselection)
        # convert to boolean: treat nonzero as True
        if autoselect_arr.dtype.kind in {'i', 'u', 'f'}:
            autoselect_bool = autoselect_arr != 0
        else:
            # try to parse strings like '0'/'1'
            autoselect_bool = np.array([bool(int(x)) for x in autoselect_arr], dtype=bool)

        corrects_bool = np.asarray(corrects, dtype=bool)

        # length checks: we require same length
        if len(autoselect_bool) != len(corrects_bool):
            raise ValueError(f"[{self.cond_name}] length mismatch: autoselection ({len(autoselect_bool)}) vs corrects ({len(corrects_bool)})")

        if blk_names and len(blk_names) != len(autoselect_bool):
            # warn but allow (order must be preserved — better to raise)
            raise ValueError(f"[{self.cond_name}] length mismatch: blk_names ({len(blk_names)}) vs trials ({len(autoselect_bool)})")

        intersection = np.logical_and(autoselect_bool, corrects_bool)

        self.behavior = {
            'blk_names': blk_names,
            'autoselection': autoselect_bool,
            'corrects': corrects_bool,
            'intersection': intersection
        }
        self._log(f"[{self.cond_name}] behavior extracted ({int(intersection.sum())} selected trials)")
        return self.behavior

    def compute_behavior_maps(self, keep_blob_nan=True):
        """
        Compute three maps (all/correct/incorrect), their peaks distributions,
        and blobs (binary masks above statistical threshold).
        """
        if self.filtered_data is None:
            raise RuntimeError(f"[{self.cond_name}] zscored_data is None — run compute_zscore first.")
        if self.behavior is None or 'corrects' not in self.behavior:
            raise RuntimeError(f"[{self.cond_name}] behavior info missing — run extract_behavior first.")
        if self.peaks_distribution is None:
            raise RuntimeError(f"[{self.cond_name}] peaks_distribution missing — pass it at init.")

        nt     = self.filtered_data.shape[1]
        t0, t1 = self.time_window_used if self.time_window_used else (0, nt)

        map_all = self.map

        mask = self.behavior['corrects']
        if mask is None or len(mask) != self.filtered_data.shape[0]:
            raise ValueError(f"[{self.cond_name}] behavior mask length mismatch with trial count.")
        

        # Subset peaks distributions
        peaks_x, peaks_y = map(np.asarray, self.peaks_distribution)
        peaks_all = (peaks_x, peaks_y)

        # Further median filter on contours before plotting them
        self.blob_binary   = median_filter(self.blob_binary, size=(5,5))

        if sum(mask) != len(self.filtered_data): 
            sel_correct   = self.compute_zscore(np.nanmean(self.filtered_data[mask], axis = 0))
            sel_correct   = sel_correct[t0:t1, :, :]
            map_correct   = np.nanmean(sel_correct, axis=0) 
            peaks_correct = (peaks_x[mask], peaks_y[mask])
            blob_correct, _   = self.compute_blob(map_correct, keep_values_nan = keep_blob_nan, threshold = self.statistical_threshold) 

            sel_incorrect   = self.compute_zscore(np.nanmean(self.filtered_data[~mask], axis = 0))
            sel_incorrect   = sel_incorrect[t0:t1, :, :]
            map_incorrect   = np.nanmean(sel_incorrect, axis=0) 
            peaks_incorrect = (peaks_x[~mask], peaks_y[~mask])
            blob_incorrect, _  = self.compute_blob(map_incorrect, keep_values_nan = keep_blob_nan, threshold = self.statistical_threshold) 

            # Further median filter on contours before plotting them
            blob_correct    = median_filter(blob_correct, size=(5,5))
            blob_incorrect  = median_filter(blob_incorrect, size=(5,5))

        else:
            map_correct   = None
            peaks_correct = None
            blob_correct  = None

            map_incorrect   = None
            peaks_incorrect = None
            blob_incorrect  = None


        results = {
            'map': {
                'all': map_all,
                'correct': map_correct,
                'incorrect': map_incorrect,
            },
            'peaks': {
                'all': peaks_all,
                'correct': peaks_correct,
                'incorrect': peaks_incorrect,
            },
            'blob': {
                'all': self.blob_binary,
                'correct': blob_correct,
                'incorrect': blob_incorrect,
            }
        }

        self._log(f"[{self.cond_name}] behavior maps, peaks, and blobs computed")
        return results

    def compute_blob_timecourse(self):
        """
        Extract time courses from blob region for each behavioral conditioblob computed with thresholdn.
        Returns time series data for all trials, correct trials, and incorrect trials.
        """
        if self.zscored_data is None:
            raise RuntimeError(f"[{self.cond_name}] zscored_data is None — run compute_zscore first.")
        if self.behavior is None or 'corrects' not in self.behavior:
            raise RuntimeError(f"[{self.cond_name}] behavior info missing — run extract_behavior first.")
        if self.blob_binary is None:
            raise RuntimeError(f"[{self.cond_name}] blob_binary is None — compute blob first.")
        
        # Get behavioral mask
        mask = self.behavior['corrects']
        if mask is None or len(mask) != self.filtered_data.shape[0]:
            raise ValueError(f"[{self.cond_name}] behavior mask length mismatch with trial count.")
        
        # Compute z-scores for each trial (not averaged)
        z_all = self.zscored_data
        
        # Separate correct and incorrect trials
        z_correct   = z_all[mask] if np.any(mask) else np.array([])
        z_incorrect = z_all[~mask] if np.any(~mask) else np.array([])
        
        # Extract time courses from blob region
        def extract_blob_timecourse(z_data):
            """Extract mean signal from blob region across time for each trial"""
            if z_data.size == 0:
                return np.array([])
            
            center_activation, _ = process.find_highest_sum_area(self.map, 30)
            y_sh, x_sh  = self.map.shape
            print(f'Center for blob tc analysis: [{center_activation[0]}, {center_activation[1]}]')
            tc_mask = utils.sector_mask((y_sh, x_sh), center_activation, 15, (0, 360))    
            timecourses = np.array([process.time_course_signal(i, abs(tc_mask - 1)) for i in z_data])

            # mask = self.blob_binary  # shape (180, 218)

            # timecourses = []
            # for trial in z_data:  # trial shape = (65, 180, 218)
            #     # Apply mask to spatial dims, keep time
            #     masked_pixels = trial[:, mask]  # shape (65, n_masked_pixels)
            #     blob_timecourse = np.nanmean(masked_pixels, axis=1)  # mean over pixels
            #     timecourses.append(blob_timecourse)
            
            return np.array(timecourses)

        # Extract time courses for each condition        
        self._log(f'{z_all.shape} {z_correct.shape} {z_incorrect.shape} {self.blob_binary.shape}')

        tc_all       = extract_blob_timecourse(z_all)

        if (len(z_correct) != 0) and (len(z_incorrect) != 0):
            tc_correct   = extract_blob_timecourse(z_correct)
            tc_incorrect = extract_blob_timecourse(z_incorrect)
        else:
            tc_correct   = []
            tc_incorrect = []
        
        results = {
            'timecourse': {
                'all': tc_all,
                'correct': tc_correct,
                'incorrect': tc_incorrect,
            },
            'n_trials': {
                'all': len(z_all),
                'correct': len(z_correct) if z_correct.size > 0 else 0,
                'incorrect': len(z_incorrect) if z_incorrect.size > 0 else 0,
            }
        }
        
        self._log(f"[{self.cond_name}] blob timecourses computed - "
                f"all: {results['n_trials']['all']}, "
                f"correct: {results['n_trials']['correct']}, "
                f"incorrect: {results['n_trials']['incorrect']} trials")
        
        return results

    # --- convenience: run the whole pipeline for this condition ---
    def run_full_pipeline(self,
                          median_kernel: int = 3,
                          gaussian_sigma: float = 1.0,
                          threshold: float = None,
                          keep_blob_nan: bool = True,
                          compute_behavior: bool = True):
        """
        Convenience wrapper to run everything in order:
          1) spatial filters
          2) z-score
          3) map
          4) blob (threshold)
          5) behavior extraction (if behavior_dict provided)
        """
        self.filtered_data = self.apply_spatial_filter(self.raw_data, median_kernel = median_kernel, gaussian_sigma = gaussian_sigma)
        self.zscored_data  = self.compute_zscore(self.filtered_data)
        self.zscore_cd     = self.compute_zscore(np.nanmean(self.filtered_data, axis = 0))
        self.map           = self.compute_map(self.zscore_cd)
        self.blob_binary, self.blob_values = self.compute_blob(self.map, threshold=threshold, keep_values_nan=keep_blob_nan)

        if compute_behavior:
            self.extract_behavior()
        return self

    def __repr__(self):
        return f"<ActiveCortex cond={self.cond_name} map_shape={None if self.map is None else self.map.shape}>"

def get_md_files(path_to_derivatives, list_conds, behavior_flag = True, get_trial_mask = True, get_md_data = True):
    dict_data = {}
    dict_autoselection = {}

    for name_cond in list_conds:
        cd = md.Condition()
        cd.cond_name = name_cond
        cd.load_cond(os.path.join(path_to_derivatives, 'md_data','md_data_'+name_cond))

        if get_trial_mask:
            autoselection  = cd.autoselection
            blk_names      = cd.blk_names
            trial_metadata = cd.trials if behavior_flag else None
            dict_autoselection[name_cond] = {'autoselection': autoselection,
                                            'blk_names': blk_names,
                                            'trials': trial_metadata}

            corrects = [cd.trials[blk].correct_behav for blk in blk_names] if behavior_flag else None
            dict_autoselection[name_cond]['corrects'] = corrects 
        if get_md_data:
            p_raw  = cd.binned_data 
            dict_data[name_cond] = p_raw 
        del cd
    return dict_data, dict_autoselection


def plot_blob_timecourse(timecourse_results, time_series_info, y_lim = None, name_cond = '', title_plot=None, name_analysis_ = 'RetinotopicPositions', store_path = dv.STORAGE_PATH, store_pic = True, ext = '.png'):
    """
    Plot time course data from blob region analysis.
    
    Parameters:
    -----------
    timecourse_results : dict
        Output from compute_blob_timecourse method
    time_series_info : tuple
        (zero, time_interval, time_bins) for time axis
    title_plot : str, optional
        Title for the plot
    fig : matplotlib.figure.Figure, optional
        Figure to add axes to
    fig_h : float, optional
        Figure height for axes positioning
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure(figsize=(10, 8))
    
    zero, time_interval, time_bins = time_series_info
    ax_time = fig.add_axes([0.1, 0.15, 0.8, .85])
    ax_time.spines[['top', 'right']].set_visible(False)
    
    # Plot each condition
    colors = {'all': 'black', 'correct': 'green', 'incorrect': 'red'}
    alphas = {'all': 0.3, 'correct': 0.2, 'incorrect': 0.2}
    
    # Determine time axis
    if (zero is not None) and (time_interval is not None) and (time_bins is not None):
        x_tc = np.arange(-zero*time_interval, (time_bins*time_interval)-zero*time_interval, time_interval)
    else:
        # Use the length of the first available time series
        for condition in ['all', 'correct', 'incorrect']:
            if timecourse_results['timecourse'][condition].size > 0:
                x_tc = np.arange(len(timecourse_results['timecourse'][condition][0]))
                break
    
    # Collect all data for y-axis limits
    all_data = []
    
    for condition, color in colors.items():
        time_series_data = timecourse_results['timecourse'][condition]
        
        if time_series_data.size == 0:
            continue
            
        all_data.extend(time_series_data.flatten())
        
        # Plot confidence band and mean
        ax_time.fill_between(x_tc,
                        np.nanpercentile(time_series_data, 95, axis=0),
                        np.nanpercentile(time_series_data, 5, axis=0), 
                        color=color, alpha=alphas[condition])
        
        ax_time.plot(x_tc, np.nanmean(time_series_data, axis=0), 
                    label=f'{condition.capitalize()} (n={timecourse_results["n_trials"][condition]})', 
                    color=color, lw=3)
    
    # Add vertical line at time zero
    if all_data:
        if y_lim is None:
            y_min, y_max = np.nanpercentile(all_data, 15), np.nanpercentile(all_data, 95)
        else:
            y_min, y_max = y_lim
        ax_time.vlines(0, y_min, y_max, ls='--', lw=2, color='gold')
        ax_time.set_ylim(y_min, y_max)
    
    # Formatting
    if title_plot is not None:
        ax_time.set_title(f'{title_plot}', fontsize=20)
    ax_time.tick_params(axis='both', which='major', labelsize=16)
    ax_time.set_xlabel('Time - ms', fontsize=18)
    ax_time.set_ylabel('Z-score Signal', fontsize=18)
    ax_time.legend()

    if store_pic:
        # Storing picture
        tmp = dv.set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)
        plt.savefig(os.path.join(tmp, f'blob_tc_analysis_{name_cond}{ext}'))
        plt.close('all')
    else:
        plt.show()
    return 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Launching retinotopy analysis pipeline')

    parser.add_argument('--path_md', 
                        dest='path_md',
                        type=str,
                        required=True,
                        help='The middle process datafolder path')
    
    parser.add_argument('--threshold', 
                        dest='threshold',
                        type=float,
                        default=2, #zscore
                        required=False,
                        help='Statistically significant threshold for blobs')

    parser.add_argument('--med_kernel', 
                        dest='median_kernel',
                        type=int,
                        default=3, #pixels
                        required=False,
                        help='Spatial kernel for median filter')
    
    parser.add_argument('--gaus_kernel', 
                        dest='gaussian_kernel',
                        type=float,
                        default=1, #std in pixels
                        required=False,
                        help='Spatial kernel (std) for gaussian filter')
    
    parser.add_argument('--denoised', 
                        dest='denoise_flag',
                        type=bool,
                        default= True,
                        required=False,
                        help='Switch for denoised data or regular dF/F0')
    
    parser.add_argument('--store', 
                        dest='store_flag',
                        type=bool,
                        default= True,
                        required=False,
                        help='Switch for storing output data')
    
    parser.add_argument('--visualize', 
                        dest='vis_switch',
                        type=bool,
                        default= True,
                        required=False,
                        help='Switch for visualizing output data')
    
    parser.add_argument('--spatial_filter', 
                        dest='spatial_filter_switch',
                        type=bool,
                        default= True,
                        required=False,
                        help='Switch for filtering data (gaussian and median)')

    start_process_time = datetime.datetime.now().replace(microsecond=0)

    log = utils.setup_custom_logger('myapp')  
    args = parser.parse_args()
    utils.stampa(f'{args}', logger = log)      

    session_deriv   = args.path_md
    session_acs     = ActiveCortexSession(session_deriv, 
                                          threshold = args.threshold,
                                          spatial_filter= args.spatial_filter_switch,
                                          store_flag=args.store_flag, 
                                          denoise_flag = args.denoise_flag, 
                                          filter_kernel = args.median_kernel,
                                          gaussian_kernel = args.gaussian_kernel,
                                          vis_switch = args.vis_switch,
                                          logger=log)

    # tmp_blnk        = np.nanmean(session_acs.data['blank'], axis =0)    
    utils.stampa(f'Active cortex analysis for session {session_acs.id_name} elaborated in {datetime.datetime.now().replace(microsecond=0)-start_process_time}!\n', logger=log)                                

    start_process_time_cds = datetime.datetime.now().replace(microsecond=0)
    session_acs.build_conditions()
    session_acs.build_sub_conditions()
    utils.stampa(f'Conditions for active cortex analysis processed in {datetime.datetime.now().replace(microsecond=0)-start_process_time_cds}!\n', logger=log)                                

    tmp_name_cd = list(conds.keys())
    ac = conds[tmp_name_cd[0]]
    print(ac.map.shape, ac.blob_binary.sum(), ac.behavior['intersection'].sum())
    utils.stampa(f'Analysis elaborated in {datetime.datetime.now().replace(microsecond=0)-start_process_time}!\n', logger=log)                                
