import argparse, datetime
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
                 logger          = None,  
                 store_switch    = False,
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
                 **kwargs):

        if logger is None:
            self.log = utils.setup_custom_logger('myapp')
        else:
            self.log = logger     

        self.path_to_derivatives   = path_session
        self.path_session          = path_session.split('derivatives')[0]

        self.acquisition_frequency = acquisition_fq #Hz
        self.time_bin              = (1/self.acquisition_frequency)*1000 #ms
        self.green                 = utils.get_green(green_name, self.path_session, log=self.log)

        # Get original data shape 
        try:
            self.original_frame_shape = self.green.shape
        except:
            self.original_frame_shape = (1312, 1312)

        self.denoise_switch        = denoise_flag
        self.vis_switch            = vis_switch
        self.store_switch          = store_switch
        self.pretriggerz_flag      = pretrigger_zero
        self.filter_flag           = spatial_filter # Does not work for zmaps_flag
        self.filter_kernel         = filter_kernel # Does not work for zmaps_flag
        
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

        self.data_loader        = retino.RetinoLoaderManager(self.path_session,  flag_denoise=self.denoise_switch, storage_path=self.retin_folder, pretrigger_flag = self.pretriggerz_flag)

        self.session_id_name    = self.data_loader.id_name 
        self.retino_data        = self.data_loader.data  # dict of condition -> RetinotopicObject(s)
        self.time_window_per_cd = self.get_time_window()
        utils.stampa(f'{self.time_window_per_cd}', logger=self.log)

        self.list_conds      = list(self.data_loader.cond_am) + list(self.data_loader.cond_pos) + [blank_name] 

        self.data, self.dict_autoselection = get_md_files(self.path_to_derivatives, self.list_conds, behavior_flag = trial_metadata_flag, get_md_data = (not self.denoise_switch))
        
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

        _, self.nt, self.ny, self.nx = self.data[blank_name].shape
        self.spatial_bin     = np.nanmax(self.original_frame_shape)/np.nanmax([self.ny, self.nx])  #Import green and import an md file and check the difference in frame shape
        self.pixel_spacing   = self.spatial_bin*(cortical_dim*(optical_ratio))/np.nanmax(self.original_frame_shape)     

        
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

    def build_conditions(self,
                         mean_blank_forz,
                         std_blank,
                         threshold = 2.5,
                         median_kernel = 3,
                         gaussian_sigma = 1.0,
                         keep_blob_nan = True,
                         compute_behavior = True):
        """
        Build an ActiveCortex object for each condition in self.list_conds.
        Returns a dict: cond_name -> ActiveCortex instance.
        """
        conditions = {}
        for cond_name in self.list_conds:
            # safety checks
            if cond_name not in self.data:
                utils.stampa(f"[build_conditions] skipping {cond_name}: no data in self.data", logger=self.log)
                continue

            data = self.data[cond_name]
            tw = self.time_window_per_cd.get(cond_name, None)
            behavior_dict = self.dict_autoselection.get(cond_name, {})

            ac = ActiveCortex(cond_name=cond_name,
                              data=data,
                              time_window=tw,
                              mean_blank_forz=mean_blank_forz,
                              std_blank=std_blank,
                              statistical_threshold=threshold,
                              behavior_dict=behavior_dict,
                              logger=self.log)
            # run pipeline
            ac.run_full_pipeline(median_kernel=median_kernel,
                                 gaussian_sigma=gaussian_sigma,
                                 threshold=threshold,
                                 keep_blob_nan=keep_blob_nan,
                                 compute_behavior=compute_behavior)

            conditions[cond_name] = ac

        return conditions


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
                 mean_blank_forz=None,
                 std_blank=None,
                 statistical_threshold = 2.5,
                 behavior_dict = None,
                 logger = None):
        self.cond_name = cond_name
        self.raw_data = np.asarray(data, dtype=np.float32)  # keep original copy
        self.time_window = time_window  # expected (begin_frame, end_frame)
        self.mean_blank_forz = mean_blank_forz
        self.std_blank = std_blank
        self.statistical_threshold = statistical_threshold
        self.behavior_dict = behavior_dict or {}
        self.logger = logger

        # placeholders
        self.filtered_data = None
        self.zscored_data = None
        self.map = None
        self.time_window_used = None
        self.blob_binary = None
        self.blob_values = None
        self.behavior = None

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
    def apply_spatial_filter(self, median_kernel: int = 3, gaussian_sigma: float = 1.0):
        """
        Apply median filter (spatial) then gaussian filter (spatial) on last two dims.
        median_kernel: integer kernel size (will be forced odd if even).
        gaussian_sigma: sigma in pixels for gaussian filter (if <=0 gaussian is skipped).
        """
        if median_kernel is None:
            median_kernel = 1
        median_kernel = int(median_kernel)
        if median_kernel < 1:
            raise ValueError("median_kernel must be >= 1")

        # median filter kernels are usually odd — force odd and inform user
        if median_kernel % 2 == 0:
            median_kernel += 1
            self._log(f"[{self.cond_name}] median_kernel even -> bumped to odd: {median_kernel}")

        # convert to float32 for filtering
        arr = np.asarray(self.raw_data, dtype=np.float32)

        # apply median filter only on last two axes (ny, nx) without mixing trials/time:
        # median_filter accepts a tuple 'size' that matches array ndim
        size = (1, 1, median_kernel, median_kernel)
        if median_kernel > 1:
            self._log(f"[{self.cond_name}] applying median filter (kernel={median_kernel})")
            arr = median_filter(arr, size=size, mode='reflect')
        else:
            self._log(f"[{self.cond_name}] skipping median filter (kernel={median_kernel})")

        # gaussian filter: use sigma per-axis so we don't blur across trials/time axes
        if gaussian_sigma is not None and gaussian_sigma > 0:
            sigma = (0, float(gaussian_sigma), float(gaussian_sigma), float(gaussian_sigma))
            self._log(f"[{self.cond_name}] applying gaussian filter on nt, ny and nx (sigma={gaussian_sigma})")
            arr = gaussian_filter(arr, sigma=sigma, mode='reflect')
        else:
            self._log(f"[{self.cond_name}] skipping gaussian filter (sigma={gaussian_sigma})")

        self.filtered_data = arr
        return self.filtered_data

    # --- zscore using your existing md.get_zscore ---
    def compute_zscore(self):
        """
        Calls md.get_zscore(filtered_data, mean_blank_forz, std_blank, logger)
        Requires mean_blank_forz and std_blank to be provided at init (precomputed).
        """
        if self.filtered_data is None:
            raise RuntimeError(f"[{self.cond_name}] filtered_data is None — run apply_spatial_filter first.")
        if self.mean_blank_forz is None or self.std_blank is None:
            raise RuntimeError(f"[{self.cond_name}] mean_blank_forz and std_blank must be provided to compute zscore.")

        self._log(f"[{self.cond_name}] computing z-score using provided blank mean/std")
        # md.get_zscore is expected to handle the full array shape and return z-scored data
        self.zscored_data = md.get_zscore(self.filtered_data, self.mean_blank_forz, self.std_blank, logger=self.logger)
        return self.zscored_data

    # --- map computation ---
    def compute_map(self, time_window = None):
        """
        Average z-scored data across (trials, frames in time window) to form a 2D map.
        If time_window is None, uses self.time_window, if that is None uses entire time axis.
        Sets self.time_window_used to the actual slice (t0, t1).
        """
        if self.zscored_data is None:
            raise RuntimeError(f"[{self.cond_name}] zscored_data is None — run compute_zscore first.")

        nt = self.zscored_data.shape[1]
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

        sel = self.zscored_data[:, t0:t1, :, :]
        # mean across trials and time (axis 0 and 1)
        with np.errstate(invalid='ignore'):
            self.map = np.nanmean(sel, axis=(0, 1))
        self.time_window_used = (t0, t1)
        self._log(f"[{self.cond_name}] computed map over frames {t0}:{t1} (shape {self.map.shape})")
        return self.map

    # --- blob (threshold mask) ---
    def compute_blob(self, threshold: float = None, keep_values_nan: bool = True):
        """
        Compute blob masks from self.map:
          - blob_binary: boolean mask (map > threshold)
          - blob_values: map values where mask True; NaN (or 0) elsewhere
        If threshold is None uses self.statistical_threshold.
        """
        if self.map is None:
            raise RuntimeError(f"[{self.cond_name}] map is None — run compute_map first.")
        thr = self.statistical_threshold if threshold is None else float(threshold)
        self.statistical_threshold = thr

        # mask: ignore NaNs
        mask = np.isfinite(self.map) & (self.map > thr)
        self.blob_binary = mask
        if keep_values_nan:
            vals = np.where(mask, self.map, np.nan)
        else:
            vals = np.where(mask, self.map, 0.0)
        self.blob_values = vals

        n_pixels = np.sum(mask)
        self._log(f"[{self.cond_name}] blob computed with threshold={thr} -> {int(n_pixels)} pixels selected")
        return self.blob_binary, self.blob_values

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
        self.apply_spatial_filter(median_kernel=median_kernel, gaussian_sigma=gaussian_sigma)
        self.compute_zscore()
        self.compute_map()
        self.compute_blob(threshold=threshold, keep_values_nan=keep_blob_nan)
        if compute_behavior:
            self.extract_behavior()
        return self

    def __repr__(self):
        return f"<ActiveCortex cond={self.cond_name} map_shape={None if self.map is None else self.map.shape}>"


def get_md_files(path_to_derivatives, list_conds, behavior_flag = True, get_trial_mask = True, get_md_data = True, logger = None):
    dict_data = {}
    dict_autoselection = {}
    for name_cond in list_conds:    
        utils.stampa(f'{name_cond} cd process starts...')
        cd           = md.Condition()
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
                        default=5, #zscore
                        required=False,
                        help='Spatial kernel for median filter')
    
    parser.add_argument('--gaus_kernel', 
                        dest='gaussian_kernel',
                        type=float,
                        default=1.5, #zscore
                        required=False,
                        help='Spatial kernel (std) for gaussian filter')
    

    start_process_time = datetime.datetime.now().replace(microsecond=0)

    log = utils.setup_custom_logger('myapp')  
    args = parser.parse_args()
    utils.stampa(f'{args}', logger = log)      

    session_deriv   = args.path_md

    session_acs     = ActiveCortexSession(session_deriv, denoise_flag = True, logger=log)
    mean_blank_forz = np.nanmean(session_acs.data['blank'], axis = (0, 1))
    std_blank       = np.nanstd(session_acs.data['blank'], axis = (0, 1))/np.sqrt(session_acs.data['blank'].shape[1])
    conds           = session_acs.build_conditions(mean_blank_forz, std_blank,
                                                   threshold=args.threshold,
                                                   median_kernel=args.median_kernel,
                                                   gaussian_sigma=args.gaussian_kernel)
    tmp_name_cd = list(conds.keys())
    ac = conds[tmp_name_cd[0]]
    print(ac.map.shape, ac.blob_binary.sum(), ac.behavior['intersection'].sum())
    utils.stampa(f'Active cortex analysis for session {session_acs.id_name} elaborated in {datetime.datetime.now().replace(microsecond=0)-start_process_time}!\n', logger=log)                                
