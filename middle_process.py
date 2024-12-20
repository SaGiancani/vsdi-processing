import argparse, blk_file, datetime, utils
import process_vsdi as process
import data_visualization as dv
import ana_logs as al
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from scipy.io import savemat

LABEL_CONDS_PATH = 'metadata/labelConds.txt' 

class Condition:
    """
    Initializes attributes
    Default values for:
    *session_header: dict. It contains some of the metainformation of the session: it's a subset of the initial args parser, 
                     if the middle_process script is ran on terminal. See get_session_header method for details 
    *session_name: str. Unique string identifier, obtained with subject name, experiment and session date
    *cond_name: str. The name of the condition.
    *cond_id: int. The condition id number.
    *binned_data: numpy.array. The tensor with the raw binned data. It has shape: (Trials, Time, Y, X)  
    *df_fz: numpy.array. The tensor with the preprocessed data: it is the deltaF/F0. It has shape (Trials, Time, Y, X)
    *time_course: numpy.array. The matrix with the deltaF/F0 averaged over a circular ROI on the center of the frame. 
                  It has shape (Trials, Time)
    *averaged_df: numpy.array. The tensor with the average over trials of the deltaF/F0. It has shape (Time, X, Y)
    *averaged_timecourse: numpy.array. The array with the average over trials of the timecourse computed averaging
                          over a circular ROI on the center of the frame. It has shape (Time)
    *autoselection: list. It is a list of ones and zeros: 1 at the corresponding index of selected good trials, and 0 
                    for not selected ones. It is used for masking the trials.
    *blk_names: list. List of strings, with the all the BLK filenames of the condition.
    *trials: dict. It is a dict of ana_logs.Trial objects. The keys of the dictionary are the strings of blk_names, and the 
             values are Trial objects: these represent wrappers of BaseReport and other log files metadata.  
    *z_score: numpy.array. It is a tensor of the z_score for the condition, computed over the deltaF/F0 of the different
              trials, respect the blank condition.

    Parameters
    ----------
    condition_name : str
        The name of the condition. None as default
    condition_numb: int
        The id number for the condition. None as default
    session_header: dict
        The header of the corresponding session: it is useful for instancing important parameters.
    """
    def __init__(self, condition_name = None, condition_numb = None, session_header = None):
        self.session_header = session_header
        if self.session_header is None:
            self.session_name = None
        else:
            comp = os.path.normpath(self.session_header['path_session']).split(os.sep)
            self.session_name = comp[-2].split('sub-')[1]+'-'+comp[-3].split('exp-')[1] + '_' + comp[-1].split('-')[1] 
        self.cond_name = condition_name
        self.cond_id = condition_numb
        self.binned_data = None 
        self.df_fz = None
        self.time_course = None
        self.averaged_df = None
        self.averaged_timecourse = None
        self.autoselection = None
        self.blk_names = None
        self.trials = None
        self.z_score = None
    
    def store_cond(self, t):
        '''
        Storing method. All the parameters are wrapped within a list.
        Built-in storage folder md_data within derivatives. The pickle file takes the name md_data_cond_name
        '''
        tp = [self.session_header, self.session_name, self.cond_name, self.cond_id, self.binned_data, self.df_fz, self.time_course, self.averaged_df, self.averaged_timecourse, self.autoselection, self.blk_names, self.trials, self.z_score]
        utils.inputs_save(tp, os.path.join(t,'md_data','md_data_'+self.cond_name))
        return
    
    def load_cond(self, path):
        '''
        Loading method. All the parameters are unwrapped from a list.
        It can be problematic from versioning.
        '''
        tp = utils.inputs_load(path)
        self.session_header = tp[0]
        self.session_name = tp[1]
        self.cond_name = tp[2]
        self.cond_id = tp[3]
        self.binned_data = tp[4]
        self.df_fz = tp[5]
        self.time_course = tp[6]
        self.averaged_df = tp[7]
        self.averaged_timecourse = tp[8]
        self.autoselection = tp[9]
        self.blk_names = tp[10]
        self.trials = tp[11]
        try:
            self.z_score = tp[12]
        except:
            print('z_score attribute not found')
        return
    
    def get_behav_latency(self, blank_id):
        '''
        Behavioral latency time collection method. If the trial was with correct fixation and it is not belonging to
        blank condition, than the behavioral latency is stored, for computing the mean and the standard error. 
        '''
        tmp = [trial.behav_latency for trial in self.trials.values() if (trial.fix_correct and self.cond_id != blank_id)]
        tmp_ = [i for i in tmp if i>0]
        return float(np.mean(tmp_)), float(np.std(tmp_)/np.sqrt(len(tmp_))), tmp

    def get_success_rate(self):
        '''
        The method computes the rate of success of the behavioral task: it returns mean and standard error of the rate.
        '''
        tmp = [1 if trial.correct_behav else 0 for trial in self.trials.values()]
        return float(np.mean(tmp)), float(np.std(tmp)/np.sqrt(len(tmp))), tmp
    
    def get_orientation_behav(self):
        '''
        The method stores the behavioral outcome for each trial of the condition.
        '''
        return (([trial.orientation for trial in self.trials.values()], [trial.orientation_outcome for trial in self.trials.values()]))


# Inserting inside the class variables and features useful for one session: we needs an object at this level for
# keeping track of conditions, filenames, selected or not flag for each trial.
class Session:
    def __init__(self,
                 path_session, 
                 spatial_bin = 3,
                 temporal_bin = 1,
                 zero_frames = None,
                 tolerance = 20,
                 mov_switch = False,
                 deblank_switch = False,
                 detrend_switch = False,
                 conditions_id =None,
                 chunks = 1,
                 strategy = 'mae',
                 logs_switch =False,  
                 base_report_name= 'BaseReport.csv',
                 base_head_dim = 19, 
                 logger = None, 
                 condid = None, 
                 store_switch = False, 
                 matlab_switch = False,
                 data_vis_switch = True, 
                 end_frame = None,
                 filename_particle = 'vsd_C', 
                 **kwargs):
        """
        Initializes attributes
        Default values for:
        *log: logging.Formatter. Logger for log file creation. Useful for background running. If it is not given as input,
              it is instanciated.
        *cond_names: list. List of strings with the condition names of the considered conditions. Check the method
                     get_condition_name.
        *header: dict. It contains some of the metainformation of the session: it's a subset of the initial args parser, 
                 if the middle_process script is ran on terminal. See get_session_header method for details 
        *all_blks: list. All the .BLK files contained inside path_session/rawdata/. It is a list of strings. See get_all_blks method
                   See get_all_blks method for more details.
        *cond_dict: dict. Dictionary of the conditions: it has condition id numbers as keys and condition names as values. 
                    See get_condition_name for more details.
        *blank_id: int. Integer value, representing the blank id number. If it is not given as input, it extracts automatically from the
                   labelConds file the index corresponding to "blank" condition.
        *session_blks: list. All the .BLK, per condition, considered for the processing, either selected and not selected. It is a subset 
                       of all_blks. It is a list of strings
        *trials_name: list. Only the selected .BLK files, per condition, considered for the processing. The selection is performed by an
                      autoselection algorithm. Check get_selection_trials method for more hints.
        *auto_selected: list. It is a list of ones and zeros: 1 at the corresponding index of selected good trials, and 0 
                        for not selected ones. It is used for masking the trials.
        *conditions: list. It is a list of integers: it is a control variable, it should contain a list of integers all the same, 
                     corresponding to the condition id number. It should have the same length of session_blks.
        *counter_blank: int. Security value for keeping track of the storage of blank signal.
        *visualization_switch: bool. A boolean used as switch for figure storing processing. True for storing, False for not.
        *storage_switch: bool. A boolean used as switch for Condition object storing. True for storing, false for not.
        *avrgd_time_courses: numpy.array. An array which is the average over trials of the time courses obtained averaging over a circular 
                             ROI on the center of the frame. It has shape (Conditions, Time)
        *avrgd_df_fz: numpy.array. A tensor which is the average over trials, for each condition, of the deltaF/F0. It has shape 
                      (Conditions, Time, Y, X)
        *z_score: numpy.array. A tensor which is the zscore of the deltaF/F0, computed with the blank signal.
        *base_report: pandas.DataFrame. The logfile BaseReport. It contains all the timing for each trial, stored and not. It performs a 
                      safety polishing on not matching BLK filenames and trials, using the condition number presents on the filename and 
                      on each row of the logfile. 
        *piezo: list. List of lists. Each list is a trial.
        *heart_beat: list. List of lists. Each list is a trial.
        *time_course_blank: numpy.array. Same as avrgd_time_courses, but only for the blank condition. Used for deblanking conditions 
                            different from blank. None by default. See get_signal with blank_id as input case, for more details. It has 
                            shape (Time)
        *f_f0_blank: numpy.array. Same as avrgd_df_fz, but only for blank condition. Used for deblanking. It has shape (Time, Y, X)
        *stde_f_f0_blank: numpy.array. The standard error over blank condition trials. It has shape (Time, Y, X)

        Parameters
        ----------
        path_session: str
            The path of the session. It has to be the folder containing rawdata, metadata, derivatives, source.
        spatial_bin: int
            The spatial binning value. 3 as default
        temporal_bin: int
            The temporal binning value. 1 as default
        zero_frames: int
            The number of frames considered as zero. Value used in deblanking and deltaF/F0 computing.
        tolerance: int
            DEPRECATED
        mov_switch: bool
            DEPRECATED
        deblank_switch: bool
            Switch for deblanking or not. False as default.
        conditions_id: list
            It is a list of considered conditions. None as default: if it's not, all the conditions are automatically considered.
        chunks: int
            Number of chunks, for chunk strategy in autoselection algorithm. 1 by default
        strategy: str
            Strategy used for autoselection analysis. mae by default: together with mse are the ones considering the number of chunks
        logs_switch: bool
            Switch for logfiles wrapping. False by default.
        base_report_name: str
            BaseReport name. BaseReport.csv by default. BaseReport_.csv also common
        base_head_dim: int
            Number of rows within the header is present: 19 by default.
        logger: logging.Formatter
            Logger for printing. None by default
        condid: int
            Blank id number. None by default and automatic extraction with method get_blank_id
        store_switch: bool
            Switch for Condition objects storing.
        data_vis_switch: bool
            Switch for storing of figures of processing and preprocessing.
        end_frame: int
            The index of the considered ending frame. None by defaults
        """
        if logger is None:
            self.log = utils.setup_custom_logger('myapp')
        else:
            self.log = logger
        self.detrend_switch = detrend_switch     
        self.matlab_switch = matlab_switch                       
        self.cond_names = None
        self.header = self.get_session_header(path_session, spatial_bin, temporal_bin, tolerance, mov_switch, deblank_switch, conditions_id, chunks, strategy, logs_switch)
        self.all_blks = get_all_blks(self.header['path_session'], sort = True) # all the blks, sorted by creation date -written on the filename-.
        if len(self.all_blks) == 0:
            print('Check the path: no blks found')
        self.cond_dict = self.get_condition_name()
        self.cond_names = list(self.cond_dict.values())
        self.blank_id = get_blank_id(self.cond_names, cond_id=condid)
        self.filename_particle = filename_particle

        # This can be automatized, with zero_frames, extracting parameters from BaseReport
        # Avoiding to load a BLK file
        # A blk loaded for useful hyperparameters
        blk = blk_file.BlkFile(os.path.join(self.header['path_session'],'rawdata', self.all_blks[np.random.randint(len(self.all_blks)-1)]), 
                               self.header['spatial_bin'], 
                               self.header['temporal_bin'],
                               detrend_switch    = self.detrend_switch,
                               filename_particle = self.filename_particle)
        tmp = blk.header['nframesperstim']
        print(f'n. frames header {tmp}')
        print(f'n. frames signal {blk.signal.shape[0]}')
        print(f'n. frames binned signal {blk.binned_signal.shape[0]}')
        self.header['n_frames'] = blk.header['nframesperstim']
        self.header['original_height'] = blk.header['frameheight']
        self.header['original_width'] = blk.header['framewidth']
        
        # Setting key frames
        if end_frame is None:
            self.header['ending_frame'] = int(round(self.header['n_frames']*0.9))
        else:
            self.header['ending_frame'] = end_frame

        if zero_frames is None:
            self.header['zero_frames'] = int(round(self.header['n_frames']*0.1))
        else:
            self.header['zero_frames'] = zero_frames

        # If considered conditions are not explicitly indicated, then all the conditions are considered
        # The adjustment of conditions_id set has to be done ALWAYS before the session_blks extraction       
        if self.header['conditions_id'] is None:
            self.header['conditions_id'] = get_condition_ids(self.all_blks, filename_particle = self.filename_particle)
        else:
            self.header['conditions_id'] = list(set(self.header['conditions_id']+[self.blank_id]))
        # only the used blks for the selection

        if self.header['mov_switch']:
            self.motion_indeces = None
        
        self.session_blks = None
        self.trials_name = None 
        self.auto_selected = None
        self.conditions = None
        self.counter_blank = 0  
        
        self.visualization_switch = data_vis_switch
        self.storage_switch = store_switch

        self.avrgd_time_courses = None
        self.avrgd_df_fz = None
        self.z_score = None

        # Loading the BaseReport and SignalData in case of logs_switch
        if self.header['logs_switch']:
            try:
                start_time = datetime.datetime.now().replace(microsecond=0)
                self.log.info(f'Length of all_blks list: {len(self.all_blks)}')
                base_report, _ = al.get_basereport(self.header['path_session'], self.all_blks, name_report = base_report_name, header_dimension = base_head_dim)
                # Separator converter processing: , -string- to . -float-.
                for i in list(base_report.columns):
                    try:
                        base_report[[i]] = base_report[[i]].applymap(al.separator_converter)
                    except:
                        print(f'Column {i} is not a float')
                base_report, count = al.discrepancy_blk_attribution(base_report)
                print(f'Mismatch for {count} blk files')                
                self.base_report = base_report
                # Check in case of presence of BLK file with no correspondance in BaseReport
                # In case of presence, they are removed from all_blks
                security_check_blks = set(self.all_blks).difference(set(list(self.base_report.loc[self.base_report['Preceding Event IT'] == 'FixCorrect','BLK Names'])))
                if len(security_check_blks) != 0:
                    self.log.info(f'Length of all_blks before popping off the elements: {len(self.all_blks)}')
                    for j in list(security_check_blks):
                        self.all_blks.remove(j)
                        self.log.info(f'{j} popped out')
                    self.log.info(f'Length of all_blks list after popping off from get_basereport: {len(self.all_blks)}')

                self.log.info('BaseReport properly loaded!')
                self.log.info('BaseReport loading time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
                start_time = datetime.datetime.now().replace(microsecond=0)
                self.time_stamp, self.piezo, self.heart_beat, self.toogle, self.triginstim, ((self.starting_times, self.ending_times)), self.affidability = al.get_analog_signal(self.header['path_session'], self.base_report, name_report = 'SignalData.csv')
                self.log.info('Piezo and Heart Beat signals properly loaded!')
                self.log.info('Analogic signals loading time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
                self.base_report = self.base_report.loc[(self.base_report['Preceding Event IT'] == 'FixCorrect')] 
            except:
                self.log.info('Something went wrong loading the BaseReport or SignalData')
                self.base_report, self.time_stamp, self.piezo, self.heart_beat,   = None, None, None, None
                self.toogle, self.triginstim, ((self.starting_times, self.ending_times)), self.affidability  = None, None, (None, None), None
            #try:
            #    path_trackreport = utils.find_thing('TrackerLog.csv', self.header['path_session'])
            #except:
            #    self.log('Something went wrong loading the TrackerLog')
        else:
            self.base_report, self.time_stamp, self.piezo, self.heart_beat  = None, None, None, None
            self.toogle, self.triginstim, ((self.starting_times, self.ending_times)), self.affidability  = None, None, (None, None), None

        self.time_course_blank = None
        self.f_f0_blank = None
        self.stde_f_f0_blank = None
        if self.header['deblank_switch']:
        # TO NOTICE: deblank_switch add roi_signals, df_fz, auto_selected, conditions, counter_blank and overwrites the session_blks
            # Calling get_signal in the instantiation of Session allows to obtain the blank signal immediately.
            _ = self.get_signal(self.blank_id)


    def get_signal(self, condition):
        '''
        Parameters:
            self: This parameter refers to the instance of the class Session. 
            It is used to access instance variables and methods within Session class.
            condition: An integer representing the condition for which the signal is being extracted.

        Method Description:
            This method is responsible for extracting signals and related data for a specific condition.
            It starts by filtering a list of "blks" that match the provided condition.
            It defines "zero_of_cond" and "end_of_cond" based on attributes in the Session header.
            Depending on whether the condition is the "blank_id" (the blank condition), it follows different logic paths.
            If the condition is the "blank_id":
                It extracts signals, data frames, conditions, and other information using the signal_extraction function.
                It processes and stores various data, including the signal, normalized signal, and z-scores.
                Visualization and other operations are performed if specified.
                If storage is enabled, a "Condition" object is instantiated and stored.
            If the condition is not the "blank_id":
                Similar signal extraction and processing are performed as in the "blank_id" case, but some data is 
                appended to existing class attributes. Visualization and storage are handled in the same way.
            Various data and variables are deleted from memory to free up resources.
            The method returns a binary mask based on a selection criterion for trials.
            
        '''
        # All the blank blks
        blks = [f for f in self.all_blks if (int(f.split(self.filename_particle)[1][0:2])==condition)]
        zero_of_cond = self.header['zero_frames']
        end_of_cond = self.header['ending_frame']
        # Blank signal extraction
        self.log.info(f'Trials of condition {condition} loading starts:')
        if condition == self.blank_id:
            sig, df_f0, conditions, raws, trials, blks = signal_extraction(self.header, blks, None, self.header['deblank_switch'], self.base_report, self.blank_id, self.time_stamp, self.piezo, self.heart_beat, detrend = self.detrend_switch, filename_particle = self.filename_particle)
            size_df_f0 = np.shape(df_f0)
            # For sake of storing coherently, the F/F0 has to be demeaned: dF/F0. 
            # But the one for normalization is kept without demean
            sig = sig - 1
            self.counter_blank = size_df_f0[0]
            mask = self.get_selection_trials(condition, sig)
            self.conditions = conditions
            self.auto_selected = mask
            self.session_blks = blks
            indeces_select = np.where(self.auto_selected==1)
            indeces_select = indeces_select[0].tolist()      
            # In this order for deblank signal
            tmp = np.nanmean(df_f0[indeces_select, :, :, :], axis=0)
            tmp_std = np.nanstd(df_f0[indeces_select, :, :, :], axis=0)
            self.f_f0_blank = tmp
            self.stde_f_f0_blank = tmp_std/np.sqrt(len(indeces_select))
            z = process.zeta_score(df_f0[indeces_select, :, :, :], self.f_f0_blank, self.stde_f_f0_blank)
            # Signal for zscore: it is the raw signal binned, only normalized for the average frame among the zero frames.
            print(f'Shape z_score {z.shape}')
            self.z_score = np.reshape(z, (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
            # Subtraction for 1 equivalent to deblanking (F0) -dF/F0-
            df_f0 = df_f0 - 1
            self.avrgd_df_fz = np.reshape(np.nanmean(df_f0[indeces_select, :, :, :], axis=0), (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
            # Average time course over the condition
            tmp_ = np.nanmean(sig[indeces_select, :], axis=0)
            self.avrgd_time_courses = np.reshape(tmp_, (1, tmp.shape[0]))
            # It's important that 1 is not subtracted to this blank_df: it is the actual blank signal
            # employed for normalize the signal             
            self.time_course_blank = tmp_
            self.log.info('Blank signal computed')
                        
        else:
            sig, df_f0, conditions, raws, trials, blks = signal_extraction(self.header, blks, self.f_f0_blank, self.header['deblank_switch'], self.base_report, self.blank_id, self.time_stamp, self.piezo, self.heart_beat, detrend= self.detrend_switch, filename_particle = self.filename_particle)
            mask = self.get_selection_trials(condition, sig)
            self.conditions = self.conditions + conditions
            self.auto_selected = np.array(self.auto_selected.tolist() + mask.tolist(), dtype=int)
            self.session_blks = self.session_blks + blks
            indeces_select = np.where(np.array(mask)==1)
            indeces_select = indeces_select[0].tolist()
            #df_f0 = df_f0.reshape(1, df_f0.shape[1], df_f0.shape[2], df_f0.shape[3] ) 
            t =  np.nanmean(df_f0[indeces_select, :, :, :], axis=0)
            self.avrgd_df_fz = np.concatenate((self.avrgd_df_fz, t.reshape(1, t.shape[0], t.shape[1], t.shape[2])), axis=0) 
            self.log.info(f'Shape averaged dF/F0: {np.shape(self.avrgd_df_fz )}')
            t_ =  np.nanmean(sig[indeces_select, :], axis=0)
            self.avrgd_time_courses = np.concatenate((self.avrgd_time_courses,  t_.reshape(1,  t_.shape[0])), axis=0) 
            self.log.info(f'Shape averaged tc: {np.shape(self.avrgd_time_courses )}')
            if self.base_report is not None:
                zero_of_cond = int(np.nanmean([v.zero_frames for v in trials.values()]))
                foi_of_cond = int(np.nanmean([v.FOI for v in trials.values()]))
                print('Average Prestimulus time: ') 
                print(np.nanmean([v.onset_stim - v.start_stim for v in trials.values()]))
                end_of_cond = zero_of_cond + foi_of_cond
            temp_raw = raws[indeces_select, :, :, :]
            t_ = np.array([process.deltaf_up_fzero(i, zero_of_cond, deblank = True, blank_sign=None) for i in temp_raw])
            #z = process.zeta_score(self.avrgd_df_fz[-1, :, :, :], None, None, full_seq = True)
            z = process.zeta_score(t_, self.f_f0_blank, self.stde_f_f0_blank)
             #def zeta_score(sig_cond, sig_blank, std_blank, zero_frames = 20):

            self.z_score = np.concatenate((self.z_score, z.reshape(1, z.shape[0], z.shape[1], z.shape[2])), axis=0) 

        #def deltaf_up_fzero(vsdi_sign, n_frames_zero, deblank = False, blank_sign = None, outlier_tresh = 1000):
        if self.visualization_switch:
            self.roi_plots(condition, sig, mask, blks)
            self.log.info(f'Zero frames {zero_of_cond}, n° of considered frames {20} and end of frames {int((end_of_cond))}')
            dv.time_sequence_visualization(zero_of_cond, 20, end_of_cond, df_f0[indeces_select, :, :, :], np.array(blks)[indeces_select], 'cond'+str(condition), self.header, self.set_md_folder(), log_ = self.log, max_trials = 20)

        # If storage switch True, than a Condition object is instantiate and stored
        if self.storage_switch:
            start_time = datetime.datetime.now().replace(microsecond=0)
            cond = Condition(self.cond_dict[condition], condition, self.header)
            cond.binned_data = raws
            cond.df_fz = df_f0
            cond.time_course = sig
            cond.autoselection = mask
            cond.blk_names = blks
            cond.averaged_df = self.avrgd_df_fz[-1, :, :, :]
            cond.z_score = self.z_score[-1, :, :, :]
            cond.averaged_timecourse = self.avrgd_time_courses[-1, :]
            if self.base_report is not None:
                cond.trials = trials
            #Storing folder
            t = self.set_md_folder()
            if not os.path.exists(os.path.join(t,'md_data')):
                os.makedirs(os.path.join(t,'md_data'))
            cond.store_cond(t)

            if self.matlab_switch:
                cond_name  = self.cond_dict[condition] 
                cond_name  = cond_name.replace('-', '')
                cond_name  = cond_name.replace('°', '')
                cond_name  = cond_name.replace('.', '')
                path_store_2mat = dv.set_storage_folder(storage_path = t, name_analysis = 'pickle2MAT')
                
                path_tmp_raw    = os.path.join(path_store_2mat , f'raw_{cond_name}.mat')
                path_tmp_df     = os.path.join(path_store_2mat , f'dFF0_{cond_name}.mat')
                path_tmp_sel    = os.path.join(path_store_2mat , f'selection_mask_{cond_name}.mat')

                savemat(path_tmp_raw, {f'raw_{cond_name}': raws}, format='5')                
                savemat(path_tmp_df, {f'dFF0_{cond_name}': df_f0}, format='5')                
                savemat(path_tmp_sel, {f'sel_mask_{cond_name}': self.auto_selected}, format='5')      

            del cond
            self.log.info('Storing condition time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))                
        del df_f0
        del raws
        del sig
        del trials
        return mask

    def get_blks(self):
        '''
        The .BLKs filenames corresponding to the choosen id conditions, from the considered path_session, are picked.        
        '''
#        if self.session_blks is None:
        #This condition check is an overkill
        if ((self.header['conditions_id'] is None) or (len(self.header['conditions_id']) == len(self.cond_names))):
            self.log.info('BLKs for all conditions sorted by time creation')
            return self.all_blks
        else:
            self.log.info('BLKs for conditions ' + str(self.header['conditions_id']) + 'sorted by time creation')
            tmp = [f for f in self.all_blks if (int(f.split(self.filename_particle)[1][0:2]) in self.header['conditions_id'])]
            try:
                a = sorted(tmp, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
            except:
                self.log.info('Warning: sorting BLK filenames was not performed')
                a = tmp
            return a 
        
    def get_condition_name(self):
        '''
        The method returns a list of condition's names: if a labelConds.txt exist inside metadata's folder, 
        than names are simply loaded: this method is the fastest. Otherwise it looks for labelConds.txt in 
        the whole session path: this is slower. In case of no existence of labelConds file a dictionary of 
        names with "Condition #" style is built.
        The output of the method is a dictionary with key the id condition and value the condition name.
        '''
        # Try the fastest method, in other words looking from the file in metadata folder
        try:
            with open(os.path.join(self.header['path_session'], LABEL_CONDS_PATH)) as f:
                contents = f.readlines()
            return  {j+1:i.split('\n')[0] for j, i in enumerate(contents) if len(i.split('\n')[0])>0}
        # If does not find it, then look in general in session folder if there is the labelConds.txt file -slower-
        except:
            tmp = utils.find_thing('labelConds.txt', self.header['path_session'])
            # If also with find_thing there is no labelConds.txt file, than loaded as name Condition n#
            if len(tmp) == 0:
                self.log.info('Check the labelConds.txt presence inside the session folder and subfolders')
                cds = get_condition_ids(self.all_blks, filename_particle = self.filename_particle)
                return {j+1:'Condition ' + str(c) for j, c in enumerate(cds)}
            # else, load the labelConds from the alternative path
            else :
                with open(tmp[0]) as f:
                    contents = f.readlines()
                return  {j+1:i.split('\n')[0] for j, i in enumerate(contents) if len(i.split('\n')[0])>0}

    def get_session_header(self, path_session, spatial_bin, temporal_bin, tolerance, mov_switch, deblank_switch, conditions_id, chunks, strategy, logs_switch):
        header = {}
        header['path_session'] = path_session
        header['spatial_bin'] = spatial_bin
        header['temporal_bin'] = temporal_bin
        header['tolerance'] = tolerance
        header['mov_switch'] = mov_switch
        header['deblank_switch'] = deblank_switch
        header['detrend_switch'] = self.detrend_switch
        header['conditions_id'] = conditions_id
        header['chunks'] = chunks
        header['strategy'] = strategy
        header['logs_switch'] = logs_switch
        return header
    
    def get_session(self):
        '''
        Parameters:
            self: This parameter represents the instance of the class Session and is used to access instance 
            variables and methods.

        Method Description:
            This method is part of a data processing workflow for an experimental VSDI session, handling 
            multiple experimental conditions.
            If there are multiple conditions defined (as indicated by the length of 'conditions_id' in the class header), 
            the method iterates through each condition (denoted as 'cd').
            For each condition, it logs the start of the loading procedure and the condition name.
            It then calls the get_signal method for the current condition, which is responsible for signal 
            extraction and related processing. 
            After processing, it logs the number of trials selected for the condition.
            The method keeps track of the total number of trials selected across all conditions.
            It also constructs an array of trial names based on the selected trials.
            If the visualization_switch is enabled (a boolean attribute), it generates time sequence visualizations
            for the data. This includes visualizations of averaged conditions and z-scores.
            The method returns without any explicit return value.        
        '''
        if len(self.header['conditions_id']) > 1:
            for cd in self.header['conditions_id']:
                c_name = self.cond_dict[cd]
                if cd != self.blank_id:
                    self.log.info('Procedure for loading BLKs of condition ' +str(cd)+' starts')
                    self.log.info('Condition name: ' + c_name)                        
                    tmp = self.get_signal(cd)
                    self.log.info(str(int(sum(tmp))) + '/' + str(len(tmp)) +' trials have been selected for condition '+str(c_name))
                    
            self.log.info('Globally ' + str(int(sum(self.auto_selected))) + '/' + str(len(self.session_blks)) +' trials have been selected!')
            session_blks = np.array(self.session_blks)
            self.trials_name = session_blks[self.auto_selected]

        if self.visualization_switch:
            # zero_frames and ending_frames have to be recovered by trials
            # titles gets the name of blank condition as first, since it was stored first
            for i, j in enumerate(self.avrgd_df_fz):
                print(j.shape)
                # dF/F0
                dv.whole_time_sequence(j, 
                                       mask = np.ones((j[0, :, :].shape), dtype=bool),
                                       name=f'df_average_cond{i+1}', 
                                       max=80, min=20,
                                       handle_lims_blobs = ((97.72, 100)),
                                       #significant_thresh = np.percentile(dict_retino[name_cond].signal, 97.72), 
                                       # global_cntrds = [dict_retino[name_cond].retino_pos],
                                       # colors_centr = colrs,
                                       ext='png',
                                       name_analysis_= os.path.join(self.set_md_folder(), 'activity_maps'))

                dv.whole_time_sequence(self.z_score[i], 
                                       mask = np.ones((j[0, :, :].shape), dtype=bool),
                                       name=f'zscore_cond{i+1}', 
                                       max=80, min=20,
                                       handle_lims_blobs = ((97.72, 100)),
                                       #significant_thresh = np.percentile(dict_retino[name_cond].signal, 97.72), 
                                       # global_cntrds = [dict_retino[name_cond].retino_pos],
                                       # colors_centr = colrs,
                                       ext='png',
                                       name_analysis_= os.path.join(self.set_md_folder(), 'activity_maps'))
                # dv.time_sequence_visualization(self.header['zero_frames'], 
                #                                20,
                #                                self.header['ending_frame'], 
                #                                self.avrgd_df_fz, 
                #                                [self.cond_dict[self.blank_id]]+[self.cond_dict[c] for c in self.header['conditions_id'] if c!=self.blank_id] ,
                #                                'avrgd_conds', self.header, self.set_md_folder(), log_ = self.log)
                # #def time_sequence_visualization(start_frame, n_frames_showed, end_frame, data, titles, title_to_print, header, path_, circular_mask = True, log_ = None, max_trials = 20):
                # # Double deblanking: further blank subtraction here
                # dv.time_sequence_visualization(self.header['zero_frames'], 20, self.header['ending_frame'], self.z_score, [self.cond_dict[self.blank_id]]+[self.cond_dict[c] for c in self.header['conditions_id'] if c!=self.blank_id] , 'zscores', self.header, self.set_md_folder(), c_ax_ = (np.nanpercentile(self.z_score, 15), np.nanpercentile(self.z_score, 90)), log_ = self.log)

        else:
            self.log.info('No visualization charts.')
        return

    def get_selection_trials(self, condition, time_course):
        '''
        Parameters:
            self: This parameter represents the instance of the class Session and is used to access instance variables and methods.
            condition: An integer representing the condition for which selection trials are being determined.
            time_course: numpy.array 3d matrix representing the time courses for a specific condition.

        Method Description:
            This method is responsible for automatically selecting trials based on a specified strategy.
            It first retrieves various parameters from the class instance, such as the strategy, the number of frames, and other attributes.
            It logs the start of the autoselection process for the given condition.
            The selection strategy depends on the value of the strategy attribute in the class header. The following strategies are supported:
                + 'mse' or 'mae' (Mean Squared Error or Mean Absolute Error): In this case, the data is divided into chunks, and overlap strategy
                   is used to select trials. The number of chunks can be determined based on the number of frames -all the chunks of same number of frames- 
                   and the 'chunks' attribute in the class header.
                + 'roi', 'roi_signals', or 'ROI': This strategy is based on selecting trials using region of interest (ROI) information, with 
                  specified tolerance and zero frames.
                + 'statistic', 'statistical', or 'quartiles': A statistical strategy based on statistical measures is used 
                   to select trials.
            The selected trials are stored in the tmp variable.
            The method logs the time it took for the autoselection process for the given condition.
            It returns the selected trials (tmp) as the result of the method.        
        '''
        strategy = self.header['strategy']
        n_frames = self.header['n_frames']

        start_time = datetime.datetime.now().replace(microsecond=0)
        self.log.info(f'Autoselection for Condition: {condition}')
        if strategy in ['mse', 'mae']:
            self.log.info('Chunks division strategy choosen')
            if  n_frames%self.header['chunks']==0:
                nch = self.header['chunks']
            else:
                self.log.info('Warning: Number of chunks incompatible with number of frames, 1 trial = 1 chunk then is considered') 
                nch = 1
            # Condition per condition
            #tmp = np.zeros(np.shape(time_course)[0], dtype=int)
            #self.log.info(np.array(self.session_blks)[indeces.tolist()])
            #self.log.info(indeces)
            _, tmp, _, _, _  = overlap_strategy(time_course, condition, self.set_md_folder(), self.header,  switch_vis = self.visualization_switch, n_chunks=nch, loss = strategy)
            #indeces = np.arange(0, np.shape(time_course)[0], dtype=int)

        elif strategy in ['roi', 'roi_signals', 'ROI']:
            tmp = roi_strategy(time_course, self.header['tolerance'], self.header['zero_frames'])

        elif strategy in ['statistic', 'statistical', 'quartiles']:
            tmp = statistical_strategy(time_course)

        #self.log.info(np.array(self.conditions)[self.auto_selected])
        #self.log.info(self.trials_name)
        self.log.info('Autoselection loop time for condition ' +str(condition)+ ': ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
        return tmp

    def roi_plots(self, cd_i, sig, mask, blks):
        #session_name = self.header['path_session'].split('/')[-2]+'-'+self.header['path_session'].split('/')[-3].split('-')[1]
        comp = os.path.normpath(self.header['path_session']).split(os.sep)
        session_name = comp[-2].split('sub-')[1]+'-'+comp[-3].split('exp-')[1] + '_' + comp[-1].split('-')[1] 
        blank_sign = self.time_course_blank
        cdi_select = np.where(mask==1)
        cdi_select = cdi_select[0].tolist()
        cdi_unselect = np.where(mask==0)
        cdi_unselect = cdi_unselect[0].tolist()
        # Number of possible columns
        b = [4,5,6]
        a = [len(mask)%i for i in b]
        columns = b[a.index(min(a))]

        fig = plt.figure(constrained_layout=True, figsize = (columns*4, int(np.ceil(len(mask)/columns)+1)*2), dpi = 80)
        title = f'Condition #{cd_i}' 
        try:
            if self.cond_names is not None:
                title = title + ': ' + self.cond_names[cd_i-1]
        except:
            None
        fig.suptitle(title)# Session name
        # Height_ratios logic implementation
        rat = [1]*(int(np.ceil(len(mask)/columns))+1)
        rat[-1] = 3
        subfigs = fig.subfigures(nrows=int(np.ceil(len(mask)/columns))+1, ncols=1, height_ratios=rat)
        for row, subfig in enumerate(subfigs):
            #subfig.suptitle('Bottom title')
            axs = subfig.subplots(nrows=1, ncols=columns, sharex=True, sharey=True)
            x = np.arange(0, self.header['n_frames'])
            for i, ax in enumerate(axs):
                count = row*columns + i
                if count < len(mask):
                    ax.set_ylim(np.nanmin(sig[cdi_select, :]) - (np.nanmax(sig[cdi_select]) - np.nanmin(sig[cdi_select]))*0.005, 
                                np.nanmax(sig[cdi_select, :]) + (np.nanmax(sig[cdi_select]) - np.nanmin(sig[cdi_select]))*0.005)
                    if mask[count]==1:
                        color = 'b'
                    else:
                        color = 'r'
                    ax.plot(sig[count, :], color)
                    ax.set_title(blks[count])
                    ax.errorbar(x, np.nanmean(sig[cdi_select, :], axis = 0), yerr=(np.nanstd(sig[cdi_select, :], axis = 0)/np.sqrt(len(cdi_select))), 
                                fmt='--', color = 'k', elinewidth = 0.5)
                    ax.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
                    #ax.set_ylim(-0.002,0.002)
                if row<len(subfigs)-2:
                    ax.get_xaxis().set_visible(False)
                elif row<len(subfigs)-1:
                    ax.get_xaxis().set_visible(True)
                elif row == len(subfigs)-1:
                    ax.axis('off')
                    ax_ = subfig.subplots(1, 1)
                    ax_.set_ylim(np.nanmin(sig[cdi_select, :]) - (np.nanmax(sig[cdi_select]) - np.nanmin(sig[cdi_select]))*0.005, 
                                 np.nanmax(sig[cdi_select, :]) + (np.nanmax(sig[cdi_select]) - np.nanmin(sig[cdi_select]))*0.005)
                    for i in sig[cdi_select[:-1], :]:
                        ax_.plot(x, i, 'gray', linewidth = 0.5)
                    ax_.plot(x, sig[cdi_select[-1], :], 'gray', linewidth = 0.5, label = 'Trials')
                    ax_.plot(x, np.nanmean(sig[cdi_select, :], axis=0), 'k', label = 'Average Selected trials', linewidth = 2)
                    ax_.plot(x, np.nanmean(sig[cdi_unselect, :], axis=0), 'crimson', label = 'Average Unselected trials', linewidth = 2)
                    ax_.plot(x, np.nanmean(sig, axis=0), 'green', label = 'Average All trials Cond. ' + str(cd_i), linewidth = 2)
                    ax_.plot(x, blank_sign, color='m', label = 'Average Blank Signal' ,linewidth = 2)
                    #ax_.plot(list(range(0,np.shape(sig)[1])), blank_sign, color='m', label = 'Average Blank Signal' ,linewidth = 5)
                    ax_.legend(loc="upper left")                
                    ax_.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
                
        tmp = self.set_md_folder()
        if not os.path.exists(os.path.join(tmp,'time_course')):
            os.makedirs(os.path.join(tmp,'time_course'))
        plt.savefig(os.path.join(tmp,'time_course', session_name+'_tc_0'+str(cd_i)+'.png'))
        #plt.savefig((path_session+'/'session_name +'/'+ session_name+'_roi_0'+str(cd_i)+'.png')
        plt.close('all')
        return

    def set_md_folder(self):
        session_path = self.header['path_session']
        if self.header['strategy'] in ['mse', 'mae']: 
            strat_depend = '_strategy' + str(self.header['strategy']) + \
                '_n_chunk' + str(self.header['chunks'])
        elif self.header['strategy'] in ['roi', 'roi_signals', 'ROI']: 
            strat_depend = '_strategy' + str(self.header['strategy']) + \
                '_tol' + str(self.header['tolerance'])

        folder_name = 'spcbin' + str(self.header['spatial_bin']) \
            + '_timebin' + str(self.header['temporal_bin']) \
            + '_zerofrms' + str(self.header['zero_frames']) \
            + strat_depend\
            + '_mov' + str(self.header['mov_switch'])\
            + '_dtrend' + str(self.detrend_switch)\
            + '_deblank' + str(self.header['deblank_switch'])
        
        folder_path = os.path.join(session_path, 'derivatives/',folder_name)               
        if not os.path.exists(folder_path):
        #if not os.path.exists( path_session+'/'+session_name):
            os.makedirs(folder_path)
            #os.mkdirs(path_session+'/'+session_name)
        return folder_path
    
    def split_behav_blks(self, autoselection = False): 
        '''
        The method gets as input a condition and returns two lists of tuples, each tuple contains an index and corresponding blk filename.
        The two lists correspond one to correct behavior blks, and the second to uncorrect behavior.         
        Input:
            cond: Condition object.
            autoselection: boolean flag. If True returns an intersection of the correct/uncorrect indeces with the autoselected in the session.
        Output:
            2 lists of tuples.
        '''
        correct_blks_p1, uncorrect_blks_p1 = list(), list()
        indeces_correct, indeces_uncorrect = list(), list()
        for k,v in self.trials.items():
            if v.correct_behav:
                correct_blks_p1.append(k)
                indeces_correct.append(self.blk_names.index(k))
            else:
                uncorrect_blks_p1.append(k)
                indeces_uncorrect.append(self.blk_names.index(k))
        
        if autoselection:
            indeces_correct_ = list(set(indeces_correct).intersection(set(np.where(self.autoselection)[0])))
            correct_blks_p1 = [self.blk_names[i] for i in indeces_correct_]
            indeces_uncorrect_ = list(set(indeces_uncorrect).intersection(set(np.where(self.autoselection)[0]))) 
            uncorrect_blks_p1 = [self.blk_names[i] for i in indeces_uncorrect_]
            
        return [(self.blk_names.index(i), i) for i in correct_blks_p1], [(self.blk_names.index(i), i) for i in uncorrect_blks_p1]

def get_condition_ids(all_blks, filename_particle = 'vsd_C'):
    '''
    The method returns a list of all the condition's ids, taken from the .BLK names.
    '''
    return list(set([int(i.split(filename_particle)[1][0:2]) for i in all_blks]))

def get_blank_id(cond_names, cond_id = None):
    '''
    The method returns the index of blank condition.
    Some session require a specific condition index: cond_id variable designed for manual setting.
    If it is None -by default-, the method checks among the condition names: if labelConds.txt
    file exists, than the position of "blank" label is picked. Otherwise the position of last condition 
    is picked.
    '''
    if cond_id is None:
        try:
            tmp = [idx for idx, s in enumerate(cond_names) if 'blank' in s][-1]+1
            print('Blank id: ' + str(tmp))
            return tmp
        except IndexError:
            print('No clear blank condition was identified: the last condition has picked')
            tmp = len(cond_names)
            print('Blank id: ' + str(tmp))
            return tmp
    else:
        return cond_id

def signal_extraction(header, blks, blank_s, blnk_switch, base_report, blank_id, time, piezo, heart, detrend = False, log = None, blks_load = True, filename_particle = 'vsd_C'):
    '''
    Parameters:
        header: the Session header. A dictionary containing various parameters and metadata.
        blks: A list of trial (BLK) filenames to process.
        blank_s: The blank signal, aka a reference signal.
        blnk_switch: A switch that determines whether blank subtraction is applied.
        base_report: A report containing information about trials (could be None if not available).
        blank_id: The ID of the blank condition.
        time: Time information.
        piezo: Piezo data.
        heart: Heart data.
        log: A logging object for recording information (could be None).
        blks_load: A flag indicating whether to load BLK files.
        filename_particle: a string particle for distinguish between VSDI or IOI recordings.

    Function Description:
        The function begins by initializing some variables and parameters, including trials_dict, path_rawdata, and flag_remove.
        It determines the type of strategy for processing the trials, such as 'mse', 'mae', 'roi', 'statistic', and more, based 
        on the strategy attribute in the header.
        If blks_load is True, it loads the trial data. For each trial in blks, it does the following:
            Loads the trial data using the blk_file.BlkFile class and stores it in variables such as raws, delta_f, and sig.
            The specific processing applied to the data depends on the chosen strategy. For example, if the strategy is 'mse' or 
            'mae', the data is divided into chunks.
            The trial information, including condition, is stored in the conditions list.
            The function logs information about the loading process.
            If a trial is empty, it is removed from the list of trials.
        If blks_load is False, it doesn't load the trial data and only constructs the trials_dict.
        The function returns the extracted and processed data: sig (time course), delta_f (delta F/F0), conditions (conditions of 
        the trials), raws (raw data), and trials_dict (trial information).    
    '''
    #motion_indeces, conditions = [], []
    conditions = []
    path_rawdata = os.path.join(header['path_session'],'rawdata/')
    flag_remove = False
    if base_report is not None:
        trials_dict = dict()
        greys = al.get_greys(header['path_session'], int(os.path.join(path_rawdata, blks[0]).split(filename_particle)[1][0:2]))
    else:
        trials_dict = None
        
    if blks_load:

        if log is None:
            print(f'The blank_signal exist: {blank_s is not None}')
            print(f'The blank switch is: {blnk_switch}')
        else:
            log.info(f'The blank_signal exist: {blank_s is not None}')
            log.info(f'The blank switch is: {blnk_switch}')

        for i, blk_name in enumerate(blks):
            start_time = datetime.datetime.now().replace(microsecond=0)
            if base_report is not None:
                trial = al.get_trial(base_report, blk_name, time, heart, piezo, greys[1], greys[0], blank_id)
                # If the trial is empty, likely for absence of BLK name correspondance in BaseReport, it pops out the blkname from the list
                if trial is None:
                    print('Empty Trial')
                    blks.remove(blk_name)
                    flag_remove = True
                    if log is None:
                        print(f'{blk_name} was popped off')
                    else:
                        log.info(f'{blk_name} was popped off')
                # Otherwise store it
                elif trial is not None:
                    flag_remove = False
                    trials_dict[blk_name] = trial   
                    zero = trial.zero_frames
            else:
                zero = header['zero_frames']
            print(f'Employeed zero for normalization is {zero}')
            # If the blk name is not popped off, the blk is loaded
            if not flag_remove:    
                # Get BLK file
                # If first BLK file, than the header is stored
                if i == 0:
                    BLK = blk_file.BlkFile(
                        os.path.join(path_rawdata, blk_name),
                        header['spatial_bin'],
                        header['temporal_bin'],
                        header = None, 
                        detrend_switch    = detrend,
                        filename_particle = filename_particle)

                    header_blk = BLK.header
                    raws = np.empty((len(blks), header['n_frames'], header['original_height']//header['spatial_bin'], header['original_width']//header['spatial_bin']))
                    delta_f = np.empty((len(blks), header['n_frames'], header['original_height']//header['spatial_bin'], header['original_width']//header['spatial_bin']))
                    sig = np.empty((len(blks), header['n_frames']))
                    roi_mask = blk_file.circular_mask_roi(header['original_width']//header['spatial_bin'], header['original_height']//header['spatial_bin'])
                else:
                    BLK = blk_file.BlkFile(
                        os.path.join(path_rawdata, blk_name), 
                        header['spatial_bin'], 
                        header['temporal_bin'], 
                        header = header_blk,
                        detrend_switch    = detrend, 
                        filename_particle = filename_particle)
                
                # Log prints
                if log is None:
                    print(f'The blk file {blk_name} is loaded')
                else:
                    log.info(f'The blk file {blk_name} is loaded')
                    
                #at the end something like (nblks, 70, 1)         
                conditions.append(BLK.condition)
                BLK.binned_signal[np.where(BLK.binned_signal==0)] = np.nan
                
                # Sanity check for blks with wrong time dimension
                if BLK.binned_signal.shape[0] != raws.shape[1]:

                    if log is None:
                        print(f'Trial n. {i+1}/{len(blks)} {blk_name} mismatch with time shape of the session. Discarded')      
                        print(f'Shape of {blk_name} {BLK.binned_signal.shape} and shape of raws {raws.shape}')          
                    else:
                        log.info(f'Trial n. {i+1}/{len(blks)} {blk_name} mismatch with time shape of the session. Discarded')               
                        log.info(f'Shape of {blk_name} {BLK.binned_signal.shape} and shape of raws {raws.shape}')          
                    
                    raws = raws[0:-2, :, :, :]
                    delta_f = delta_f[0:-2, :, :, :]
                    sig = sig[0:-2, :]     
                    i = i-1
                    blks.remove(blk_name)

                    if base_report is not None:
                        trials_dict[blk_name] = None   
                else: 
                    raws[i, :, :, :] = BLK.binned_signal 
                    delta_f[i, :, :, :] =  process.deltaf_up_fzero(BLK.binned_signal, zero, deblank=blnk_switch, blank_sign = blank_s)
                    sig[i, :] = process.time_course_signal(delta_f[i, :, :, :], roi_mask)     # Log prints
                    
                # Trial storing
                start_time = datetime.datetime.now().replace(microsecond=0)

                if log is None:
                    print('Trial n. '+str(i+1)+'/'+ str(len(blks))+' loaded in ' + str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!')
                else:
                    log.info('Trial n. '+str(i+1)+'/'+ str(len(blks))+' loaded in ' + str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!')
            # If trials is discarded then parallel strategy of reshaping of output matrices
            else:
                if i == 0:
                    BLK = blk_file.BlkFile(
                        os.path.join(path_rawdata, blk_name),
                        header['spatial_bin'],
                        header['temporal_bin'],
                        header = None, 
                        detrend_switch    = detrend,
                        filename_particle = filename_particle)

                    header_blk = BLK.header
                    raws = np.empty((len(blks-1), header['n_frames'], header['original_height']//header['spatial_bin'], header['original_width']//header['spatial_bin']))
                    delta_f = np.empty((len(blks-1), header['n_frames'], header['original_height']//header['spatial_bin'], header['original_width']//header['spatial_bin']))
                    sig = np.empty((len(blks-1), header['n_frames']))
                    roi_mask = blk_file.circular_mask_roi(header['original_width']//header['spatial_bin'], header['original_height']//header['spatial_bin'])
                else:
                    # Discarding one element in the trial dimension, since the blk was deleted
                    raws = raws[0:-2, :, :, :]
                    delta_f = delta_f[0:-2, :, :, :]
                    sig = sig[0:-2, :]
    else:
        for i, blk_name in enumerate(blks):
            start_time = datetime.datetime.now().replace(microsecond=0)
            if base_report is not None:
                trial = al.get_trial(base_report, blk_name, time, heart, piezo, greys[1], greys[0], blank_id)
                # If the trial is empty, likely for absence of BLK name correspondance in BaseReport, it pops out the blkname from the list
                if trial is None:
                    print('Empty Trial')
                    blks.remove(blk_name)
                    if log is None:
                        print(f'{blk_name} was popped off')
                    else:
                        log.info(f'{blk_name} was popped off')
                # Otherwise store it
                elif trial is not None:
                    trials_dict[blk_name] = trial   
                    zero = trial.zero_frames
            else:
                zero = header['zero_frames']
        sig, delta_f, conditions, raws = None, None, None, None
    return sig, delta_f, conditions, raws, trials_dict, blks
    
def roi_strategy(matrix, tolerance, zero_frames):
    '''
    The method works.
    '''
    # framesOK=abs(signalROI-mat_meanSigROI)>toleranceLevel*mat_semSigROI;
    size = np.shape(matrix)
    tmp = np.zeros(size)
    for i, roi in enumerate(matrix):
        tmp[i, :] = signal.detrend(np.nan_to_num(roi))
    # Blank subtraction on tmp -no demean if blank subt-
    # Mean ROI signal over trials -70, shape output-
    # The 0 shape is the number of trials
    selected_frames_mask = np.abs(tmp - np.mean(tmp, axis=0))>\
        tolerance*(np.std(tmp, axis=0)/np.sqrt(np.shape(tmp)[0]))
    #This could be tricky: not on all the frames.
    autoselect = np.sum(selected_frames_mask, axis=1)<((size[1]-zero_frames)/2)
    mask_array = np.zeros(size[0], dtype=int)
    mask_array[autoselect] = 1
    return mask_array

def overlap_strategy(matrix, cd_i, path, header, switch_vis = False, separators = None, n_chunks = 1, loss = 'mae', threshold = 'median'):
    '''
    Parameters:
        matrix: A data matrix that you want to process.
        cd_i: The condition ID for which this strategy is being applied.
        path: The path to a directory where visualization results may be saved.
        header: A dictionary containing various parameters and metadata: namely, the Session header.
        switch_vis: A boolean flag that determines whether to create visualizations (default is False).
        separators: A list of indices that can be used to manually specify chunk boundaries (default is None).
        n_chunks: The number of chunks to divide the data into.
        loss: The loss metric used for selecting regions (e.g., 'mae' or 'mse').
        threshold: A thresholding method for region selection (default is 'median').

    Function Description:
        If separators is not provided, the function divides the data into n_chunks chunks -equally dimensioned chunks-
        and computes the loss metric for each pair of chunks using a selected loss metric (loss).
        The loss metric could be Mean Absolute Error ('mae') or Mean Squared Error ('mse').
        The function computes the pairwise loss metrics between chunks and creates a matrix (tmp_m_) representing 
        the loss between different pairs of chunks.
        The m matrix stores the sum of loss values for each chunk.
        The function applies a thresholding method (threshold) to the m matrix to select specific regions of interest 
        within the chunks.
        The selected regions are stored in the autoselect list.
        The mask_array is a binary array that indicates the selected regions within the matrix.
        If switch_vis is True, the function generates visualizations based on the selected regions.
        The function returns the autoselect list, mask_array, coordinates of selected regions, distribution information, 
        and normalized loss values.
    '''
    if separators is None:
        if  matrix.shape[1] % n_chunks == 0:
            matrix_ = matrix.reshape(matrix.shape[0], n_chunks, -1)
            tmp_m_ = np.zeros((n_chunks, matrix.shape[0], matrix.shape[0]))
            
            for m in range(n_chunks):
                tmp_m = np.zeros((matrix.shape[0], matrix.shape[0]))

                for n, i in enumerate(matrix_):
                    tmp = []

                    for j in matrix_:    
                        if loss == 'mae':
                            tmp.append(np.nanmean(np.abs(np.subtract(i[m, :], j[m, :]))))
                        elif loss == 'mse':
                            tmp.append(np.nanmean(np.square(np.subtract(i[m, :], j[m, :]))))

                    tmp_m[n, :] = np.asarray(tmp)    
                tmp_m_[m, :, :] = tmp_m
            m = np.nansum(tmp_m_, axis=1)
        else:
            # This check has to be done before running the script
            print('Use a proper number of chunks: exact division for the number of frames required')
                
    else:
        #print(f'The separators are: {separators}')
        tmp_list = list()
        for i, n in enumerate(separators):
            if i == 0:
                tmp_list.append(matrix[:, 0:n])
                tmp_list.append(matrix[:, n:separators[i+1]])
            elif len(separators)-1 == i:
                tmp_list.append(matrix[:, n:])
            else:
                tmp_list.append(matrix[:, n:separators[i+1]])
        for i in tmp_list:
            print(i.shape)
        
        tmp_m_ = list()
        n_chunks = len(tmp_list)
        for m in range(n_chunks):
            tmp_m = list()

            for i in tmp_list[m]:
                tmp = []
                for j in tmp_list[m]:    
                    if loss == 'mae':
                        tmp.append(np.nanmean(np.abs(np.subtract(i[:], j[:]))))
                    elif loss == 'mse':
                        tmp.append(np.nanmean(np.square(np.subtract(i[:], j[:]))))
                tmp_m.append(tmp)    
            tmp_m_.append(tmp_m)
            #print(np.shape(tmp_list))
        m = np.nansum(tmp_m_, axis=1)

    t_whol = list()
    coords = list()
    distr_info = list()
    ms_norm = list()

    for i in range(n_chunks):
        t, l, m_norm = process.lognorm_thresholding(m[i, :], switch = threshold)
        coords.append((l[0], l[3]))
        t_whol.append(t)
        distr_info.append(l)
        ms_norm.append(m_norm)

    # Intersection between the selected ones
    autoselect = list(set.intersection(*map(set,t_whol)))
    mask_array = np.zeros(m.shape[1], dtype=int)
    mask_array[autoselect] = 1
    
    if switch_vis:
        dv.chunk_distribution_visualization(coords, ms_norm, distr_info, cd_i, header, matrix, autoselect, mask_array, path)

    # Mask of selected ones
    return autoselect, mask_array, coords, distr_info, ms_norm

def statistical_strategy(matrix, up=75, bottom=25):
    size = np.shape(matrix)
    stds = np.std(matrix, axis = 1)
    autoselect = np.where((np.percentile(stds, q=bottom)<stds) & (np.percentile(stds, q=up)>stds))[0]
    # For combatibility with other methods, conversion in mask
    mask_array = np.zeros(size[0], dtype=int)
    mask_array[autoselect] = 1
    return mask_array

def get_all_blks(path_session, sort = True):
    '''
    All the .BLKs filenames, from the considered path_session, are picked.
    The list can be sorted by datetime or not, with the boolean variable sort.
    Sorted by time by default.
    '''
    tmp = [f.name for f in os.scandir(os.path.join(path_session,'rawdata')) if (f.is_file()) and (f.name.endswith(".BLK"))]
    try: 
        _ = datetime.datetime.strptime(tmp[0].split('_')[2] + tmp[0].split('_')[3], '%d%m%y%H%M%S')
        if sort:
            sort = True
    except:
        sort = False
    if sort:
        return sorted(tmp, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
    else:
        return tmp

def get_selected(matrix, autoselection):
    indeces = np.where(autoselection == 1)[0]
    if len(matrix.shape) == 4:
        df = matrix[indeces, :, :, :]
    elif len(matrix.shape) == 2:
        df = matrix[indeces, :]
    return df


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Launching autoselection pipeline')
    
    parser.add_argument('--path', 
                        dest='path_session',
                        type=str,
                        required=True,
                        help='The session path')
    
    parser.add_argument('--s_bin', 
                        dest='spatial_bin',
                        default= 3,
                        type=int,
                        help='The spatial bin value')

    parser.add_argument('--t_bin', 
                        dest='temporal_bin',
                        default= 1,
                        type=int,
                        required=False,
                        help='The time bin value')

    parser.add_argument('--tol', 
                        dest='tolerance',
                        type=int,
                        default = 20,
                        required=False,
                        help='Tolerance value for autoselection') 

    parser.add_argument('--mov', 
                        dest='mov_switch',
                        action='store_true')
    parser.add_argument('--no-mov', 
                        dest='mov_switch', 
                        action='store_false')
    parser.set_defaults(mov_switch=False)

    parser.add_argument('--dblnk', 
                        dest='deblank_switch',
                        action='store_true')
    parser.add_argument('--no-dblnk', # Bug 24/07/2023: AttributeError: 'Session' object has no attribute 'f_f0_blank'
                        dest='deblank_switch', 
                        action='store_false')
    parser.set_defaults(deblank_switch=False)

    parser.add_argument('--dtrend', 
                        dest='detrend_switch',
                        action='store_true')
    parser.add_argument('--no-dtrend', 
                        dest='detrend_switch', 
                        action='store_false')
    parser.set_defaults(detrend_switch=False)

    parser.add_argument('--cid', 
                    action='append', 
                    dest='conditions_id',
                    default=None,
                    type=int,
                    help='Conditions to analyze: None by default -all the conditions-')

    parser.add_argument('--chunks', 
                        dest='chunks',
                        type=int,
                        default = 1,
                        required=False,
                        help='Number of elements value for autoselection') 

    parser.add_argument('--strategy', 
                        dest='strategy',
                        type=str,
                        default = 'mae',
                        required=False,
                        help='Strategy for the autoselection: choose between mse/mae, statistical, roi -kevin equation-')  

    parser.add_argument('--store', 
                        dest='store_switch',
                        action='store_true')
    parser.add_argument('--no-store', 
                        dest='store_switch', 
                        action='store_false')
    parser.set_defaults(store_switch=False)   

    parser.add_argument('--matlab', 
                        dest='matlab_switch',
                        action='store_true')
    parser.add_argument('--no-matlab', 
                        dest='matlab_switch', 
                        action='store_false')
    parser.set_defaults(store_switch=False)   

    parser.add_argument('--logs_data', 
                        dest='logs_switch',
                        action='store_true')
    parser.add_argument('--no-logs_data', 
                        dest='logs_switch', 
                        action='store_false')
    parser.set_defaults(logs_switch=False)   

    parser.add_argument('--br_name', 
                        dest='base_report_name',
                        type=str,
                        default = 'BaseReport.csv',
                        required=False)  
    
    parser.add_argument('--br_head_dim', 
                        dest='base_head_dim',
                        type=int,
                        default = 19,
                        required=False)  
    
    parser.add_argument('--blank_id', 
                        dest='condid',
                        type=int,
                        default = None,
                        required=False) 

    parser.add_argument('--vis', 
                        dest='data_vis_switch', 
                        action='store_true')
    parser.add_argument('--no-vis', 
                        dest='data_vis_switch', 
                        action='store_false')
    parser.set_defaults(data_vis_switch=False) 

    parser.add_argument('--particle', 
                        dest='filename_particle',
                        type=str,
                        default = 'vsd_C',
                        required=False)    
    
    parser.add_argument('--zero', 
                        dest='zero_frames',
                        type=int,
                        default = None,
                        required=False) 
    

    logger = utils.setup_custom_logger('myapp')
    logger.info('Start\n')
    args = parser.parse_args()
    logger.info(args)
    # Check on quality of inserted data
    assert args.spatial_bin > 0, "Insert a value greater than 0"    
    assert args.temporal_bin > 0, "Insert a value greater than 0"    
    assert args.strategy in ['mse', 'mae', 'roi', 'roi_signals', 'ROI', 'statistic', 'statistical', 'quartiles'], "Insert a valid name strategy: 'mse', 'mae', 'roi', 'roi_signals', 'ROI', 'statistic', 'statistical', 'quartiles'"    
    start_time = datetime.datetime.now().replace(microsecond=0)
    session = Session(logger = logger, **vars(args))
    session.get_session()
    logger.info('Time for BLK preprocessing pipeline: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    #utils.inputs_save(session, 'session_prova')
    #utils.inputs_save(session.session_blks, 'blk_names')

# 38, 18, 38