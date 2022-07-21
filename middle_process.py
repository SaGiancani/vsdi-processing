import argparse, blk_file, datetime, process, utils
import ana_logs as al
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

LABEL_CONDS_PATH = 'metadata/labelConds.txt' 

class Condition:
    """
    Initializes attributes
    Default values for:
    *session_header = all the .BLK files contained inside path_session/rawdata/. It is a list of strings
    *cond_names = list of conditions' names.  
    *header = a dictionary with the kwargs value. See get_session_header method for details
    *session_blks = all the .BLK, per condition, considered for the processing. It is a subset of all_blks. It is a list of strings
    *motion_indeces = unused
    *time_course_signals = all the time courses of the considered BLKs. It is a numpy array of shape n_session_blk, n_frames, 1
    *trials_name = the .BLKs' filename of each selected trial. It is a list of strings
    *df_fz = deltaF/F0 for each selected trial. It is a numpy array of shape selected_trials, width, height
    *auto_selected = list of integers: 0 for not selected trial, 1 for selected. 
    *conditions = list of integers: the integer corresponds to the number of condition.
    Parameters
    ----------
    filename : str
        The path of the external file, containing the raw image
    """
    def __init__(self, condition_name, condition_numb, session_header):
        self.session_header = session_header
        self.session_name =  self.session_header['path_session'].split('/')[-2]+'-'+self.session_header['path_session'].split('/')[-3].split('-')[1] 
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
    
    def store_cond(self, t):
        tp = [self.session_header, self.session_name, self.cond_name, self.cond_id, self.binned_data, self.df_fz, self.time_course, self.averaged_df, self.averaged_timecourse, self.autoselection, self.blk_names, self.trials]
        utils.inputs_save(tp, os.path.join(t,'md_data','md_data_'+self.cond_name))
        return
    
    def load_cond(self, path):
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
        return

# Inserting inside the class variables and features useful for one session: we needs an object at this level for
# keeping track of conditions, filenames, selected or not flag for each trial.
class Session:
    def __init__(self, base_report_name = 'BaseReport.csv', base_head_dim = 19, logger = None, condid = None, store_switch = False, data_vis_switch = True, end_frame = 60, **kwargs):
        """
        Initializes attributes
        Default values for:
        *all_blks = all the .BLK files contained inside path_session/rawdata/. It is a list of strings
        *cond_names = list of conditions' names.  
        *header = a dictionary with the kwargs value. See get_session_header method for details
        *session_blks = all the .BLK, per condition, considered for the processing. It is a subset of all_blks. It is a list of strings
        *motion_indeces = unused
        *time_course_signals = all the time courses of the considered BLKs. It is a numpy array of shape n_session_blk, n_frames, 1
        *trials_name = the .BLKs' filename of each selected trial. It is a list of strings
        *df_fz = deltaF/F0 for each selected trial. It is a numpy array of shape selected_trials, width, height
        *auto_selected = list of integers: 0 for not selected trial, 1 for selected. 
        *conditions = list of integers: the integer corresponds to the number of condition.
        Parameters
        ----------
        filename : str
            The path of the external file, containing the raw image
        """
        if logger is None:
            self.log = utils.setup_custom_logger('myapp')
        else:
            self.log = logger              
        self.cond_names = None
        self.header = self.get_session_header(**kwargs)
        self.all_blks = get_all_blks(self.header['path_session'], sort = True) # all the blks, sorted by creation date -written on the filename-.
        self.cond_dict = self.get_condition_name()
        self.cond_names = list(self.cond_dict.values())
        self.blank_id = self.get_blank_id(cond_id=condid)

        # This can be automatized, with zero_frames, extracting parameters from BaseReport
        # Avoiding to load a BLK file
        # A blk loaded for useful hyperparameters
        blk = blk_file.BlkFile(os.path.join(self.header['path_session'],'rawdata', self.all_blks[np.random.randint(len(self.all_blks)-1)]), 
                            self.header['spatial_bin'], 
                            self.header['temporal_bin'],
                            self.header['zero_frames'])
        self.header['n_frames'] = blk.header['nframesperstim']
        self.header['original_height'] = blk.header['frameheight']
        self.header['original_width'] = blk.header['framewidth']
        self.header['ending_frame'] = end_frame
        # If considered conditions are not explicitly indicated, then all the conditions are considered
        # The adjustment of conditions_id set has to be done ALWAYS before the session_blks extraction       
        if self.header['conditions_id'] is None:
            self.header['conditions_id'] = self.get_condition_ids()
        else:
            self.header['conditions_id'] = list(set(self.header['conditions_id']+[self.blank_id]))
        # only the used blks for the selection

        if self.header['mov_switch']:
            self.motion_indeces = None
        
        self.session_blks = None
        self.time_course_signals = None
        self.trials_name = None 
        self.df_fzs = None
        self.raw_data = None
        self.auto_selected = None
        self.conditions = None
        self.counter_blank = 0  
        
        self.visualization_switch = data_vis_switch
        self.storage_switch = store_switch

        self.avrgd_time_courses = None
        self.avrgd_df_fz = None

        # Loading the BaseReport and SignalData in case of logs_switch
        if self.header['logs_switch']:
            try:
                start_time = datetime.datetime.now().replace(microsecond=0)
                self.log.info(f'Length of all_blks list: {len(self.all_blks)}')
                base_report, tris = al.get_basereport(self.header['path_session'], self.all_blks, name_report = base_report_name, header_dimension = base_head_dim)
                # Separator converter processing: , -string- to . -float-.
                for i in list(base_report.columns):
                    try:
                        base_report[[i]] = base_report[[i]].applymap(al.separator_converter)
                    except:
                        print(f'Column {i} is not a float')                
                self.base_report = base_report
                if tris[3]:
                    #self.all_blks.pop(tris[2])
                    self.log.info(f'Length of all_blks list after popping off from get_basereport: {len(self.all_blks)}')
                self.log.info('BaseReport properly loaded!')
                self.log.info('BaseReport loading time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
            except:
                self.log.info('Something went wrong loading the BaseReport')

            #try:
            #    path_trackreport = utils.find_thing('TrackerLog.csv', self.header['path_session'])
            #except:
            #    self.log('Something went wrong loading the TrackerLog')
            
            try:
                start_time = datetime.datetime.now().replace(microsecond=0)
                self.piezo, self.heart_beat = al.get_analog_signal(self.header['path_session'], self.base_report, name_report = 'SignalData.csv')
                self.log.info('Piezo and Heart Beat signals properly loaded!')
                self.log.info('Analogic signals loading time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
            except:
                self.log.info('Something went wrong loading the SignalData')
        else:
            self.base_report, self.piezo, self.heart_beat  = None, None, None

        #if self.header['deblank_switch']:
        # TO NOTICE: deblank_switch add roi_signals, df_fz, auto_selected, conditions, counter_blank and overwrites the session_blks
        self.time_course_blank = None
        self.f_f0_blank = None
        # Calling get_signal in the instantiation of Session allows to obtain the blank signal immediately.
        _ = self.get_signal(self.blank_id)


    def get_blank_id(self, cond_id = None):
        '''
        The method returns the index of blank condition.
        Some session require a specific condition index: cond_id variable designed for manual setting.
        If it is None -by default-, the method checks among the condition names: if labelConds.txt
        file exists, than the position of "blank" label is picked. Otherwise the position of last condition 
        is picked.
        '''
        if cond_id is None:
            try:
                tmp = [idx for idx, s in enumerate(self.cond_names) if 'blank' in s][0]+1
                self.log.info('Blank id: ' + str(tmp))
                return tmp
            except IndexError:
                self.log.info('No clear blank condition was identified: the last condition has picked')
                tmp = len(self.cond_names)
                self.log.info('Blank id: ' + str(tmp))
                return tmp
        else:
            return cond_id

    def get_signal(self, condition):
        # All the blank blks
        blks = [f for f in self.all_blks if (int(f.split('vsd_C')[1][0:2])==condition)]
        # Blank signal extraction
        self.log.info(f'Trials of condition {condition} loading starts:')
        if condition == self.blank_id:
            sig, df_f0, conditions, raws, trials = signal_extraction(self.header, blks, None, self.header['deblank_switch'], self.base_report, self.blank_id, self.piezo, self.heart_beat)
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
            tmp = np.mean(df_f0[indeces_select, :, :, :], axis=0)
            df_f0 = df_f0 - 1
            self.avrgd_df_fz = np.reshape(np.mean(df_f0[indeces_select, :, :, :], axis=0), (1, tmp.shape[0], tmp.shape[1], tmp.shape[2]))
            tmp_ = np.mean(sig[indeces_select, :], axis=0)
            self.avrgd_time_courses = np.reshape(tmp_, (1, tmp.shape[0]))
            # It's important that 1 is not subtracted to this blank_df: it is the actual blank signal
            # employed for normalize the signal             
            self.time_course_blank = tmp_
            self.f_f0_blank = tmp
            
            self.log.info('Blank signal computed')
                        
        else:
            sig, df_f0, conditions, raws, trials = signal_extraction(self.header, blks, self.f_f0_blank, self.header['deblank_switch'], self.base_report, self.blank_id, self.piezo, self.heart_beat)
            mask = self.get_selection_trials(condition, sig)
            self.conditions = self.conditions + conditions
            self.auto_selected = np.array(self.auto_selected.tolist() + mask.tolist(), dtype=int)
            self.session_blks = self.session_blks + blks
            indeces_select = np.where(np.array(mask)==1)
            indeces_select = indeces_select[0].tolist()
            #df_f0 = df_f0.reshape(1, df_f0.shape[1], df_f0.shape[2], df_f0.shape[3] ) 
            t =  np.mean(df_f0[indeces_select, :, :, :], axis=0)
            self.avrgd_df_fz = np.concatenate((self.avrgd_df_fz, t.reshape(1, t.shape[0], t.shape[1], t.shape[2])), axis=0) 
            print(f'Shape averaged dF/F0: {np.shape(self.avrgd_df_fz )}')
            t_ =  np.mean(sig[indeces_select, :], axis=0)
            self.avrgd_time_courses = np.concatenate((self.avrgd_time_courses,  t_.reshape(1,  t_.shape[0])), axis=0) 
            print(f'Shape averaged tc: {np.shape(self.avrgd_time_courses )}')

        if self.visualization_switch:
            self.roi_plots(condition, sig, mask, blks)
            if self.base_report is not None:
                zero_of_cond = int(np.mean([v.zero_frames for v in trials.values()]))
                foi_of_cond = int(np.mean([v.FOI for v in trials.values()]))
                time_sequence_visualization(zero_of_cond, foi_of_cond,  int(np.mean(zero_of_cond + foi_of_cond)), df_f0[indeces_select, :, :, :], np.array(blks)[indeces_select], 'cond'+str(condition), self.header, self.set_md_folder(), log_ = self.log, max_trials = 20)
            else:
                time_sequence_visualization(self.header['zero_frames'], 20, self.header['ending_frame'], df_f0[indeces_select, :, :, :], np.array(blks)[indeces_select], 'cond'+str(condition), self.header, self.set_md_folder(), log_ = self.log, max_trials = 20)

        # If storage switch True, than a Condition object is instantiate and stored
        if self.storage_switch:
            start_time = datetime.datetime.now().replace(microsecond=0)
            cond = Condition(self.cond_dict[condition], condition, self.header)
            if self.header['raw_switch']:
                cond.binned_data = raws
            cond.df_fz = df_f0
            cond.time_course = sig
            cond.autoselection = mask
            cond.blk_names = blks
            cond.averaged_df = self.avrgd_df_fz[-1, :, :, :]
            cond.averaged_timecourse = self.avrgd_time_courses[-1, :]
            if self.base_report is not None:
                cond.trials = trials
            #Storing folder
            t = self.set_md_folder()
            if not os.path.exists(os.path.join(t,'md_data')):
                os.makedirs(os.path.join(t,'md_data'))
            cond.store_cond(t)
            del cond
            self.log.info('Storing condition time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))                

        return sig, df_f0, mask, blks, raws

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
            tmp = [f for f in self.all_blks if (int(f.split('vsd_C')[1][0:2]) in self.header['conditions_id'])]
            return sorted(tmp, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))

    def get_condition_ids(self):
        '''
        The method returns a list of all the condition's ids, taken from the .BLK names.
        '''
        return list(set([int(i.split('vsd_C')[1][0:2]) for i in self.all_blks]))
        
    def get_condition_name(self):
        '''
        The method returns a list of condition's names: if a labelConds.txt exist inside metadata's folder, 
        than names are simply loaded. Otherwise a list of names with "Condition #" style is built.
        '''
        try:
            with open(os.path.join(self.header['path_session'], LABEL_CONDS_PATH)) as f:
                contents = f.readlines()
            return {j+1:i.split('\n')[0] for j, i in enumerate(contents)}
        except FileNotFoundError:
            self.log.info('Check the labelConds.txt presence inside the metadata subfolder')
            cds = self.get_condition_ids()
            return {j+1:'Condition ' + str(c) for j, c in enumerate(cds)}
        except NotADirectoryError:
            self.log.info(os.path.join(self.header['path_session'], LABEL_CONDS_PATH) +' path does not exist')
            cds = self.get_condition_ids()
            return {j+1:'Condition ' + str(c) for j, c in enumerate(cds)}

    def get_session_header(self, path_session, spatial_bin, temporal_bin, zero_frames, tolerance, mov_switch, deblank_switch, conditions_id, chunks, strategy, raw_switch, logs_switch):
        header = {}
        header['path_session'] = path_session
        header['spatial_bin'] = spatial_bin
        header['temporal_bin'] = temporal_bin
        header['zero_frames'] = zero_frames
        header['tolerance'] = tolerance
        header['mov_switch'] = mov_switch
        header['deblank_switch'] = deblank_switch
        header['conditions_id'] = conditions_id
        header['chunks'] = chunks
        header['strategy'] = strategy
        header['raw_switch'] = raw_switch
        header['logs_switch'] = logs_switch
        return header
    
    def get_session(self):

        if len(self.header['conditions_id']) > 1:
            for cd in self.header['conditions_id']:
                c_name = self.cond_dict[cd]
                if cd != self.blank_id:
                    self.log.info('Procedure for loading BLKs of condition ' +str(cd)+' starts')
                    self.log.info('Condition name: ' + c_name)                        
                    _, _, tmp, _, _ = self.get_signal(cd)
                    self.log.info(str(int(sum(tmp))) + '/' + str(len(tmp)) +' trials have been selected for condition '+str(c_name))
                    
            self.log.info('Globally ' + str(int(sum(self.auto_selected))) + '/' + str(len(self.session_blks)) +' trials have been selected!')
            session_blks = np.array(self.session_blks)
            self.trials_name = session_blks[self.auto_selected]

        if self.visualization_switch:
            # titles gets the name of blank condition as first, since it was stored first
            time_sequence_visualization(self.header['zero_frames'], 20, self.header['ending_frame'], self.avrgd_df_fz, [self.cond_dict[self.blank_id]]+[self.cond_dict[c] for c in self.header['conditions_id'] if c!=self.blank_id] , 'avrgd_conds', self.header, self.set_md_folder(), log_ = self.log)

        else:
            self.log.info('Warning: Something weird in get_session')
        return

    def get_selection_trials(self, condition, time_course):
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
        session_name = self.header['path_session'].split('/')[-2]+'-'+self.header['path_session'].split('/')[-3].split('-')[1]
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
                    ax.set_ylim(np.min(sig[cdi_select, :]) - (np.max(sig[cdi_select]) - np.min(sig[cdi_select]))*0.005, np.max(sig[cdi_select, :]) + (np.max(sig[cdi_select]) - np.min(sig[cdi_select]))*0.005)
                    if mask[count]==1:
                        color = 'b'
                    else:
                        color = 'r'
                    ax.plot(sig[count, :], color)
                    ax.set_title(blks[count])
                    ax.errorbar(x, np.mean(sig[cdi_select, :], axis = 0), yerr=(np.std(sig[cdi_select, :], axis = 0)/np.sqrt(len(cdi_select))), fmt='--', color = 'k', elinewidth = 0.5)
                    ax.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
                    #ax.set_ylim(-0.002,0.002)
                if row<len(subfigs)-2:
                    ax.get_xaxis().set_visible(False)
                elif row<len(subfigs)-1:
                    ax.get_xaxis().set_visible(True)
                elif row == len(subfigs)-1:
                    ax.axis('off')
                    ax_ = subfig.subplots(1, 1)
                    ax_.set_ylim(np.min(sig[cdi_select, :]) - (np.max(sig[cdi_select]) - np.min(sig[cdi_select]))*0.005, np.max(sig[cdi_select, :]) + (np.max(sig[cdi_select]) - np.min(sig[cdi_select]))*0.005)
                    for i in sig[cdi_select[:-1], :]:
                        ax_.plot(x, i, 'gray', linewidth = 0.5)
                    ax_.plot(x, sig[cdi_select[-1], :], 'gray', linewidth = 0.5, label = 'Trials')
                    ax_.plot(x, np.mean(sig[cdi_select, :], axis=0), 'k', label = 'Average Selected trials', linewidth = 2)
                    ax_.plot(x, np.mean(sig[cdi_unselect, :], axis=0), 'crimson', label = 'Average Unselected trials', linewidth = 2)
                    ax_.plot(x, np.mean(sig, axis=0), 'green', label = 'Average All trials Cond. ' + str(cd_i), linewidth = 2)
                    ax_.plot(x, blank_sign, color='m', label = 'Average Blank Signal' ,linewidth = 2)
                    #ax_.plot(list(range(0,np.shape(sig)[1])), blank_sign, color='m', label = 'Average Blank Signal' ,linewidth = 5)
                    ax_.legend(loc="upper left")                
                    ax_.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
                
        tmp = self.set_md_folder()
        if not os.path.exists(os.path.join(tmp,'time_course')):
            os.makedirs(os.path.join(tmp,'time_course'))
        plt.savefig(os.path.join(tmp,'time_course', session_name+'_tc_0'+str(cd_i)+'.png'))
        #plt.savefig((path_session+'/'session_name +'/'+ session_name+'_roi_0'+str(cd_i)+'.png')
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
            + '_deblank' + str(self.header['deblank_switch'])
        
        folder_path = os.path.join(session_path, 'derivatives/',folder_name)               
        if not os.path.exists(folder_path):
        #if not os.path.exists( path_session+'/'+session_name):
            os.makedirs(folder_path)
            #os.mkdirs(path_session+'/'+session_name)
        return folder_path

def signal_extraction(header, blks, blank_s, blnk_switch, base_report, blank_id, piezo, heart, log = None):
    #motion_indeces, conditions = [], []
    conditions = []
    path_rawdata = os.path.join(header['path_session'],'rawdata/')
    if base_report is not None:
        trials_dict = dict()
        greys = al.get_greys(header['path_session'], int(os.path.join(path_rawdata, blks[0]).split('vsd_C')[1][0:2]))
    else:
        trials_dict = None

    if log is None:
        print(f'The blank_signal exist: {blank_s is not None}')
        print(f'The blank switch is: {blnk_switch}')
    else:
        log.info(f'The blank_signal exist: {blank_s is not None}')
        log.info(f'The blank switch is: {blnk_switch}')

    for i, blk_name in enumerate(blks):
        start_time = datetime.datetime.now().replace(microsecond=0)
        # If first BLK file, than the header is stored
        if base_report is not None:
            trial_df = base_report.loc[base_report['BLK Names'] == blk_name]
            trial_series = trial_df.iloc[0]
            trial = trial_series.to_dict()
            #    def __init__(self, report_series_trial, heart, piezo, session_path, blank_cond, index, log = None, stimulus_fr = None, zero_fr = None, time_res = 10, blk_file = None):
            if (heart is not None) and (piezo is not None):
                trial = al.Trial(trial, heart[trial_df.index[0]], piezo[trial_df.index[0]], header['path_session'], blank_id, trial_df.index[0], greys[1], greys[0])
            else:
                trial = al.Trial(trial, None, None, header['path_session'], blank_id, trial_df.index[0], greys[1], greys[0])
            trials_dict[blk_name] = trial   
            zero = trial.zero_frames
        else:
            zero = header['zero_frames']
 

        if i == 0:
            BLK = blk_file.BlkFile(
                os.path.join(path_rawdata, blk_name),
                header['spatial_bin'],
                header['temporal_bin'],
                zero,
                header = None)

            header_blk = BLK.header
            raws = np.zeros((len(blks), header['n_frames'], header['original_height']//header['spatial_bin'], header['original_width']//header['spatial_bin']))
            delta_f = np.zeros((len(blks), header['n_frames'], header['original_height']//header['spatial_bin'], header['original_width']//header['spatial_bin']))
            sig = np.zeros((len(blks), header['n_frames']))
            roi_mask = blk_file.circular_mask_roi(header['original_width']//header['spatial_bin'], header['original_height']//header['spatial_bin'])
        else:
            BLK = blk_file.BlkFile(
                os.path.join(path_rawdata, blk_name), 
                header['spatial_bin'], 
                header['temporal_bin'], 
                zero,
                header = header_blk)
        
        # Log prints
        if log is None:
            print(f'The blk file {blk_name} is loaded')
        else:
            log.info(f'The blk file {blk_name} is loaded')
            
        #at the end something like (nblks, 70, 1)         
        conditions.append(BLK.condition)
        raws[i, :, :, :] =  BLK.binned_signal 
        delta_f[i, :, :, :] =  process.deltaf_up_fzero(BLK.binned_signal, zero, deblank=blnk_switch, blank_sign = blank_s)
        sig[i, :] = process.time_course_signal(delta_f[i, :, :, :], roi_mask)     # Log prints

        if log is None:
            print('Trial n. '+str(i+1)+'/'+ str(len(blks))+' loaded in ' + str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!')
        else:
            log.info('Trial n. '+str(i+1)+'/'+ str(len(blks))+' loaded in ' + str(datetime.datetime.now().replace(microsecond=0)-start_time)+'!')
    
    return sig, delta_f, conditions, raws, trials_dict
    
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
                            tmp.append(np.abs(np.subtract(i[m, :], j[m, :])).mean())
                        elif loss == 'mse':
                            tmp.append(np.square(np.subtract(i[m, :], j[m, :])).mean())

                    tmp_m[n, :] = np.asarray(tmp)    
                tmp_m_[m, :, :] = tmp_m
                
            m = np.sum(tmp_m_, axis=1)
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
                        tmp.append(np.abs(np.subtract(i[:], j[:])).mean())
                    elif loss == 'mse':
                        tmp.append(np.square(np.subtract(i[:], j[:])).mean())
                tmp_m.append(tmp)    
            tmp_m_.append(tmp_m)
            #print(np.shape(tmp_list))
        m = np.sum(tmp_m_, axis=1)

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
        chunk_distribution_visualization(coords, ms_norm, distr_info, cd_i, header, matrix, autoselect, mask_array, path)

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
    tmp = [f.name for f in os.scandir(os.path.join(path_session,'rawdata/')) if (f.is_file()) and (f.name.endswith(".BLK"))]
    if sort:
        return sorted(tmp, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
    else:
        return tmp

def time_sequence_visualization(start_frame, n_frames_showed, end_frame, data, titles, title_to_print, header, path_, circular_mask = True, log_ = None, max_trials = 20):
    start_time = datetime.datetime.now().replace(microsecond=0)
    session_name = header['path_session'].split('/')[-2]+'-'+header['path_session'].split('/')[-3].split('-')[1]
    # Array with indeces of considered frames: it starts from the last considerd zero_frames
    considered_frames = np.round(np.linspace(start_frame-1, end_frame-1, n_frames_showed))
    # Borders for caxis
    max_bord = np.percentile(data, 75)
    min_bord = np.percentile(data, 10)
    if log_ is not None:
        print(f'Max value heatmap: {max_bord}')
        print(f'Min value heatmap: {min_bord}')
    else:
        log_.info(f'Max value heatmap: {max_bord}')
        log_.info(f'Min value heatmap: {min_bord}')
    # Implementation for splitting big matrices for storing
    pieces = int(np.ceil(len(data)/max_trials))
    tmp_list = list()
    separators = np.linspace(0, len(data), pieces+1, endpoint=True, dtype=int)
    print(separators)
    for i, n in enumerate(separators):
        if i != 0:
            matrix = data[separators[i-1]:n, :, :, :]

            count = 0
            fig = plt.figure(constrained_layout=True, figsize = (n_frames_showed-2, len(matrix)), dpi = 80)
            fig.suptitle(f'Session {session_name}')# Session name
            subfigs = fig.subfigures(nrows=len(matrix), ncols=1)
            for sequence, subfig in zip(matrix, subfigs):
                subfig.suptitle(f'{titles[count]}')
                axs = subfig.subplots(nrows=1, ncols=n_frames_showed)

                # Showing each frame
                for df_id, ax in zip(considered_frames, axs):
                    Y = sequence[int(df_id), :, :]
                    if circular_mask:
                        mask = utils.sector_mask(Y.shape, (Y.shape[0]//2, Y.shape[1]//2), (np.min(np.shape(Y)))*0.40, (0,360) )
                        Y[~mask] = np.NAN
                    ax.axis('off')
                    pc = ax.pcolormesh(Y, vmin=min_bord, vmax=max_bord, cmap=utils.PARULA_MAP)
                subfig.colorbar(pc, shrink=1, ax=axs)#, location='bottom')
                count +=1
                
            tmp = path_
            if not os.path.exists(os.path.join(tmp,'activity_maps')):
                os.makedirs(os.path.join(tmp,'activity_maps'))
            plt.savefig(os.path.join(tmp,'activity_maps', session_name+'_piece0'+str(i)+'_'+str(title_to_print)+'.png'))

    if log_ is not None:
        log_.info('Plotting heatmaps time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    else:
        print('Plotting heatmaps time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    return  

def chunk_distribution_visualization(coords, m_norm, l, cd_i, header, tc, indeces_select, mask_array, path):
    strategy = header['strategy']
    session_name = header['path_session'].split('/')[-2]+'-'+header['path_session'].split('/')[-3].split('-')[1]
    colors_a = utils.COLORS
    xxx=np.linspace(0.001,np.max(list(zip(*coords))[1]),1000)
    #print(len(l))
    title = f'Condition #{cd_i}' 
    fig = plt.figure(constrained_layout = True, figsize=(25, 10))
    fig.suptitle(title)# Session name
    #plt.title(f'Condition {cond_num}')
    subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[2,1.25])
    axs = subfigs[0].subplots(nrows=1, ncols=3)#, sharey=True)

    for i,j in enumerate(l):
        axs[2].plot(xxx, process.log_norm(xxx, j[1], j[2]), color = colors_a[i], alpha = 0.5)
        axs[2].plot(list(zip(*coords))[1], list(zip(*coords))[0], "k", marker=".", markeredgecolor="red", ls = "")

        # Median + StdDev
        # Median + StdDev
        mean_o = np.exp(j[1] + j[2]*j[2]/2.0)
        median_o = np.exp(j[1])
        median_o_std = (median_o + 2*np.sqrt((np.exp(j[2]*j[2])-1)*np.exp(j[1]+j[1]+j[2]*j[2])))
        mean_o_std = mean_o + 2*np.sqrt((np.exp(j[2]*j[2])-1)*np.exp(j[1]+j[1]+j[2]*j[2]))
        #plt.axvline(x=median_o, color = colors_a[-i], linestyle='--')
        axs[2].axvline(x=median_o_std, color = colors_a[i], linestyle='-')
        # Mean + StdDev
        #plt.axvline(x=mean_o, color = colors_a[i+1], linestyle='--')
        axs[2].axvline(x=mean_o_std, color = colors_a[i], linestyle='-')


            # We can set the number of bins with the *bins* keyword argument.
        #axs[0].hist(dist1, bins=n_bins)
        #axs[1].hist(dist2, bins=n_bins)
        #plt.gca().set_title()
        axs[0].set_ylabel(strategy)
        axs[0].set_xlabel('Trials')
        #plt.plot(range(len(mae[i, :])), mae[i, :], marker="o", markeredgecolor="red", markerfacecolor="green", ls="-")    
        axs[0].plot(range(len(m_norm[i])), m_norm[i], marker="o", markeredgecolor="red", markerfacecolor=colors_a[i], ls="")#, marker="o", markeredgecolor="red", markerfacecolor="green")
        #plt.plot(range(len(mse[i, :])), [np.mean(mse[i, :])-0.5*np.std(mse[i, :])]*len(mse[i, :]),  ls="--", color = colors_a[i])
        axs[0].plot(range(len(m_norm[i])), [mean_o_std]*len(m_norm[i]),  ls="-", color = colors_a[i])
        #plt.plot(range(len(mae[i, :])), [median_o_std]*len(mae[i, :]),  ls="-", color = colors_a[-i])
        
        #mse[i, :] 
        #plt.plot(range(0, np.max(mse[0])), )
        axs[1].set_ylabel('Count')
        axs[1].set_xlabel(strategy)
        #plt.gca().set_title(f'Histogram for Condition {cond_num}')
        axs[1].hist(m_norm[i], bins = 50, color=colors_a[i], alpha=0.8)

    axs = subfigs[1].subplots(nrows=1, ncols=2)#, sharey=True)

    unselected = []
    for l, (i, sel) in enumerate(zip(tc, mask_array)):
        if sel == 0:
            col = 'crimson'
            alp = 1
            tmp_u = i
            unselected.append(l)
        #else:
            #col = 'grey'
            #alp = 1
            #tmp_s = i
            axs[0].plot(i, color = col, linewidth = 0.5, alpha = alp)
    #axs[0].plot(np.arange(60),tmp_s, color = 'grey', label = 'Selected trials' )
    shapes = np.shape(tc)
    axs[0].plot(np.arange(shapes[1]), tmp_u, color = 'crimson', linewidth = 0.5, label = 'Unselected trials')
    axs[0].plot(np.arange(shapes[1]), np.mean(tc[indeces_select], axis=0), color = 'k', linewidth = 2, label = 'Average among selected trials')
    axs[0].plot(np.arange(shapes[1]), np.mean(tc[unselected], axis=0), color = 'red', linewidth = 2, label = 'Average among unselected trials')
    axs[0].legend(loc = 'upper left')
    axs[0].set_ylim(np.min(tc[indeces_select]) - (np.max(tc[indeces_select]) - np.min(tc[indeces_select]))*0.05, np.max(tc[indeces_select]) + (np.max(tc[indeces_select]) - np.min(tc[indeces_select]))*0.05)
    #plt.subplot(2,3,5)
    for k, i in enumerate(tc[indeces_select[:-1]]):
        axs[1].plot(i, 'gray', linewidth = 0.5)
    axs[1].plot(tc[indeces_select[-1]], 'gray', linewidth = 0.5, label = 'Trials')
    axs[1].plot(np.arange(shapes[1]), np.mean(tc[indeces_select], axis=0), color = 'k', linewidth = 2, label = 'Average among selected trials')
    axs[1].plot(np.arange(shapes[1]), np.mean(tc[unselected], axis=0), color = 'red', linewidth = 2, label = 'Average among unselected trials')
    axs[1].set_ylim(np.min(tc[indeces_select]) - 0.0005, np.max(tc[indeces_select]) + 0.0005)    
    axs[1].legend(loc = 'upper left')
        
    tmp = path
    if not os.path.exists(os.path.join(tmp,'chunks_analysis')):
        os.makedirs(os.path.join(tmp,'chunks_analysis'))
    plt.savefig(os.path.join(tmp,'chunks_analysis', session_name+'_chunks_0'+str(cd_i)+'.png'))
    return

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
    
    parser.add_argument('--zero',
                        dest='zero_frames',
                        type=int,
                        default = 20,
                        required=False,
                        help='The first frames considered zero')    

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
    parser.add_argument('--no-dblnk', 
                        dest='deblank_switch', 
                        action='store_false')
    parser.set_defaults(deblank_switch=False)

    parser.add_argument('--raw', 
                        dest='raw_switch',
                        action='store_true')
    parser.add_argument('--no-raw', 
                        dest='raw_switch', 
                        action='store_false')
    parser.set_defaults(raw_switch=False)
    
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
    

    logger = utils.setup_custom_logger('myapp')
    logger.info('Start\n')
    args = parser.parse_args()
    logger.info(args)
    # Check on quality of inserted data
    assert args.spatial_bin > 0, "Insert a value greater than 0"    
    assert args.temporal_bin > 0, "Insert a value greater than 0"    
    assert args.zero_frames > 0, "Insert a value greater than 0"    
    assert args.strategy in ['mse', 'mae', 'roi', 'roi_signals', 'ROI', 'statistic', 'statistical', 'quartiles'], "Insert a valid name strategy: 'mse', 'mae', 'roi', 'roi_signals', 'ROI', 'statistic', 'statistical', 'quartiles'"    
    start_time = datetime.datetime.now().replace(microsecond=0)
    session = Session(logger = logger, **vars(args))
    session.get_session()
    logger.info('Time for blks autoselection: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    #utils.inputs_save(session, 'session_prova')
    #utils.inputs_save(session.session_blks, 'blk_names')

# 38, 18, 38