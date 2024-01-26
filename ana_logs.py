import datetime, os, utils
import numpy as np
import pandas as pd

class Trial:
    def __init__(self, report_series_trial, heart, piezo, blank_cond, grey_end, grey_start, log = None, stimulus_fr = None, zero_fr = None, time_res = 10, blk_file = None):
        self.blk = blk_file
        self.name = report_series_trial['BLK Names']
        self.condition = int(report_series_trial['IDcondition'])
        self.fix_correct = report_series_trial['Preceding Event IT'] == 'FixCorrect'
        self.correct_behav = report_series_trial['behav Correct'] == 1

        # Behavioral outcome
        try:
            self.orientation = int(report_series_trial['Orientation Behav'])
            self.orientation_outcome = int(report_series_trial['IDrejected ROI'])
            self.get_orientation_outcome()
            print(self.orientation_outcome)
        except:
            pass

        # Probable bug in behav Correct variable in some session: all the delays -even if broken- are stored
        try:
            self.behav_latency = int(report_series_trial['Onset Time_ Behav Correct']) -  int(report_series_trial['Onset Time_ Behav Stim'])# - 500
        except:
            self.behav_latency = 0        

        self.id_trial = int(report_series_trial['Total Trial Number']) - 1
        # if self.fix_correct and self.correct_behav and self.condition != blank_cond:
        #     self.behav_latency = int(report_series_trial['Onset Time_ Behav Correct']) -  int(report_series_trial['Onset Time_ Behav Stim'])# - 500
        # else:
        #     self.behav_latency = 0

        if self.condition != blank_cond:
            if stimulus_fr is None:
                # Total registration time (End registration (PNG flow + Post Stimulus time) - Start registration) - Starting Grey frames*temporal resolution - Ending Grey frames*temporal resolution + 25 ms latency 
                stimulus_fr = round((report_series_trial['Onset Time_ End Stim'] - report_series_trial['Onset Time_ Stim'] - (grey_end*time_res) - (grey_start*time_res) + 25)/time_res)         
            
            self.FOI = stimulus_fr

            if zero_fr is None:
                #PreStimulus Time + nGreyFrames*10ms + 25ms Response Latency
                zero_fr = round(((report_series_trial['Onset Time_ Stim'] - report_series_trial['Onset Time_ Pre Stim']) + 25 + (grey_start*time_res))/time_res) 
            
            self.zero_frames = zero_fr

        else:
            self.zero_frames = 20
            self.FOI = 35

        self.start_stim = float(separator_converter(report_series_trial['Onset Time_ Pre Stim']))
        self.onset_stim = float(separator_converter(report_series_trial['Onset Time_ Stim']))
        self.end_trial = float(separator_converter(report_series_trial['Onset Time_ End Stim']))
        # print(self.onset_stim - self.start_stim)
        self.heart_signal = heart
        self.piezo_signal = piezo
    
def get_orientation_outcome(self):
    '''
    The method based on the position encode given by EventIDE behavioral positions, store the corresponding angle.
    8 possible positions, encoded by an integer from 1 to 8. Each of them corresponds to an angle multiple of 45°.
    If orientation_outcome is 0, than the behavioral outcome was correct, then the orientation corresponding to the 
    stimulus is stored. Otherwise, the actual angle is stored. 
    ''' 
    if self.orientation_outcome == 0:
        self.orientation_outcome = self.orientation
    elif self.orientation_outcome == 1:
        self.orientation_outcome = 0
    elif self.orientation_outcome == 2:
        self.orientation_outcome = 45
    elif self.orientation_outcome == 3:
        self.orientation_outcome = 90
    elif self.orientation_outcome == 4:
        self.orientation_outcome = 135
    elif self.orientation_outcome == 5:
        self.orientation_outcome = 180
    elif self.orientation_outcome == 6:
        self.orientation_outcome = 225
    elif self.orientation_outcome == 7:
        self.orientation_outcome = 270
    elif self.orientation_outcome == 8:
        self.orientation_outcome = 315
    return

    
def get_trial(base_report, blk_name, time, heart, piezo, grey_end, grey_start, blank_id):
    try:
    # To investigate the reason of the try/except construct
        trial_df = base_report.loc[base_report['BLK Names'] == blk_name]
        trial_series = trial_df.iloc[0]
        trial = trial_series.to_dict()
        trial = Trial(trial, None, None, blank_id, grey_end, grey_start)
        if (heart is not None) and (piezo is not None):
            #cut_heartbeat = signal_cutter(time[trial.id_trial-1], heart[trial.id_trial-1], trial.start_stim, trial.end_trial)
            #cut_piezo = signal_cutter(time[trial.id_trial-1], piezo[trial.id_trial-1], trial.start_stim, trial.end_trial)
            trial.heart_signal = heart[trial.id_trial-1]
            trial.piezo_signal = piezo[trial.id_trial-1]
            # print(len(trial.piezo_signal))
            # print(len(trial.heart_signal))
        return trial
    except:
        if len(trial_df) == 0:
            print(f'{blk_name} has not correspondance in BaseReport')
        return None

    # trial_df = base_report.loc[base_report['BLK Names'] == blk_name]
    # trial_series = trial_df.iloc[0]
    # trial = trial_series.to_dict()
    # if (heart is not None) and (piezo is not None):
    #     trial = Trial(trial, heart[trial_df.index[0]], piezo[trial_df.index[0]], blank_id, grey_end, grey_start)
    # else:
    #     trial = Trial(trial, None, None, blank_id, grey_end, grey_start)
    # return trial

def get_greys(session_path, condition):
    pngfile_path = utils.find_thing('PNGfiles', os.path.join(session_path, 'sources'))
    try:
        grey_frames_start, grey_frames_end, _ = get_grey_frames(pngfile_path, condition)
        #n_pngs = pngs_shape[0]
    except:
        #print('Issue with PNGfiles: standard values 5 grey frames pre stimulus and 5 post will be used')
        grey_frames_start = 5
        grey_frames_end = 5
    return grey_frames_start, grey_frames_end

def add_blknames2basereport(BaseReport, all_blks):
    '''
    The method gets a BaseReport -pandas.DataFrame- and a list of all the BLK 
    filenames, and returns a BaseReport -pandas.DataFrame- with same columns
    plus one, BLK Names with the BLK filenames per each trial with FixCorrect
    as Preceding Event IT value.
    In the case that the all blks are less then the trials on the BaseReport,
    the method allows to find the best matching, discarding the absent BLKfile.
    '''
    # Bind the all_blks list to the BaseReport metadata
    try:
        # Sorting BLK filenames by date of storing -the one assigned on filename-
        #sorted_list = sorted(all_blks, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
        # Consider the BLK names, in case of FixCorrect preceding event IT
        print(f'Number of raw files: {len(all_blks)}')
        len_ = len(BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect'])
        print(f'Number of trials registered in log file: { len_ }')
        BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = all_blks
        tris = None
    except:
        print('Mismatch between BLK files and FixCorrect trials number')
        cds = BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'IDcondition'].tolist()
        #sorted_list = sorted(all_blks, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
        if (len(all_blks)!=len(cds)):
            tris = next( (idx, x, y) for idx, (x, y) in enumerate(zip(cds, all_blks)) if x!= int(y.split('_C')[1][:2]))
            print('Number of elements in BLKs list is changed:')            
            if (len(all_blks)<len(cds)):
                # If there is mismatch between the condition id in BaseReport and condition id in the BLK filename
                # It stores index, condition number, and BLK filename of the mismatch.
                all_blks.insert(tris[0], 'Missing')
                print('A Missing row is added')    
                tris = tris + (False,)        
            elif (len(all_blks)>len(cds)):
                all_blks.pop(tris[0])
                print(f'File {tris[2]} is deleted')
                tris = tris + (True,)        
                print('Take care to time course and dF/F0 matrix indeces and indexing system.')            
            elif abs(len(all_blks)-len(cds)) > 1:
                print('More than one blk missing/in surplus')
        # Consider the BLK names, in case of FixCorrect preceding event IT
        BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = all_blks
    #print(f'Tris value: {tris}')
    return BaseReport, tris


def discrepancy_blk_attribution(BaseReport):
    count = 0
    a = list()
    t = list(BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect','BLK Names'])
    for i in t:
        try:
            a.append(float(i.split('vsd_C')[1][:2]))
        except:
            a.append(1000)
    n = list(BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'IDcondition'])
    for l, (i,j) in enumerate(zip(n, a)):
        if i!=j:
            print(f'BLK filename condition {t[l]} and correspondent condition mismatched {i}')
            BaseReport.loc[BaseReport['BLK Names']  == t[l], 'BLK Names'] = np.nan
            count+=1
    return BaseReport, count


def get_basereport(session_path, all_blks, name_report = 'BaseReport.csv', header_dimension = 19):
    '''
    Load the BaseReport
    '''
    BaseReport_path = utils.find_thing(name_report, session_path, what = 'file')
    # Discarding duplicate for bugged sessions
    if len(BaseReport_path)>1:
        print(f'{len(BaseReport_path)} BaseReport are found')
        BaseReport_path = [i for i in BaseReport_path if 'bug' not in i.lower()]
    BaseReport = pd.read_csv(BaseReport_path[0], sep=';', header=header_dimension)
    # Some csv presents "" signs
    for i in list(BaseReport.columns):
        try:
            BaseReport[[i]] = BaseReport[[i]].applymap(delete_chars)
            BaseReport[[i]] = BaseReport[[i]].applymap(delete_chars)
        except:
            pass      
    #Adding BLK Names columns to the dataframe
    print('csv cleaned by unproper chars')
    print(f'Number of raw files: {len(all_blks)}')
    print(list(BaseReport.columns))
    # print(BaseReport['Preceding Event IT'])
    len_ = len(BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect'])
    print(f'Number of trials registered in log file: { len_ }')
    if abs(len_ - len(all_blks)) > 1:
        BaseReport = sorting_from_first(BaseReport, all_blks)
        tris = None
    else:
        BaseReport, tris = add_blknames2basereport(BaseReport, all_blks)
    # Check the discrepancy between IDcondition and BLK Names columns
    return BaseReport, tris

def get_denoised_basereport(all_blks, path_session, name_report = 'BaseReport_.csv', header_dimension = 19):
    start_time = datetime.datetime.now().replace(microsecond=0)
    print(f'Length of all_blks list: {len(all_blks)}')
    base_report, _ = get_basereport(path_session, all_blks, name_report = name_report, header_dimension = header_dimension)
    # Separator converter processing: , -string- to . -float-.
    for i in list(base_report.columns):
        try:
            base_report[[i]] = base_report[[i]].applymap(separator_converter)
        except:
            print(f'Column {i} is not a float')
    base_report, count = discrepancy_blk_attribution(base_report)
    print(f'Mismatch for {count} blk files')                
    # Check in case of presence of BLK file with no correspondance in BaseReport
    # In case of presence, they are removed from all_blks
    security_check_blks = set(all_blks).difference(set(list(base_report.loc[base_report['Preceding Event IT'] == 'FixCorrect','BLK Names'])))
    if len(security_check_blks) != 0:
        print(f'Length of all_blks before popping off the elements: {len(all_blks)}')
        for j in list(security_check_blks):
            all_blks.remove(j)
            print(f'{j} popped out')
        print(f'Length of all_blks list after popping off from get_basereport: {len(all_blks)}')

    print('BaseReport properly loaded!')
    print('BaseReport loading time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    start_time = datetime.datetime.now().replace(microsecond=0)
    #time_stamp, piezo, heart_beat = al.get_analog_signal(path_session, base_report, name_report = 'SignalData.csv')
    #print('Piezo and Heart Beat signals properly loaded!')
    #print('Analogic signals loading time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    base_report = base_report.loc[(base_report['Preceding Event IT'] == 'FixCorrect')]
    new_blks = all_blks     
    return base_report, new_blks

def get_basereport_header(BaseReport_path, header_dimension = 19):    
    '''
    BaseReport header builder.
    '''
    f = open(BaseReport_path, 'r')
    dict_ = {}
    for i in range(0,header_dimension-1):
        tmp = f.readline()
        tmp = tmp.split('\n')[0].split(';')
        if '*' not in tmp[0]:
            try:
                if ',' in tmp[1]:
                    a = tmp[1].split(',')
                else:
                    a = tmp[1].split('\n')

                dict_[tmp[0]] = separator_converter(a[0])
            except:
                dict_[tmp[0]] = a[0]
            if tmp[0] == 'Date':
                # Datetime day, month, year, hour, minute, seconds
                format_str = '%d/%m/%Y'
                dict_['Date'] =  (datetime.datetime.strptime(dict_['Date'], format_str).date())
                if ',' in str(tmp[3]):
                    asda = str(tmp[3]).split(',')[0]
                else:
                    asda = str(tmp[3])
                try:
                    dict_['Date'] = datetime.datetime.strptime( asda + ':00', '%H:%M:%S').replace(year=dict_['Date'].year,month=dict_['Date'].month,day=dict_['Date'].day)
                except:
                    pass
        if 'Stim_temp' in tmp[0]:
            dict_[tmp[0]] = float(tmp[1].replace(',', '.'))
                
    dict_['Export Log Files'] = bool(dict_['Export Log Files'])
    return dict_


def get_affidability_mask(base_correct, toogles_starts, toogles_ends, timestamps, affidability_range = 50):
    if len(base_correct) != len(toogles_starts):
        print('Mismatch in length of BaseReport and Toogle tracks considered')
        return 
    
    analog_timestamp_array = timestamps

    alleged_starts = analog_timestamp_array[toogles_starts]
    alleged_end    = analog_timestamp_array[toogles_ends]

    affidable_ends = np.where((np.array(base_correct['Onset Time_ End Stim']) > (alleged_end - affidability_range//2)) & 
                              ((np.array(base_correct['Onset Time_ End Stim']) < (alleged_end + affidability_range//2))))[0]

    affidable_starts = np.where((np.array(base_correct['Onset Time_ Pre Stim']) > (alleged_starts - affidability_range//2)) & 
                                ((np.array(base_correct['Onset Time_ Pre Stim']) < (alleged_starts + affidability_range//2))))[0]

    affidability_mask = np.zeros((len(base_correct)))
    affidability_mask[affidable_starts] = 1
    return affidability_mask, affidable_starts, affidable_ends


def get_signal_on_timewindow(central_timebin, signal, threshold = -1, window_length = 3000, safety_range = 0, adjustment_value = 0):
    central_timebin = int(central_timebin)
    window_side     = int(window_length//2)
    first_id        = int((central_timebin - adjustment_value - window_side))
    last_id         = int((central_timebin - adjustment_value + window_side))
    if (first_id)<0:
        first_id = 0
    print(first_id, last_id)
    tmp             = signal[(first_id):(last_id)]
    indeces         = np.where(tmp>threshold)[0]
    print(indeces)
    try:
        (start, end)    = ((first_id + indeces[0]  - safety_range, 
                            first_id + indeces[-1] + safety_range))
    except:
        (start, end)    = ((None, None))
    return start, end


def get_analog_signal(session_path, BaseReport, toogle_v = False, safety_range = 30, name_report = 'SignalData.csv'):
    analog_sign_path = utils.find_thing(name_report, session_path, what = 'file')
    SignalData = pd.read_csv(analog_sign_path[0], sep=';', header=2)
    #signal_data_header = ['Timestamp', 'TrigInStim', 'TrigHeart', 'Toogle', 'TrigReadyStim', 'CtrlShutter', 'Heart', 'Piezo', 'None']
    # Timestamp importing
    temp = SignalData[['Timestamp', 'Dev1/ai0', 'Dev1/ai2', 'Dev1/ai5', 'Dev1/ai6']].applymap(separator_converter)
    analog_timestamp_array = np.array(temp['Timestamp'])
    # TrigInStim
    analog_ai1_array = np.array(temp['Dev1/ai0'])
    # Toogle
    analog_ai3_array = np.array(temp['Dev1/ai2'])
    # HeartBeat
    analog_ai5_array = np.array(temp['Dev1/ai5'])
    # Piezo
    analog_ai6_array = np.array(temp['Dev1/ai6'])
    
    # Three csvs are synchronized
    #t = BaseReport[['Onset Time_ Pre Trial', 'Onset Time_ End Trial']]#.applymap(separator_converter)
    t         = BaseReport[['Onset Time_ Pre Stim',  'Onset Time_ End Stim']]#.applymap(separator_converter)
    onset_pre = t['Onset Time_ Pre Stim'].tolist()
    onset_end = t['Onset Time_ End Stim'].tolist()
    tracks_1, tracks_3, tracks_5, tracks_6, time = [], [], [], [], []
    temp_list         = []
    if toogle_v:

        center_time = np.mean((np.array((t['Onset Time_ End Stim'], 
                                         t['Onset Time_ Pre Stim']))), axis = 0)
#         print(t['Onset Time_ End Stim'][:5])
#         print(t['Onset Time_ Pre Stim'][:5])
#         print(center_time[:5])
        
        tmp_ai3_array    = (analog_ai3_array-np.min(analog_ai3_array, axis=0, keepdims=True))
        analog_ai3_array =  tmp_ai3_array/np.max(tmp_ai3_array, axis=0, keepdims=True)
        for center in center_time:
#             try:
            tmp = get_signal_on_timewindow(center, 
                                           analog_ai3_array, 
                                           threshold = .3, 
                                           adjustment_value=int(analog_timestamp_array[0]))
#             except:
#                 print('Exception: Onset and Offset Trial Times used')
#                 tmp = (int(onset_pre[n_c]+int(analog_timestamp_array[0])), int(onset_end[n_c]+int(analog_timestamp_array[0])))
            temp_list.append(tmp)
        toogle_pre = list(list(zip(*temp_list))[0])
        toogle_end = list(list(zip(*temp_list))[1])
        for pre, end in zip(toogle_pre, toogle_end):
            p = pre - safety_range
            e = end + safety_range
            tracks_6.append(analog_ai6_array[p:e])
            # HeartBeat
            tracks_5.append(analog_ai5_array[p:e])
            # Toogle
            tracks_3.append(analog_ai3_array[p:e])
            # TrigInStim
            tracks_1.append(analog_ai1_array[p:e])
            # TimeStamp
            time.append(analog_timestamp_array[p:e]) 
        affidability = get_affidability_mask(t, toogle_pre, toogle_end, analog_timestamp_array)

        return time, tracks_6, tracks_5, tracks_3, tracks_1, ((toogle_pre, toogle_end)), affidability
            
    else:    
        for pre, end in zip(onset_pre, onset_end):
            # Piezo
            tracks_6.append(signal_cutter(analog_timestamp_array, analog_ai6_array, pre, end))
            # HeartBeat
            tracks_5.append(signal_cutter(analog_timestamp_array, analog_ai5_array, pre, end))
            # Toogle
            tracks_3.append(signal_cutter(analog_timestamp_array, analog_ai3_array, pre, end))
            # TrigInStim
            tracks_1.append(signal_cutter(analog_timestamp_array, analog_ai1_array, pre, end))
            # TimeStamp
            time.append(signal_cutter(analog_timestamp_array, analog_timestamp_array, pre, end))
  
        return time, tracks_6, tracks_5, tracks_3, tracks_1, ((None, None)), None


def get_eye_signal(base_report, eyetrack):
    only_blks = base_report.loc[base_report['Preceding Event IT'] == 'FixCorrect']['Total Trial Number'].tolist()
    dict_tracks = dict()

    for i in only_blks:
        #eyetrack.loc[eyetrack['Trial'] == i]
        # Getting the event list per trial i
        events = eyetrack.loc[eyetrack['Trial'] == i]['Current Event'].tolist()
        # Beginning of the stimulation
        start_track = events.index('r>FixationTask>Trial>PreStim')
        # End of the stimulation
        end_track = events.index('r>FixationTask>EndTrial')
        tmp = eyetrack.loc[eyetrack['Trial'] == i, ['Gaze CVX', 'Gaze CVY']].iloc[start_track:end_track].applymap(separator_converter)
        dict_tracks[i] = (tmp['Gaze CVX'].tolist(), 
                          tmp['Gaze CVY'].tolist())
    return dict_tracks

def get_grey_frames(png_files_path, cond_id):
    start_time = datetime.datetime.now().replace(microsecond=0)
    all_pngs =  [f.name for f in os.scandir(png_files_path) if (f.is_file()) and (f.name.startswith("Cond")) and (f.name.split('Cond') and int(f.name.split('-')[0].split('Cond')[1]) == cond_id)]

    a = np.zeros((len(all_pngs), 256, 256, 3))
    for i, png in enumerate(all_pngs):
        im = imageio.imread(os.path.join(png_files_path, png))
        a[i, :, :, :] = im

    out = list()
    final_out = list()

    for count, i in enumerate(range(len(a)-1, 0, -1)):
        if not np.array_equal(a[count], a[count+1]):
                out.append(count+1)
        if not np.array_equal(a[i], a[i-1]):
                final_out.append(count+1)
    print('Time for png evaluation: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    return out[0], final_out[0], a.shape[0]

def separator_converter(s):
    '''
    Utility method: separator corrector for BaseReport.csv files
    '''
    try:
        tmp = float(s.replace(',', '.'))
    except:
        tmp = float(s)
    return tmp

def delete_chars(s, to_substitute = '"', substitute = ''):
    '''
    Utility method: separator corrector for BaseReport.csv files
    '''
    try:
        tmp = s.replace(to_substitute,substitute)
    except:
        tmp = s
    return tmp

def signal_cutter(analog_timestamp_array, signal_array, pre, end, safe_value = 100):
    '''
    analog_timestamp_array is the time array for each trial
    signal_array could be piezo or heart
    pre is the OnSet Time Pre Stim -start of recording-
    end is the Time Go Input -end of the recording-
    '''
    start_trial = np.argmin(np.abs(analog_timestamp_array - pre))
    end_trial = np.argmin(np.abs(analog_timestamp_array - (end + safe_value)))#   +1sec for safety -in trial timing-, 0.1 sec for safety in stimulus timing
    cut_signal = signal_array[start_trial:end_trial]
    return cut_signal

def sorting_from_first(BaseReport, blks): #, start_session):
    all_datetimes = [datetime.datetime.strptime(i.split('_')[2] + i.split('_')[3], '%d%m%y%H%M%S') for i in blks]
    first = datetime.datetime.strptime(blks[0].split('_')[2] + blks[0].split('_')[3], '%d%m%y%H%M%S')
    #tmp =  - head['Date']
    temporary = (((BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', ['Onset Time_ End Stim']].iloc[:].applymap(separator_converter)))/1000)*datetime.timedelta(seconds=1)
    temporary = temporary - temporary.iloc[0]
    estimated_dates = temporary + first  # adding a row
    pr_time = pd.to_datetime(estimated_dates['Onset Time_ End Stim']).tolist()
    pr_time = [str(i) for i in pr_time]
    dt_object = np.array([datetime.datetime.strptime(i.split('.')[0],  "%Y-%m-%d %H:%M:%S") for i in pr_time])
    tmp = list()
    tmp_ = list()
    for i in all_datetimes:
        tmp_.append(np.abs(dt_object - i))
        tmp.append(np.argmin(np.abs(dt_object - i)))
    tmp = np.array(tmp)
    a = ['Missing'] * len(BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect'])
    for n, i in enumerate(tmp):
        a[i] = blks[n]
    BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = a
    return BaseReport
