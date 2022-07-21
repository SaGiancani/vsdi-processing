import datetime, imageio, os, utils
import middle_process as mp
import numpy as np
import pandas as pd


class Trial:
    def __init__(self, report_series_trial, heart, piezo, session_path, blank_cond, index, n_frames, log = None, stimulus_fr = None, zero_fr = None, time_res = 10, blk_file = None):
        self.blk = blk_file
        self.index = index
        self.name = report_series_trial['BLK Names']
        self.condition = int(report_series_trial['IDcondition'])
        self.fix_correct = report_series_trial['Preceding Event IT'] == 'FixCorrect'
        self.correct_behav = report_series_trial['behav Correct'] == 1
        if self.fix_correct and self.correct_behav and self.condition != blank_cond:
            self.behav_latency = int(report_series_trial['Onset Time_ Behav Correct']) -  int(report_series_trial['Onset Time_ Behav Stim']) - 500
        else:
            self.behav_latency = None
        self.id_trial = int(report_series_trial['Total Trial Number']) - 1
        pngfile_path = utils.find_thing('PNGfiles', os.path.join(session_path, 'sources'))

        if self.condition != blank_cond:
            try:
                grey_frames_start, grey_frames_end, _ = get_grey_frames(pngfile_path, self.condition)
                #n_pngs = pngs_shape[0]
            except:
                if log is None:
                    print('Issue with PNGfiles: standard values 5 grey frames pre stimulus and 5 post will be used')
                else:
                    log.info('Issue with PNGfiles: standard values 5 grey frames pre stimulus and 5 post will be used')
                grey_frames_start = 5
                grey_frames_end = 5

            if stimulus_fr is None:
                # Total registration time (End registration (PNG flow + Post Stimulus time) - Start registration) - Starting Grey frames*temporal resolution - Ending Grey frames*temporal resolution + 25 ms latency 
                stimulus_fr = round((report_series_trial['Onset Time_ End Stim'] - report_series_trial['Onset Time_ Stim'] - grey_frames_end*time_res - grey_frames_start*time_res + 25)/time_res)         
            
            self.FOI = stimulus_fr

            if zero_fr is None:
                #PreStimulus Time + nGreyFrames*10ms + 25ms Response Latency
                zero_fr = round(((report_series_trial['Onset Time_ Stim'] - report_series_trial['Onset Time_ Pre Stim']) + 25 + grey_frames_start*time_res)/time_res) 
            
            self.zero_frames = zero_fr

        else:
            self.zero_frames = 20
            self.FOI = n_frames
            
        print(self.zero_frames)
        self.heart_signal = heart
        self.piezo_signal = piezo

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
            print(tris)
        # Consider the BLK names, in case of FixCorrect preceding event IT
        BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = all_blks
    return BaseReport, tris


def get_basereport(session_path, all_blks, name_report = 'BaseReport.csv', header_dimension = 19):
    '''
    Load the BaseReport
    '''
    BaseReport_path = utils.find_thing(name_report, session_path, what = 'file')
    BaseReport = pd.read_csv(BaseReport_path[0], sep=';', header=header_dimension)
    #Adding BLK Names columns to the dataframe
    BaseReport, tris = add_blknames2basereport(BaseReport, all_blks)
    return BaseReport, tris

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
                dict_[tmp[0]] = separator_converter(tmp[1].split('\n')[0])
            except:
                dict_[tmp[0]] = tmp[1].split('\n')[0]
            if tmp[0] == 'Date':
                # Datetime day, month, year, hour, minute, seconds
                format_str = '%d/%m/%Y'
                dict_['Date'] =  (datetime.datetime.strptime(dict_['Date'], format_str).date())
                dict_['Date'] = datetime.datetime.strptime(str(tmp[3]) + ':00', '%H:%M:%S').replace(year=dict_['Date'].year,month=dict_['Date'].month,day=dict_['Date'].day)
    dict_['Export Log Files'] = bool(dict_['Export Log Files'])
    return dict_


def get_analog_signal(session_path, BaseReport, name_report = 'SignalData.csv'):
    analog_sign_path = utils.find_thing(name_report, session_path, what = 'file')
    SignalData = pd.read_csv(analog_sign_path[0], sep=';', header=2)
    # Timestamp importing
    temp = SignalData[['Timestamp','Dev1/ai5', 'Dev1/ai6' ]].applymap(separator_converter)
    analog_timestamp_array = np.array(temp['Timestamp'])
    # HeartBeat
    analog_ai5_array = np.array(temp['Dev1/ai5'])
    # Piezo
    analog_ai6_array = np.array(temp['Dev1/ai6'])
    # Three csvs are synchronized
    t = BaseReport[['Onset Time_ Pre Trial', 'Onset Time_ End Trial']]#.applymap(separator_converter)
    onset_pre = t['Onset Time_ Pre Trial'].tolist()
    onset_end = t['Onset Time_ End Trial'].tolist()
    tracks_6, tracks_5 = [], []
    
    for pre, end in zip(onset_pre, onset_end):
        # Piezo
        tracks_6.append(signal_cutter(analog_timestamp_array, analog_ai6_array, pre, end))
        # HeartBeat
        tracks_5.append(signal_cutter(analog_timestamp_array, analog_ai5_array, pre, end))
  
    return tracks_6, tracks_5


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
        tmp = float(s.replace(',','.'))
    except:
        tmp = float(s)
    return tmp    


def signal_cutter(analog_timestamp_array, signal_array, pre, end):
    '''
    analog_timestamp_array is the time array for each trial
    signal_array could be piezo or heart
    pre is the OnSet Time Pre Stim -start of recording-
    end is the Time Go Input -end of the recording-
    '''
    start_trial = np.argmin(np.abs(analog_timestamp_array - pre))
    end_trial = np.argmin(np.abs(analog_timestamp_array - (end + 1000)))#   +1sec for safety
    cut_signal = signal_array[start_trial:end_trial]
    return cut_signal
    
