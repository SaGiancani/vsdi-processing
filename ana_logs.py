import datetime, utils
import middle_process as mp
import numpy as np
import pandas as pd

class Trial:
    def __init__(self, report_series_trial):
        self.blk = None
        self.name = report_series_trial['BLK Names']


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
        sorted_list = sorted(all_blks, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
        # Consider the BLK names, in case of FixCorrect preceding event IT
        BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = sorted_list
    except:
        print('Mismatch between BLK files and FixCorrect trials number')
        print('This strategy could solve the problem. It has to be checked\n')    
        cds = BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'IDcondition'].tolist()
        sorted_list = sorted(all_blks, key=lambda t: datetime.datetime.strptime(t.split('_')[2] + t.split('_')[3], '%d%m%y%H%M%S'))
        print(len(sorted_list))
        print(len(cds))
        if len(sorted_list)<len(cds):
            # If there is mismatch between the condition id in BaseReport and condition id in the BLK filename
            # It stores index, condition number, and BLK filename of the mismatch.
            tris = next( (idx, x, y) for idx, (x, y) in enumerate(zip(cds, sorted_list)) if x!= int(y.split('_C')[1][:2]))
        print(tris)
        sorted_list.insert(tris[0], '')
        # Consider the BLK names, in case of FixCorrect preceding event IT
        BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = sorted_list
    return BaseReport


def get_basereport(session_path, name_report = 'BaseReport.csv'):
    '''
    Load the BaseReport
    '''
    BaseReport_path = utils.find_file(name_report, session_path)
    BaseReport = pd.read_csv(BaseReport_path[0], sep=';', header=19)
    #Adding BLK Names columns to the dataframe
    BaseReport = add_blknames2basereport(BaseReport, mp.get_all_blks(session_path))
    return BaseReport


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
                dict_[tmp[0]] = toogle_from_object(tmp[1].split('\n')[0])
            except:
                dict_[tmp[0]] = tmp[1].split('\n')[0]
    dict_['Export Log Files'] = bool(dict_['Export Log Files'])
    format_str = '%d/%m/%Y'
    dict_['Date'] = (datetime.datetime.strptime(dict_['Date'], format_str).date())
    return dict_


def toogle_from_object(s):
    '''
    Utility method
    '''
    try:
        tmp = float(s.replace(',','.'))
    except:
        tmp = float(s)
    return tmp    