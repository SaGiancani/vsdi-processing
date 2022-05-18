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
            elif (len(all_blks)>len(cds)):
                all_blks.pop(tris[0])
                print(f'File {tris[0]} is deleted')            
            print(tris)
        # Consider the BLK names, in case of FixCorrect preceding event IT
        BaseReport.loc[BaseReport['Preceding Event IT'] == 'FixCorrect', 'BLK Names'] = all_blks
    return BaseReport, tris


def get_basereport(session_path, name_report = 'BaseReport.csv', header_dimension = 19):
    '''
    Load the BaseReport
    '''
    BaseReport_path = utils.find_file(name_report, session_path)
    BaseReport = pd.read_csv(BaseReport_path[0], sep=';', header=header_dimension)
    #Adding BLK Names columns to the dataframe
    BaseReport, tris = add_blknames2basereport(BaseReport, mp.get_all_blks(session_path, sort = True))
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