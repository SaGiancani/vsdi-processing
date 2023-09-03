import argparse
import blk_file as blk
import middle_process as md
import os
import process_vsdi as process
import re, retinotopy
import datetime
import data_visualization as dv
import ana_logs as al
import utils

import numpy as np
import matplotlib.pyplot as plt

import datetime

PROTOCOL = 'Retinotopy'


class Retino_IOI:
    def __init__(self,
                 path_md, 
                 zero_frames = None,
                 deblank_switch=False,
                 conditions_id =None,
                 logs_switch =False,  
                 base_report_name= 'BaseReport.csv',
                 base_head_dim = 20, 
                 condid = None, 
                 selection_switch = False, 
                 data_vis_switch = True, 
                 end_frame = None,
                 selection_per_condition = None,
                 filename_particle = 'int_C',
                 **kwargs):
        
        self.path_files = path_md
        self.path_session = get_session_path_from_md(self.path_files)
        print(self.path_session)
        self.name_session = retinotopy.get_session_id_name(self.path_session)
        print(self.name_session)
        self.filename_particle = filename_particle
        self.zero_frames = zero_frames
        self.all_blks = md.get_all_blks(self.path_session, sort = True)
        self.cond_dict = self.get_cond_names()
        self.cond_names = list(self.cond_dict.values())

        self.relative_stimulus_coordinates , _, _ = get_relative_coordinates(self.cond_names)
        self.blank_id = md.get_blank_id(self.cond_names, cond_id=condid)
        if conditions_id is None:
            self.cond_ids = md.get_condition_ids(self.all_blks, filename_particle = self.filename_particle)
        else:
            self.cond_ids= list(set(self.cond_ids+[self.blank_id]))
        self.center_render_stimulus = get_absolute_coordinates(self.path_session, name_report = base_report_name, hd_dim = base_head_dim)
        self.selection_switch = selection_switch
        self.selection_per_condition = selection_per_condition

    def get_cond_names(self):
        tmp = utils.find_thing('labelConds.txt', self.path_session)
        # If also with find_thing there is no labelConds.txt file, than loaded as name Condition n#
        if len(tmp) == 0:
            self.log.info('Check the labelConds.txt presence inside the session folder and subfolders')
            cds = self.cond_ids
            return {j+1:'Condition ' + str(c) for j, c in enumerate(cds)}
        # else, load the labelConds from the alternative path
        else :
            with open(tmp[0]) as f:
                contents = f.readlines()
            return  {j+1:i.split('\n')[0] for j, i in enumerate(contents) if len(i.split('\n')[0])>0}      

    def run_ioi_retinotopic_analysis(self):
        dict_data = get_load_cond(self.path_files, self.cond_dict, self.blank_id, selection=self.selection_switch, cond_selection=self.selection_per_condition)
        tmp = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis = os.path.join(PROTOCOL + '_IOI', self.name_session))
        utils.inputs_save(dict_data, os.path.join(tmp, f'data_session_{self.name_session}'))
        return
    
def get_session_path_from_md(path_md):
    return path_md.split('derivatives')[0][:-1]    
            
def get_absolute_coordinates(session_path, name_report = 'BaseReport.csv', hd_dim = 20):
    basereport_path = utils.find_thing(name_report, session_path, what = 'file')
    if len(basereport_path)>1:
        print(f'{len(basereport_path)} BaseReport are found')
        basereport_path = [i for i in basereport_path if 'bug' not in i.lower()]
    print(basereport_path)
    header_br = al.get_basereport_header(basereport_path[0], header_dimension=hd_dim) 
    print(header_br) 
    x = header_br['C V X Stim_temp']
    y = header_br['C V Y Stim_temp']
    return (x, y)
 
def get_relative_coordinates(cond_names):
    print(cond_names)
    coords_stimulus = [(float(i.split('posX')[1][0:4].replace(',', '.')), float(i.split('posY')[1][0:].replace(',', '.'))) for i in cond_names if ('blank' not in i)]
    xs = list(set(list(list(zip(*coords_stimulus))[0])))
    xs.sort()
    ys = list(set(list(list(zip(*coords_stimulus))[1])))
    ys.sort()
    return coords_stimulus, xs, ys

def get_cond(path_md, cond_name, cd_id, zero_frame = 10, selection = False, cond_selection = None):
    # Blank instance
    cd = md.Condition(condition_name = cond_name, condition_numb=cd_id) 
    cd.load_cond(os.path.join(path_md,'md_data_'+cond_name))
    all_raw = np.copy(cd.binned_data)
    dfs = np.array([process.deltaf_up_fzero(i, zero_frame, deblank = True, blank_sign = None) for i in all_raw])

    if selection:
        if cond_selection is None:
            # Selection blank trials
            indeces = cd.autoselection
        else:
            indeces = np.zeros(len(dfs))
            indeces[cond_selection] = 1
        indeces_blank = np.where(indeces == 1)[0]
        dfs = dfs[indeces_blank, :, :, :]
    del cd
    return all_raw, dfs

def get_load_cond(path_md, dict_cond, blank_id, selection = False, cond_selection = None):

    # In case of existence of a matrix of selection for all the conditions then:
    if cond_selection is not None:
        c_b_sel = cond_selection[0, :]
        c_o_sel = cond_selection[1:,:]
    # Otherwise keep the None value
    else:
        c_b_sel = None
        c_o_sel = None

    dict_data = dict()

    # Load the blank first
    raw_blank, df_blank = get_cond(path_md, dict_cond[blank_id], blank_id, selection = selection, cond_selection = c_b_sel)
    dict_data[dict_cond[blank_id]] = [np.nanmean(raw_blank, axis = 0), np.nanmean(df_blank, axis = 0)]  

    # Then the other conditions
    for k,v in dict_cond.items():
        if k!= blank_id:
            raw_cd, df_cd = get_cond(path_md, dict_cond[k], k, selection = selection, cond_selection = c_o_sel)
            average_dfs = np.nanmean(df_cd, axis = 0)
            average_raws = np.nanmean(raw_cd, axis = 0)
            dict_data[v] = [average_raws, average_dfs]  
    
    del raw_cd, df_cd

    return dict_data

def set_cond_dict_per_coordinates(choosen_coord, cond_dict):
    '''
    choosen_coord: char, either 'x' or 'y'
    '''
    if (choosen_coord != 'x') and (choosen_coord != 'y'):
        print('Error: Wrong char picked')
    
    cond_names = list(cond_dict.values())
    _, x, y = get_relative_coordinates(cond_names)

    tmp_dict = {k: v for k,v in cond_dict.items() if 'blank' not in v}

    if choosen_coord == 'x':
        new_cond_dict = {k:v for j in x for k,v in tmp_dict.items() if ((float(v.split('posX')[1][0:4].replace(',', '.')) == j))}
    elif choosen_coord == 'y':
        new_cond_dict = {k:v for j in y for k,v in tmp_dict.items() if ((float(v.split('posY')[1][0:].replace(',', '.')) == j))}

    return new_cond_dict


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Launching retinotopy analysis pipeline')

    parser.add_argument('--path_md', 
                        dest='path_md',
                        type=str,
                        required=True,
                        help='The middle process datafolder path')
                        
    parser.add_argument('--mask_rad', 
                        dest='mask_rad',
                        type=int,
                        default = 75,
                        required=False,
                        help='Time course window dimension -pixels radius-') 

    parser.add_argument('--start_time', 
                        dest='starting_time',
                        type=int,
                        default = 3500,
                        required=False,
                        help='Starting time: start stimulus response in cortex') 

    parser.add_argument('--end_time', 
                        dest='stop_time',
                        type=int,
                        default = 4500,
                        required=False,
                        help='Stop time: end stimulus response in cortex') 
    
    parser.add_argument('--sample_rate', 
                        dest='sampling_rate',
                        type=int,
                        default = 100,
                        required=False,
                        help='Sampling rate in acquisition') 

    parser.add_argument('--frame_extension', 
                        dest='frame_time_ext',
                        type=int,
                        default = 30,
                        required=False,
                        help='Time duration of one frame') 

    parser.add_argument('--tc_vis', 
                        dest='flag_timecourses', 
                        action='store_true')
    parser.add_argument('--no-tc_vis', 
                        dest='flag_timecourses', 
                        action='store_false')
    parser.set_defaults(flag_timecourses=True)  
    
    parser.add_argument('--store', 
                        dest='store_switch',
                        action='store_true')
    parser.add_argument('--no-store', 
                        dest='store_switch', 
                        action='store_false')
    parser.set_defaults(store_switch=True)   

    start_process_time = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args()

    print(args)

    workspace_retino = Retino_IOI(args.path_md)
    workspace_retino.run_ioi_retinotopic_analysis()
