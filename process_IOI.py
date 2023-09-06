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

class ReceptiveField:
    def __init__(self,
                 path_session,
                 condition_names,
                 base_report_name,
                 base_report_header):
        
        # Stimulus coordinates: relative and absolute
        self.relative_stimulus_coordinates , self.xs, self.ys = get_relative_coordinates(condition_names)
        self.center_render_stimulus = get_absolute_coordinates(path_session, name_report = base_report_name, hd_dim = base_report_header)
        print(f'Center of stimulus: {self.center_render_stimulus}')

class RFWorkspace:
    def __init__(self,
                 path_md, 
                 zero_frames = None,
                 deblank_switch=False,
                 conditions_id =None,
                 base_report_name= 'BaseReport.csv',
                 base_head_dim = 20, 
                 condid = None, 
                 selection_switch = False, 
                 data_vis_switch = True, 
                 operation_maps = 'both',
                 dimension_to_analyze = 'x',
                 store_switch = True, 
                 selection_per_condition = None,
                 filename_particle = 'int_C',
                 start_time = 3500, 
                 end_time = 4500,
                 frame_time_extension = 30,
                 sampling_rate = 100,
                 **kwargs):
        
        self.path_files = path_md
        self.path_session = get_session_path_from_md(self.path_files)
        self.name_session = retinotopy.get_session_id_name(self.path_session)
        print(self.name_session)
        self.filename_particle = filename_particle
        self.zero_frames = zero_frames
        self.all_blks = md.get_all_blks(self.path_session, sort = True)
        self.cond_dict = self.get_cond_names()
        self.cond_names = list(self.cond_dict.values())

        self.blank_id = md.get_blank_id(self.cond_names, cond_id=condid)
        if conditions_id is None:
            self.cond_ids = md.get_condition_ids(self.all_blks, filename_particle = self.filename_particle)
        else:
            self.cond_ids= list(set(self.cond_ids+[self.blank_id]))
        self.selection_per_condition = selection_per_condition

        # Switches
        self.selection_switch = selection_switch
        self.store_switch = store_switch
        self.operation_flag = operation_maps
        self.dimension_to_analyze = dimension_to_analyze
        self.visualization_switch = data_vis_switch

        # Instance receptive field
        self.rf = ReceptiveField(self.path_session, self.cond_names, base_report_name, base_head_dim)
        # Timing for extracting map
        self.start_time = start_time
        self.end_time = end_time
        self.frame_time_extension = frame_time_extension
        self.sampling_rate = sampling_rate

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

    def run_ioi_rf_analysis(self):
        if (self.operation_flag == 'cocktail') or (self.operation_flag == 'both'):
            sorted_cond_dict = set_cond_dict_per_coordinates(self.dimension_to_analyze, self.cond_dict)
        else:
            sorted_cond_dict = self.cond_dict

        averaged_dfs, averaged_raws = get_load_cond(self.path_files, 
                                                    sorted_cond_dict, 
                                                    self.blank_id,
                                                    self.cond_dict[self.blank_id], 
                                                    selection=self.selection_switch, 
                                                    cond_selection=self.selection_per_condition)

        print(f'Shape of outcome matrix: {averaged_dfs.shape}')
        dict_output =  operation_among_conditions(averaged_dfs, 
                                                  sorted_cond_dict, 
                                                  self.start_time, 
                                                  self.end_time, 
                                                  self.frame_time_extension, 
                                                  self.rf.center_render_stimulus,
                                                  type = self.operation_flag, 
                                                  coordinate = self.dimension_to_analyze)
    
        if self.store_switch:
            tmp = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis = os.path.join(PROTOCOL + '_IOI', self.name_session))
            np.save(os.path.join(tmp, f'data_session_df_{self.dimension_to_analyze}_{self.start_time}_{self.end_time}ms_{self.name_session}'), averaged_dfs)
            np.save(os.path.join(tmp, f'data_session_raw_{self.dimension_to_analyze}_{self.start_time}_{self.end_time}ms_{self.name_session}'), averaged_raws)
            utils.inputs_save(dict_output, os.path.join(tmp, f'data_session_dict_{self.dimension_to_analyze}_{self.start_time}_{self.end_time}ms_{self.name_session}'))
        
        if self.visualization_switch:
            titles = list(dict_output.keys())
            data = np.array(list(dict_output.values()))
            dv.whole_time_sequence(data, mask = np.ones((data.shape[1], data.shape[2]), dtype = bool),
                                   blbs = [np.empty((data.shape[1], data.shape[2]))]*60, 
                                   n_columns = 4, titles = titles, mappa = 'gray', 
                                   max_bord=1.005, min_bord=.998,
                                   store_path=dv.STORAGE_PATH, 
                                   name_analysis_= os.path.join(PROTOCOL + '_IOI', self.name_session),
                                   name = f'Maps {self.dimension_to_analyze} dimension _ {self.start_time}_{self.end_time}ms_{self.name_session}', ext='pdf')            
        return
    
def get_session_path_from_md(path_md):
    return path_md.split('derivatives')[0][:-1]    
            
def get_absolute_coordinates(session_path, name_report = 'BaseReport.csv', hd_dim = 20):
    basereport_path = utils.find_thing(name_report, session_path, what = 'file')
    if len(basereport_path)>1:
        print(f'{len(basereport_path)} BaseReport are found')
        basereport_path = [i for i in basereport_path if 'bug' not in i.lower()]
    header_br = al.get_basereport_header(basereport_path[0], header_dimension=hd_dim) 
    x = header_br['C V X Stim_temp']
    y = -header_br['C V Y Stim_temp']
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

def get_load_cond(path_md, dict_cond, blank_id, blank_name, selection = False, cond_selection = None):

    # In case of existence of a matrix of selection for all the conditions then:
    if cond_selection is not None:
        c_b_sel = cond_selection[0, :]
        c_o_sel = cond_selection[1:,:]
    # Otherwise keep the None value
    else:
        c_b_sel = None
        c_o_sel = None

    # dict_data = dict()

    # Load the blank first
    print('\n'+blank_name )
    raw_blank, df_blank = get_cond(path_md, blank_name, blank_id, selection = selection, cond_selection = c_b_sel)
    average_blank_raw = np.nanmean(raw_blank, axis = 0)
    average_blank_df = np.nanmean(df_blank, axis = 0)
    # dict_data[dict_cond[blank_id]] = [average_blank_raw, average_blank_df]

    # Instance output matrix  
    number_conds = len([i for i in dict_cond.values() if 'blank' not in i])
    output_data_matrix_df = np.empty((number_conds, average_blank_df.shape[0], average_blank_df.shape[1], average_blank_df.shape[2]))
    output_data_matrix_df[:] = np.nan
    output_data_matrix_raw = np.copy(output_data_matrix_df)
    del raw_blank, df_blank

    # Then the other conditions
    counter = 0
    for k,v in dict_cond.items():
        print('\n'+v )
        if k!= blank_id:
            raw_cd, df_cd = get_cond(path_md, dict_cond[k], k, selection = selection, cond_selection = c_o_sel)
            average_dfs = np.nanmean(df_cd, axis = 0)
            average_raws = np.nanmean(raw_cd, axis = 0)
            # dict_data[v] = [average_raws, average_dfs]
            output_data_matrix_df[counter, :, :, :] = average_dfs
            output_data_matrix_raw[counter, :, :, :] = average_raws
            counter +=1
    
        del raw_cd, df_cd
    
    # Concatenating blank and actual conditions together
    output_data_matrix_df = np.concatenate((average_blank_df.reshape(1, average_blank_df.shape[0], average_blank_df.shape[1], average_blank_df.shape[2]), output_data_matrix_df), axis=0) 
    output_data_matrix_raw = np.concatenate((average_blank_raw.reshape(1, average_blank_raw.shape[0], average_blank_raw.shape[1], average_blank_raw.shape[2]), output_data_matrix_raw), axis=0) 

    return output_data_matrix_df, output_data_matrix_raw

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

def operation_among_conditions(maps, sorted_cds_dictionary, start_time, stop_time, frame_time_ext, absolute_center, type = 'cocktail', coordinate = 'x'):
    '''
    type: str, it can be 'cocktail', 'regular', 'both'. 
          'cocktail' for division against other conditions, 'regular' for blank condition.
          'both' it permorfs both
    coordinate: str, it can be 'x', 'y' or None. It is active only if type = 'cocktail'
    '''
    print(f'The center of the stimulation is at {absolute_center}dva ')
    tmp_data = maps[1:, :, :, :]
    tmp_blank = maps[0, :, :, :]
    
    if (type == 'cocktail') or (type == 'both'):
        coords, x, y = get_relative_coordinates(list(sorted_cds_dictionary.values()))
        if coordinate == 'x':
            n_considered_conds = len(coords)/len(x)
            picked_cord = np.array(x) + absolute_center[0]
            print_todo = 'columns'

        elif coordinate == 'y':
            n_considered_conds = len(coords)/len(y)
            picked_cord = np.array(y) + absolute_center[1]
            print_todo = 'rows'

        # Loop for averaging every n_considered_conds
        indeces = np.arange(0, len(coords)+1, n_considered_conds, dtype = int)
        output_matrix_cocktail = np.array([np.nanmean(tmp_data[indeces[i-1]:indeces[i], start_time//frame_time_ext:stop_time//frame_time_ext, :, :], axis = (0, 1)) for i in range (1, len(indeces))])
        print(f'Shape of {print_todo} normalization output matrix: {output_matrix_cocktail.shape}')
        cocktail_dict = {f'{coordinate}: {round(a, 2)}/{round(i, 2)}': b/j for a,b in zip(picked_cord, output_matrix_cocktail) for i, j in zip(picked_cord, output_matrix_cocktail)}
        data_dict = cocktail_dict
        print(f'Normalization between {print_todo} computed!')

    if (type == 'regular') or (type == 'both'):
        # Adjustment for absolute center
        coords = list(zip(np.array(list(zip(*coords))[0]) + absolute_center[0], np.array(list(zip(*coords))[1]) + absolute_center[1]))
        coords = [(round(i,2), round(j,2)) for (i, j) in coords]
        blnk = np.nanmean(tmp_blank[start_time//frame_time_ext:stop_time//frame_time_ext, :, :], axis = 0) 
        output_matrix_regular = np.array([np.nanmean(i[start_time//frame_time_ext:stop_time//frame_time_ext, :, :], axis = 0)/blnk for i in tmp_data])
        print(f'Shape of blank normalization output matrix: {output_matrix_regular.shape}')
        regular_dict = {(str(a)): b for a,b in zip(coords, output_matrix_regular)}
        data_dict = regular_dict
        print('Blank normalization computed!')

    if (type == 'both'):
        data_dict = {**cocktail_dict, **regular_dict}
        
    return data_dict


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
                        dest='start_time',
                        type=int,
                        default = 3500,
                        required=False,
                        help='Starting time: start stimulus response in cortex') 
    
    parser.add_argument('--operation_maps', 
                        dest='operation_maps',
                        type=str,
                        default = 'both',
                        required=False,
                        help='Operations between maps: choose between cocktail, regular, both') 
    
    parser.add_argument('--dimension_to_analyze', 
                        dest='dimension_to_analyze',
                        type=str,
                        default = 'x',
                        required=False,
                        help='Dimension of operation: choose between x or y')  

    parser.add_argument('--end_time', 
                        dest='end_time',
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
                        dest='frame_time_extension',
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

    parser.add_argument('--vis', 
                        dest='data_vis_switch', 
                        action='store_true')
    parser.add_argument('--no-vis', 
                        dest='data_vis_switch', 
                        action='store_false')
    parser.set_defaults(data_vis_switch=False) 


    start_process_time = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args()

    print(args)

    workspace_rf = RFWorkspace(**vars(args))
    workspace_rf.run_ioi_rf_analysis()
