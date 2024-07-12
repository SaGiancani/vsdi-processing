import numpy as np
import os
import pandas as pd

from mat4py import loadmat
from scipy.optimize import curve_fit

PATH_FL_DOUBLEAM    = r'C:\Users\Neopto\Desktop\Scripts\vsdi_processing\Analysis\AM3_Human\forplot\cross\fl_amup_amdown_dAM_for_plots.png'
PATH_NOFL_DOUBLEAM  = r'C:\Users\Neopto\Desktop\Scripts\vsdi_processing\Analysis\AM3_Human\forplot\nocross\nofl_amup_amdown_dAM_for_plots.png'
PATH_FL_BARAM       = r'C:\Users\Neopto\Desktop\Scripts\vsdi_processing\Analysis\AM3_Human\forplot\cross\fl_amdown_BARS_for_plots.png'
PATH_NOFL_BARAM     = r'C:\Users\Neopto\Desktop\Scripts\vsdi_processing\Analysis\AM3_Human\forplot\nocross\nofl_amdown_BARS_for_plots.png'
PATH_FL_BARAM_AFT   = r'C:\Users\Neopto\Desktop\Scripts\vsdi_processing\Analysis\AM3_Human\forplot\cross\fl_amdown_BARSafter_for_plots.png'
PATH_NOFL_BARAM_AFT = r'C:\Users\Neopto\Desktop\Scripts\vsdi_processing\Analysis\AM3_Human\forplot\nocross\nofl_amdown_BARSafter_for_plots.png'

TITLES_CONDS_BAR      = {3.0: 'static', 6.0: 'AMup', 7.0: 'AMdown'}

TITLES_CONDS_DOUBLEAM = {3.0: 'static-AMup', 
                         4.0: 'static-static', 
                         6.0: 'AMup-AMdown', 
                         7.0: 'AMdown-AMup', 
                         8.0: 'static-AMdown'}

TITLES_CONDS_RED_DOUBLEAM = {4.0: 'static-static', 
                             6.0: 'AMup-AMdown', 
                             7.0: 'AMdown-AMup'}

TITLES_CONDS_MULTISTROKES_DOUBLEAM = {1.0:  'static-static',
                                      2.0:  'AM3up-AM3down_length1',
                                      3.0:  'AM3down-AM3up_length1',
                                      4.0:  'AM3up-AM3down_length2',
                                      5.0:  'AM3down-AM3up_length2',
                                      6.0:  'AM5up-AM5down_length2',
                                      7.0:  'AM5down-AM5up_length2',
                                      8.0:  'AM7up-AM7down_length2',
                                      9.0:  'AM7down-AM7up_length2',
                                      10.0: 'static-static_'}


BAR_PROTOCOL                     =  'AM3strokesBarposition'
DOUBLEAM_PROTOCOL                =  'AM3strokesDoubleAM_'
DOUBLEAM_MULTISPEED_PROTOCOL     =  'AM3strokesDoubleAMMultipleSpeeds'
DOUBLEAM_MULTISTROKES_PROTOCOL   =  'AMMultipleStrokesDoubleAM'

DOUBLE_AM_HEADER = ['Number Run',
                    'Trial number', 
                    'Condition number', #3 for single stroke, 6 for AMup, 7 for AMdown
                    'Eccentricity condition', #1 left, 2 right
                    'Y-Coordinate reference condition', # 1: -0.66, 2: -0.33, 3: 0, 4: 0.33, 5: 0.6
                    '-', # 1: after, 2: always
                    'Trial Start',
                    'Response Time Start',
                    'Y-Coordinate reference stimulus - pixel', # In pixel
                    'Picked Key', # -1 down, 1 up
                    'Reaction time',
                    '_',
                    '_',
                    'Response Time End']

BAR_HEADER       = ['Number Run',
                    'Trial number', 
                    'Condition number', #3 for single stroke, 6 for AMup, 7 for AMdown
                    'Eccentricity condition', #1 left, 2 right
                    'Y-Coordinate Bar - code', # 1: -0.66, 2: -0.33, 3: 0, 4: 0.33, 5: 0.6
                    'Timing Bar Rendering', # 1: after, 2: always
                    'Trial Start',
                    'Response Time Start',
                    'Y-Coordinate Bar - pixel', # In pixel
                    'Spatial Position Responsebar x',
                    'Spatial Position Responsebar y',
                    'Picked Key', # -1 down, 1 up
                    'Reaction time',
                    'Response Time End']

DOUBLE_AM_MULTISPEEDS_HEADER = ['Number Run',
                                'Trial number', 
                                'Condition number', #3 for single stroke, 6 for AMup, 7 for AMdown
                                'Eccentricity condition', #1 left, 2 right
                                'Y-Coordinate reference condition', # 1: -0.66, 2: -0.33, 3: 0, 4: 0.33, 5: 0.6
                                'Trajectory length',
                                'Trial Start',
                                'Response Time Start',
                                'Y-Coordinate reference stimulus - pixel', # In pixel
                                'Picked Key', # -1 down, 1 up
                                'Reaction time',
                                '_',
                                '_',
                                'Response Time End']

DOUBLE_AM_MULTISTROKES_HEADER = DOUBLE_AM_MULTISPEEDS_HEADER


def get_humans_data(path_workspace, 
                    subject, 
                    protocol_name = BAR_PROTOCOL, 
                    flag_eccentricity = False):
    
    if protocol_name == BAR_PROTOCOL:
        header_behav_response = BAR_HEADER
    elif protocol_name == DOUBLEAM_PROTOCOL:
        header_behav_response = DOUBLE_AM_HEADER
    elif protocol_name == DOUBLEAM_MULTISPEED_PROTOCOL:
        header_behav_response = DOUBLE_AM_MULTISPEEDS_HEADER
    elif protocol_name == DOUBLEAM_MULTISTROKES_PROTOCOL:
        header_behav_response = DOUBLE_AM_MULTISTROKES_HEADER

    # Insert subject number and convert it to string
    subject = subject_num2str(subject)

    # Load all the runs for one subject
    result_df = load_all_runs(path_workspace, subject, header_behav_response, protocol_name = protocol_name)

    # Instance variables
    eccentricity_cds   = list(set(result_df['Eccentricity condition']))
    stimulus_cds       = list(set(result_df['Condition number']))

    if protocol_name == BAR_PROTOCOL:
        y_bar_position_cds = list(set(result_df['Y-Coordinate Bar - code']))
        timing_bar_cds     = list(set(result_df['Timing Bar Rendering']))

    elif (protocol_name == DOUBLEAM_PROTOCOL) or (protocol_name == DOUBLEAM_MULTISPEED_PROTOCOL) or (protocol_name == DOUBLEAM_MULTISTROKES_PROTOCOL):
        y_bar_position_cds = list(set(result_df['Y-Coordinate reference condition']))

    # Stimulus ids 
    stimulus_cds       = list(set(result_df['Condition number']))
    # Dictionaries instance
    dict_static_cond_after  = {}
    dict_static_cond_always = {}
    
    # Loop over conditions
    for k in stimulus_cds:
        dict_static_cond_after[k]  = dict()
        dict_static_cond_always[k] = dict()

        for i in y_bar_position_cds:
            if protocol_name == BAR_PROTOCOL:
                if flag_eccentricity:
                    dict_static_cond_always[k][i] = dict()
                    dict_static_cond_after[k][i]  = dict()

                    dict_static_cond_always[k][i]['right'] = list(result_df.loc[(result_df['Condition number']            == k) & # Static
                                                                                (result_df['Eccentricity condition']      == 2) & # Right
                                                                                (result_df['Timing Bar Rendering']        == 2) & # Always present
                                                                                (result_df['Y-Coordinate Bar - code']     == i),
                                                                                'Picked Key']) 
                    dict_static_cond_after[k][i]['right'] = list(result_df.loc[(result_df['Condition number']            == k) & # Static
                                                                               (result_df['Eccentricity condition']      == 2) & # Right
                                                                               (result_df['Timing Bar Rendering']        == 1) & # After present
                                                                               (result_df['Y-Coordinate Bar - code']     == i),
                                                                                'Picked Key']) 
                    
                    dict_static_cond_always[k][i]['left'] = list(result_df.loc[(result_df['Condition number']            == k) & # Static
                                                                               (result_df['Eccentricity condition']      == 1) & # Right
                                                                               (result_df['Timing Bar Rendering']        == 2) & # Always present
                                                                               (result_df['Y-Coordinate Bar - code']     == i),
                                                                                'Picked Key']) 
                    dict_static_cond_after[k][i]['left'] = list(result_df.loc[(result_df['Condition number']            == k) & # Static
                                                                              (result_df['Eccentricity condition']      == 1) & # Right
                                                                              (result_df['Timing Bar Rendering']        == 1) & # After present
                                                                              (result_df['Y-Coordinate Bar - code']     == i),
                                                                               'Picked Key']) 
                else:
                    dict_static_cond_always[k][i] = list(result_df.loc[(result_df['Condition number']        == k) & # Static
                                                                       (result_df['Timing Bar Rendering']    == 2) & # Always present
                                                                       (result_df['Y-Coordinate Bar - code'] == i),
                                                                        'Picked Key']) 

                    dict_static_cond_after[k][i]  = list(result_df.loc[(result_df['Condition number']        == k) & # Static
                                                                       (result_df['Timing Bar Rendering']    == 1) & # After present
                                                                       (result_df['Y-Coordinate Bar - code'] == i),
                                                                        'Picked Key']) 
                    
            elif (protocol_name == DOUBLEAM_PROTOCOL) or (protocol_name == DOUBLEAM_MULTISTROKES_PROTOCOL):

                tmp_1 = list(result_df.loc[(result_df['Condition number']                     == k) & # Static
                                            (result_df['Eccentricity condition']               == 2) & # right
                                            (result_df['Y-Coordinate reference condition']     == i),
                                            'Picked Key']) 
                
                tmp_2 = list((-1)*np.array(result_df.loc[(result_df['Condition number']                     == k) & # Static
                                                            (result_df['Eccentricity condition']               == 1) & # left
                                                            (result_df['Y-Coordinate reference condition']     == i),
                                                            'Picked Key']))
                
                if flag_eccentricity:
                    dict_static_cond_always[k][i] = tmp_2

                    dict_static_cond_after[k][i]  = tmp_1

                else:
                    dict_static_cond_always[k][i] = tmp_1 + tmp_2

                    dict_static_cond_after[k][i]  = None
                

            elif protocol_name == DOUBLEAM_MULTISPEED_PROTOCOL:
                # Build an output dictionary starting from conditions
                tmp = find_all_runs(path_workspace, subject, protocol_name)
                workspace_path     = os.path.join(path_workspace, subject, 'add', tmp[0])

                # Load the MATLAB workspace
                mat_workspace      = loadmat(workspace_path)
                trajectory_lengths = mat_workspace['config']['responseData']['trajectoryLenghts']

                cond_dict = {1: 'static-static'}
                for j,n in enumerate(trajectory_lengths):
                    cond_dict[j*2+2] = f'AMup-AMdown_length{n[0]}'
                    cond_dict[j*2+3] = f'AMdown-AMup_length{n[0]}'
                cond_dict[len(cond_dict.keys())+1] = 'static-static_'

                tmp_1 = list(result_df.loc[(result_df['Condition number']                     == k) & # Static
                                            (result_df['Eccentricity condition']               == 2) & # right
                                            (result_df['Y-Coordinate reference condition']     == i),
                                            'Picked Key']) 
                
                tmp_2 = list((-1)*np.array(result_df.loc[(result_df['Condition number']                     == k) & # Static
                                                            (result_df['Eccentricity condition']               == 1) & # left
                                                            (result_df['Y-Coordinate reference condition']     == i),
                                                            'Picked Key']))
                
                if flag_eccentricity:
                    dict_static_cond_always[k][i] = tmp_2

                    dict_static_cond_after[k][i]  = tmp_1

                else:
                    dict_static_cond_always[k][i] = tmp_1 + tmp_2

                    dict_static_cond_after[k][i]  = None                    

    if protocol_name == DOUBLEAM_MULTISPEED_PROTOCOL:
        return dict_static_cond_after, dict_static_cond_always, subject, result_df, cond_dict
    else:
        return dict_static_cond_after, dict_static_cond_always, subject, result_df


def get_psychometric_curves(preprocessed_dict, flag = 'Bar', combined = True, cond_titles = None):
    if flag == 'Bar':
        titles_conds = TITLES_CONDS_BAR
    elif flag == 'DoubleAM':
        titles_conds = TITLES_CONDS_DOUBLEAM
    elif flag == 'ReductedDoubleAM':
        titles_conds = TITLES_CONDS_RED_DOUBLEAM
    elif flag == 'DoubleAMMultiSpeeds':
        titles_conds = cond_titles
    elif flag == 'DoubleAMMultiStrokes':
        titles_conds = TITLES_CONDS_MULTISTROKES_DOUBLEAM
    
    dict_sigmoids = dict()
    for i, (k,v) in enumerate(titles_conds.items()):
        
        count_val = 1
        # Considering old toy-sessions and new implementation
        if (flag == 'DoubleAMMultiStrokes') and (k == 10):
            try:
                print(f'Data point for condition {v}: {len(preprocessed_dict[k][1.0])}')
                ys_al = [(preprocessed_dict[k][1.0].count(count_val)/(len(preprocessed_dict[k][1.0]))),
                         (preprocessed_dict[k][2.0].count(count_val)/(len(preprocessed_dict[k][2.0]))),
                         (preprocessed_dict[k][3.0].count(count_val)/(len(preprocessed_dict[k][3.0]))),
                         (preprocessed_dict[k][4.0].count(count_val)/(len(preprocessed_dict[k][4.0]))),
                         (preprocessed_dict[k][5.0].count(count_val)/(len(preprocessed_dict[k][5.0])))]
            except:
                pass
        else:
            count_val = 1
            print(f'Data point for condition {v}: {len(preprocessed_dict[k][1.0])}')
            ys_al = [(preprocessed_dict[k][1.0].count(count_val)/(len(preprocessed_dict[k][1.0]))),
                     (preprocessed_dict[k][2.0].count(count_val)/(len(preprocessed_dict[k][2.0]))),
                     (preprocessed_dict[k][3.0].count(count_val)/(len(preprocessed_dict[k][3.0]))),
                     (preprocessed_dict[k][4.0].count(count_val)/(len(preprocessed_dict[k][4.0]))),
                     (preprocessed_dict[k][5.0].count(count_val)/(len(preprocessed_dict[k][5.0])))]
        dict_sigmoids[v] = ys_al
    if combined:
        if flag == 'Bar':
            dict_sigmoids['AM'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AMup']), 
                                                            np.array(1-np.array(dict_sigmoids['AMdown']))[::-1]], axis=0), axis=0, dtype = float))
        elif (flag == 'DoubleAM'):
            dict_sigmoids['AM'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AMup-AMdown']), 
                                                            np.array(1-np.array(dict_sigmoids['AMdown-AMup']))[::-1]], axis=0), axis=0, dtype = float))
            dict_sigmoids['static-AM'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['static-AMdown']), 
                                                                   np.array(1-np.array(dict_sigmoids['static-AMup']))[::-1]], axis=0), axis=0, dtype = float))
        elif (flag == 'ReductedDoubleAM'):
            dict_sigmoids['AM'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AMup-AMdown']), 
                                                            np.array(1-np.array(dict_sigmoids['AMdown-AMup']))[::-1]], axis=0), axis=0, dtype = float))
        elif (flag == 'DoubleAMMultiSpeeds'):
            amdown = list(cond_titles.values())[2:-1:2]
            amup   = list(cond_titles.values())[1:-1:2]

            dict_sigmoids[f'static'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['static-static']), 
                                                                 np.array(dict_sigmoids['static-static_'])], axis=0), axis=0, dtype = float))
            for u, d in zip(amup, amdown):
                traj = float(u.split('_length')[1])
                dict_sigmoids[f'AM_traj{traj}'] = list(np.nanmean(np.stack([np.array(dict_sigmoids[u]), 
                                                                   np.array(1-np.array(dict_sigmoids[d]))[::-1]], axis=0), axis=0, dtype = float))
        elif (flag == 'DoubleAMMultiStrokes'):
            dict_sigmoids[f'static']     = list(np.nanmean(np.stack([np.array(dict_sigmoids['static-static']), 
                                                                     np.array(dict_sigmoids['static-static_'])], axis=0), axis=0, dtype = float))
            dict_sigmoids['AM3_length1'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AM3up-AM3down_length1']), 
                                                                     np.array(1-np.array(dict_sigmoids['AM3down-AM3up_length1']))[::-1]], axis=0), axis=0, dtype = float))
            dict_sigmoids['AM3_length2'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AM3up-AM3down_length2']), 
                                                                     np.array(1-np.array(dict_sigmoids['AM3down-AM3up_length2']))[::-1]], axis=0), axis=0, dtype = float))
            dict_sigmoids['AM5_length2'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AM5up-AM5down_length2']), 
                                                                     np.array(1-np.array(dict_sigmoids['AM5down-AM5up_length2']))[::-1]], axis=0), axis=0, dtype = float))
            dict_sigmoids['AM7_length2'] = list(np.nanmean(np.stack([np.array(dict_sigmoids['AM7up-AM7down_length2']), 
                                                                     np.array(1-np.array(dict_sigmoids['AM7down-AM7up_length2']))[::-1]], axis=0), axis=0, dtype = float))

    return dict_sigmoids


def sigmoid(x, a, b, c):
    # Define the sigmoid function
    # return np.exp(-b * (x - c)) / (1 + np.exp(-b * (x - c)))
    return 1 / (1 + np.exp(-b * (x - c)))


def fit_sigmoid(ys, vertical_stretch = .98, horizontal_shift = 1, vertical_shift = 1, data_points_x = np.linspace(0, 4, 5), extrapolation = [0, 4]):
    try:
        print('Regular fitting')
        # Fit the sigmoid curve to the data
        params, covariance = curve_fit(sigmoid,data_points_x, ys, p0=[vertical_stretch, horizontal_shift, vertical_shift], method='trf', maxfev = 10000)

    except:
        print('Datapoints added')
        min_el_y = list([.01])
        max_el_y = list([.98])
        min_el_x = np.nanmin(data_points_x) -4
        max_el_x = np.nanmax(data_points_x) +4

        params, covariance = curve_fit(sigmoid, 
                                        list([min_el_x-1]) + list([min_el_x]) + list(data_points_x) + list([max_el_x]) + list([max_el_x+1]), 
                                        min_el_y + min_el_y + list(ys) + max_el_y + max_el_y, 
                                        p0=[vertical_stretch, horizontal_shift, vertical_shift], method='trf', maxfev = 10000)
    # Extract the parameters
    a, b, c = params
    
    # Generate the fitted curve
    x_fit = np.linspace(extrapolation[0], extrapolation[1], 10000)
    y_fit = sigmoid(x_fit, a, b, c)
    slope = b/4

    # Forcing to converge to a sigmoid shape if the fitting grows exponentially
    if np.max(y_fit) > 1.2:
        min_el_y = list([.01])
        max_el_y = list([.98])
        min_el_x = np.nanmin(data_points_x) -1
        max_el_x = np.nanmax(data_points_x) +1

        params, covariance = curve_fit(sigmoid, 
                                        list([min_el_x-1]) + list([min_el_x]) + list(data_points_x) + list([max_el_x]) + list([max_el_x+1]), 
                                        min_el_y + min_el_y + list(ys) + max_el_y + max_el_y, 
                                        p0=[vertical_stretch, horizontal_shift, vertical_shift], method='trf', maxfev = 10000)      
        # Extract the parameters
        a, b, c = params
        
        # Generate the fitted curve
        x_fit = np.linspace(extrapolation[0], extrapolation[1], 10000)
        y_fit = sigmoid(x_fit, a, b, c)
        slope = b/4
 
    return x_fit, y_fit, slope

def find_all_runs(path_workspace, subject,protocol_name):
    tmp = [f.name for f in os.scandir(os.path.join(path_workspace, subject, 'add')) 
           if (f.is_file()) and (f.name.endswith(".mat"))and (protocol_name in f.name)]
    return tmp

def subject_num2str(subject):
    sub_number = str(subject)
    if len(sub_number) <2:
        sub_number = '0'+sub_number
    subject    = f'sub-{sub_number}'
    return subject

def load_all_runs(path_workspace, subject, header_behav_response, protocol_name = BAR_PROTOCOL):
    # Load all the matrices name
    tmp = find_all_runs(path_workspace, subject,protocol_name)
    print(tmp)
    all_runs = list()
    # Loop over runs

    for run_number in tmp: 
        # Run variable path
        workspace_path = os.path.join(path_workspace, subject, 'add', run_number)
        # Load the MATLAB workspace
        mat_workspace = loadmat(workspace_path)
        df = pd.DataFrame(mat_workspace['config']['responseData']['expRespMat'], columns=header_behav_response)
        all_runs.append(df)    
    
    # Concatenate runs
    result_df = pd.concat(all_runs, ignore_index=True)
    if protocol_name == DOUBLEAM_PROTOCOL:
        result_df = result_df[['Number Run',
                               'Trial number', 
                               'Condition number', #3 for single stroke, 6 for AMup, 7 for AMdown
                               'Eccentricity condition', #1 left, 2 right
                               'Y-Coordinate reference condition', # 1: -0.66, 2: -0.33, 3: 0, 4: 0.33, 5: 0.6
                               'Trial Start',
                               'Response Time Start',
                               'Y-Coordinate reference stimulus - pixel', # In pixel
                               'Picked Key', # -1 down, 1 up
                               'Reaction time',
                               'Response Time End']] 
        
    elif protocol_name == BAR_PROTOCOL:
        result_df = result_df[['Number Run',
                               'Trial number', 
                               'Condition number', #3 for single stroke, 6 for AMup, 7 for AMdown
                               'Eccentricity condition', #1 left, 2 right
                               'Y-Coordinate Bar - code', # 1: -0.66, 2: -0.33, 3: 0, 4: 0.33, 5: 0.6
                               'Timing Bar Rendering', # 1: after, 2: always
                               'Trial Start',
                               'Response Time Start',
                               'Y-Coordinate Bar - pixel', # In pixel
                               'Spatial Position Responsebar x',
                               'Spatial Position Responsebar y',
                               'Picked Key', # -1 down, 1 up
    #                          'Spatial Position Responsebar x',
                               'Reaction time',
                               'Response Time End']] 
        
    return result_df

import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import matplotlib.patches as patches

def plot_merged(path_mat_workspace, sub_list, extrapolation = [0, 4], protocol = 'Bar', 
                sub_protocol = None, x_range = [0, 4], y_range = [-.66, .66], file_name = None):
    if len(sub_list) == 1:
        proportion = .85
        x_prop = 10
    else:
        proportion = 1
        x_prop = 12
        
    fig, ax = plt.subplots(2, len(sub_list),figsize=(proportion*11*len(sub_list), 2*x_prop),
                           gridspec_kw={'wspace': .2, 'hspace': .5}, 
                           dpi = 70)
    
    fig.set_facecolor('white')
    
    # Storing variables
    pse_dict          = dict()
    if len(sub_list) == 1:
        ax = np.array(np.array(ax)).reshape(2, 1)
    
    for n, i in enumerate(sub_list):
        sub = i
        pse_dict[f'sub{i}']   = dict()

        if protocol == 'Bar':
            out = get_humans_data(path_mat_workspace, i, 
                                  protocol_name = BAR_PROTOCOL)
            if (sub_protocol is None) or (sub_protocol == 'always'):
                input_data = out[1]
                colors_combined       = ['crimson', 'orange']
                img_path_fl_mot       =  PATH_FL_BARAM
                img_path_nofl_mot     =  PATH_NOFL_BARAM
                title = 'Bar presented since stimulus onset'
            if (sub_protocol == 'after'):
                input_data = out[0]
                colors_combined       = ['purple', 'pink']
                img_path_fl_mot       =  PATH_FL_BARAM_AFT
                img_path_nofl_mot     =  PATH_NOFL_BARAM_AFT
                title = 'Bar presented after stimulus offset'
            
            tmp  =  get_psychometric_curves(input_data, combined = True, flag = protocol)
            data_points_am        = len(input_data[6][1]) + len(input_data[7][1])
            data_points_stat      = len(input_data[3][1])

        elif (protocol == 'DoubleAM') or (protocol == 'ReductedDoubleAM'):
            ecc = False
            out =  get_humans_data(path_mat_workspace, i, 
                                       protocol_name =  DOUBLEAM_PROTOCOL, 
                                       flag_eccentricity= ecc)             
            input_data = out[1]
            tmp  =  get_psychometric_curves(input_data, combined = True, flag = protocol)
            asda = dict()
            img_path_fl_mot       =  PATH_FL_DOUBLEAM
            img_path_nofl_mot     =  PATH_NOFL_DOUBLEAM
            asda['AMup']   = tmp['AMup-AMdown']
            asda['AMdown'] = tmp['AMdown-AMup']
            asda['static'] = tmp['static-static']
            asda['AM']     = tmp['AM']
            tmp            = asda   
            colors_combined       = ['darkgreen', 'yellowgreen']
            data_points_am        = len(input_data[6][1]) + len(input_data[7][1])
            data_points_stat      = len(input_data[4][1])
            title = 'DoubleAM protocol'

        elif (protocol == 'DoubleAMMultiSpeeds'):
            ecc = False
            out =  get_humans_data(path_mat_workspace, i, 
                                  protocol_name =  DOUBLEAM_MULTISPEED_PROTOCOL, 
                                  flag_eccentricity= ecc)             
            input_data = out[1]
            conds      = out[-1]
            tmp        =  get_psychometric_curves(input_data, combined = True, flag = protocol, cond_titles = conds)
            asda       = dict()
            img_path_fl_mot   =  PATH_FL_DOUBLEAM
            img_path_nofl_mot =  PATH_NOFL_DOUBLEAM         
            colors_combined   = ['black', 'dimgray', 'silver', 'lightgray']
            data_points_am        = len(input_data[6][1]) + len(input_data[7][1])
            data_points_stat      = len(input_data[8][1]) + len(input_data[1][1])
            title = 'DoubleAM MultiSpeeds protocol'

        elif (protocol == 'DoubleAMMultiStrokes'):
            ecc = False
            out =  get_humans_data(path_mat_workspace, i, 
                                  protocol_name =  DOUBLEAM_MULTISTROKES_PROTOCOL, 
                                  flag_eccentricity= ecc)             
            input_data = out[1]
            tmp  =  get_psychometric_curves(input_data, combined = True, flag = protocol)
            asda = dict()
            img_path_fl_mot       =  PATH_FL_DOUBLEAM
            img_path_nofl_mot     =  PATH_NOFL_DOUBLEAM

            print(input_data.keys())

            if 10 not in input_data.keys():
                asda['static']     = tmp['static-static']
                data_points_stat   = len(input_data[1][1])
            else:
                asda['static']     = tmp['static']
                data_points_stat   = len(input_data[1][1])+ len(input_data[10][1])
                print(f'data_points_stat: {data_points_stat}')
               
            asda['AM3_t1']     = tmp['AM3_length1']
            asda['AM3_t2']     = tmp['AM3_length2']
            asda['AM5_t2']        = tmp['AM5_length2']
            asda['AM7_t2']        = tmp['AM7_length2']
            tmp                = asda   
            colors_combined    = ['midnightblue', 'royalblue',
                                  'dodgerblue', 'deepskyblue', 
                                  'lightblue']
            data_points_am     = len(input_data[6][1]) + len(input_data[7][1])
            title              = 'DoubleAM Multiple Strokes protocol'


        img_fl_mot = plt.imread(img_path_fl_mot)
        img_nofl_mot   = plt.imread(img_path_nofl_mot)
        
        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            down_am                          =  list(1-np.array(tmp['AMdown']))[::-1]
            x_up, y_up, slope_up             =  fit_sigmoid(tmp['AMup'], extrapolation = extrapolation)#,vertical_stretch = .98, horizontal_shift = 1, vertical_shift = 1)
            x_down, y_down, slope_down       =  fit_sigmoid(down_am, extrapolation = extrapolation)#,vertical_stretch = .98, horizontal_shift = 1, vertical_shift = 1)
            x_am, y_am, slope_am             =  fit_sigmoid(tmp['AM'], extrapolation = extrapolation)#,vertical_stretch = .98, horizontal_shift = 1, vertical_shift = 1)
            pse_am                           =  np.interp(.5, y_am, x_am)
            pse_dict[f'sub{i}']['slope AM']  =  slope_am

        x_static, y_static, slope_static     =  fit_sigmoid(tmp['static'], extrapolation = extrapolation)#,vertical_stretch = .98, horizontal_shift = 1, vertical_shift = 1)
        pse_dict[f'sub{i}']['slope static']  =  slope_static

        pse_stat = np.interp(.5, y_static, x_static)

        img_size = 0.035

        img_position_down_x = (.9, .15)
        img_position_up_x   = (.1, .15)

        # ---------------------------------------------------------
        # Figure 3 combined AM
        # ---------------------------------------------------------

        # Plotting fit
        ax[0][n].plot(x_static, y_static,colors_combined[0], label = 'static control')

        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            ax[0][n].plot(x_am, y_am, colors_combined[1], label = 'AM')
        else:
            dict_fit_sigm = dict()
            counter = 1
            cond_colors = dict()
            cond_colors['static'] = colors_combined[0]
            for k in tmp.keys():
                if ('traj' in k) or ('_t' in k): 
                    cond_colors[k] = colors_combined[counter]
                    dict_fit_sigm[k] =  fit_sigmoid(tmp[k], extrapolation = extrapolation)          
                    ax[0][n].plot(dict_fit_sigm[k][0], dict_fit_sigm[k][1], cond_colors[k], label = k)
                    pse_dict[f'sub{i}'][f'slope {k}']  =  dict_fit_sigm[k][2]

                    counter += 1

        # Linking dots
        ax[0][n].plot(np.linspace(0, 4, len(tmp['static'])), 
                      tmp['static'], colors_combined[0], alpha = .5, lw =.5)
        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            ax[0][n].plot(np.linspace(0, 4, len(tmp['AM'])), 
                          tmp['AM'], colors_combined[1], alpha = .5, lw =.5)
        else:
            counter = 1
            tmp_x = np.linspace(0, 4, len(tmp['static']))
            for k in tmp.keys():
                if ('traj' in k) or ('_t' in k): 
                    ax[0][n].plot(tmp_x, tmp[k], cond_colors[k], alpha = .5, lw =.5)
                    counter += 1


        print(f'Data points static: {data_points_stat}, Data points statAM {data_points_am}')
        # Plotting dots
        ax[0][n].scatter(np.linspace(0, 4, len(tmp['static'])), 
                         tmp['static'], color = colors_combined[0], s=data_points_stat*3, edgecolors='black')
        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            ax[0][n].scatter(np.linspace(0, 4, len(tmp['AM'])), 
                             tmp['AM'],  color = colors_combined[1], s=data_points_am*3, edgecolors='black')
        else:
            counter = 1
            for k in tmp.keys():
                if ('traj' in k) or ('_t' in k): 
                    ax[0][n].scatter(tmp_x, tmp[k], color = cond_colors[k], s = data_points_am*3, edgecolors='black')
                    counter += 1

        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            ax[0][n].vlines(pse_am, -.1, .5, color = colors_combined[1], ls ='-', lw=2, alpha = .5)
            ax[0][n].hlines(.5, x_range[0], np.nanmax([pse_am, pse_stat]), color = 'k', ls ='-', lw=.5, alpha = 1)

        else:
            counter = 1
            pse_tmps = dict()
            pse_colors = dict()
            for k in tmp.keys():
                if ('traj' in k) or ('_t' in k): 
                    pse_tmp     = np.interp(.5, dict_fit_sigm[k][1], dict_fit_sigm[k][0])
                    ax[0][n].vlines(pse_tmp, -.1, .5, color = cond_colors[k], ls ='-', lw=2, alpha = .5)
                    pse_tmps[k]  = pse_tmp
                    pse_colors[pse_tmp] = colors_combined[counter]
                    counter +=1 
            pse_tmps['static']  = pse_stat
            pse_colors[pse_stat] = colors_combined[0]
            ax[0][n].hlines(.5, x_range[0], np.nanmax([np.nanmax(list(pse_tmps.values())), pse_stat]), color = 'k', ls ='-', lw=.5, alpha = 1)

        ax[0][n].vlines(pse_stat, -.1, .5, color = colors_combined[0], ls ='-', lw=2, alpha = .5)
        ax[0][n].set_xlim(x_range[0], x_range[1]+.3)

        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            # Set the original y-axis ticks
            x_toplot = list(np.linspace(x_range[0], x_range[1], 5)) + [pse_stat, pse_am]
            x_toplot.sort()

            index_am   = x_toplot.index(pse_am)
            index_stat = x_toplot.index(pse_stat)
            colors_ticks = ['k'] * len(x_toplot)
            colors_ticks[index_am] = colors_combined[1]
            colors_ticks[index_stat] = colors_combined[0]
        else:
            pses = [i for i in pse_colors.keys()]
            pses.sort()
            x_toplot = list(np.linspace(x_range[0], x_range[1], 5)) + pses
            x_toplot.sort()
            colors_ticks = ['k']*len(x_toplot)
            for num, ps in enumerate(x_toplot):
                if ps in pses:
                    colors_ticks[num] = pse_colors[ps]
            
        # Calculate the slope of the line defined by the endpoints
        slope = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])

        # Extrapolate the corresponding values
        x_ticks = y_range[0] + slope * (np.array(x_toplot) - x_range[0])
        
        pse_dict[f'sub{i}']['stat']   = y_range[0] + slope * (np.array(pse_stat) - x_range[0])

        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            pse_dict[f'sub{i}']['AMup']   = y_range[0] + slope * (np.interp(.5, y_up, x_up) - x_range[0]) 
            pse_dict[f'sub{i}']['AMdown'] = y_range[0] + slope * (np.interp(.5, y_down, x_down) - x_range[0])  
            pse_dict[f'sub{i}']['AM']     = y_range[0] + slope * (np.array(pse_am) - x_range[0])   
        else:
            for k in pse_tmps.keys():
                pse_dict[f'sub{i}'][k]     = y_range[0] + slope * (np.array(pse_tmps[k]) - x_range[0])   
        
        ax[0][n].set_xticks(x_toplot)  # Include PSE in y-ticks

        labels_ = [f'{value:.2f}' for value in x_ticks]
        ax[0][n].set_xticklabels(labels_, fontsize=25, rotation = 45)
        ax[0][n].set_xlabel('Space - dva', fontsize=30)

        # Loop through the ytick labels and set their colors individually
        for label, color in zip(ax[0][n].get_xticklabels(), colors_ticks):
            label.set_color(color)

        # Plot the original data and the fitted curve
        ax[0][n].set_yticks([0, .25, .5, .75, 1])
        labels_y = [item.get_text() for item in ax[0][n].get_yticklabels()]
        labels_y[4] = '100%'
        labels_y[3] = '75%'
        labels_y[2] = '50%'
        labels_y[1] = '25%'
        labels_y[0] = '0%'
        ax[0][n].set_yticklabels(labels_y, fontsize = 25)
        ax[0][n].set_ylabel('% Motion extrapolated response', fontsize = 30)

        ax[0][n].set_ylim(-.1, 1.1)

        ax[0][n].set_title(f'{title}', fontsize = 30, color = 'k')

        ax[0][n].legend(loc = 'lower right', fontsize = 20)
    
        ax[0][n].text(-1, .75, f'sub{sub}', fontsize=30, color='k')#, rotation = 90)
        ax[0][n].spines['top'].set_color('none')
        ax[0][n].spines['right'].set_color('none')

        # ---------------------------------------------------------
        # Figure 4 combined AM
        # ---------------------------------------------------------

        # Plotting fit

        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            asymmetric_error = [[abs(np.interp(.25, y_static, x_static)-pse_stat)], [abs(np.interp(.75, y_static, x_static)-pse_stat)]]
            ax[1][n].errorbar(pse_stat, 1.5, xerr=asymmetric_error, fmt='o', 
                              markersize = data_points_stat//4, color = colors_combined[0], 
                              markeredgecolor = 'k', label = 'static' )

            asymmetric_error = [[abs(np.interp(.25, y_am, x_am)-pse_am)], [abs(np.interp(.75, y_am, x_am)-pse_am)]]
            ax[1][n].errorbar(pse_am, 3, xerr=asymmetric_error, fmt='o', 
                              markersize = data_points_am//4, color = colors_combined[1], 
                              markeredgecolor = 'k', label = 'NormAM' )
            
            pse_dict[f'sub{i}']['AM higherror']      = abs((y_range[0] + slope * (abs(np.interp(.75, y_am, x_am) - x_range[0]))) - pse_dict[f'sub{i}']['AM'])  
            pse_dict[f'sub{i}']['AM lowerror']       = abs((y_range[0] + slope * (abs(np.interp(.25, y_am, x_am) - x_range[0]))) - pse_dict[f'sub{i}']['AM'])    

            pse_dict[f'sub{i}']['AM datapoints']     = data_points_am

        else:
            print(cond_colors)
            x_pos = np.linspace(0.5, len(cond_colors.keys())+1, len(cond_colors.keys()))
            asymmetric_error = [[abs(np.interp(.25, y_static, x_static)-pse_stat)], [abs(np.interp(.75, y_static, x_static)-pse_stat)]]
            ax[1][n].errorbar(pse_stat, x_pos[0], xerr=asymmetric_error, fmt='o', 
                              markersize = data_points_stat//4, color = cond_colors['static'], 
                              markeredgecolor = 'k', label = 'static' )

            list_conds = ['static']
            for col, (k, v) in enumerate(dict_fit_sigm.items()):
                list_conds.append(k)
                asymmetric_error = [[abs(np.interp(.25, v[1], v[0]) -pse_tmps[k] )], 
                                    [abs(np.interp(.75, v[1], v[0]) -pse_tmps[k] )]]
                ax[1][n].errorbar(pse_tmps[k], x_pos[col+1], xerr=asymmetric_error, fmt='o', 
                                  markersize = data_points_am//2, color = cond_colors[k], 
                                  markeredgecolor = 'k', label = 'NormAM' )
                print((y_range[0] + slope * (abs(np.interp(.75, v[1], v[0]) - x_range[0]))), 
                     (y_range[0] + slope * (abs(np.interp(.25, v[1], v[0]) - x_range[0]))))
                print(pse_dict[f'sub{i}'][k] )
                pse_dict[f'sub{i}'][f'{k} higherror']      = abs((y_range[0] + slope * (abs(np.interp(.75, v[1], v[0]) - x_range[0]))) 
                                                                 - pse_dict[f'sub{i}'][k])  
                pse_dict[f'sub{i}'][f'{k} lowerror']       = abs((y_range[0] + slope * (abs(np.interp(.25, v[1], v[0]) - x_range[0]))) 
                                                                 - pse_dict[f'sub{i}'][k]) 
                print(pse_dict[f'sub{i}'][f'{k} higherror'], 
                      pse_dict[f'sub{i}'][f'{k} lowerror'] )
                pse_dict[f'sub{i}'][f'{k} datapoints']     = data_points_am

        pse_dict[f'sub{i}']['stat higherror']    = abs((y_range[0] + slope * (abs(np.interp(.75, y_static, x_static) - x_range[0]))) - pse_dict[f'sub{i}']['stat'])  
        pse_dict[f'sub{i}']['stat lowerror']     = abs((y_range[0] + slope * (abs(np.interp(.25, y_static, x_static) - x_range[0]))) - pse_dict[f'sub{i}']['stat'])
        pse_dict[f'sub{i}']['stat datapoints']   = data_points_stat
        
        
        ax[1][n].set_ylim(-2,6)

        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            ax[1][n].set_ylim(0, 4)
            label_cond = ['Static', 'AM']
            ax[1][n].set_yticks([1.5, 3])
            ax[1][n].set_yticklabels(label_cond, fontsize = 25)


        else:
            print(x_pos)
            ax[1][n].set_ylim(-1, x_pos[-1]+1)
            label_cond = list_conds
            print(label_cond)
            ax[1][n].set_yticklabels(label_cond, fontsize = 25, rotation = 30)
            ax[1][n].set_yticks(x_pos)
            
        ax[1][n].set_ylabel('Conditions', fontsize = 30)
        
        if (protocol != 'DoubleAMMultiSpeeds') and (protocol != 'DoubleAMMultiStrokes'):
            ax[1][n].vlines(pse_stat, 0, 1.5, ls = '--', lw=.8, color = colors_combined[0])
            ax[1][n].vlines(pse_am, 0, 3, ls = '--', lw=.8, color = colors_combined[1])
            x_toplot = list(np.linspace(-2, 6, 9)) + [pse_stat, pse_am]
            # Set the original y-axis ticks
            x_toplot.sort()
            index_am   = x_toplot.index(pse_am)
            index_stat = x_toplot.index(pse_stat)
            colors_ticks = ['k'] * len(x_toplot)
            colors_ticks[index_am] = colors_combined[1]
            colors_ticks[index_stat] = colors_combined[0]
            arrow_start = (-.2, 1.5)
            arrow_end   = (-1.6, 1.5)
            text_coord  = (-1.8, 1.7)

        else:
            ax[1][n].vlines(pse_stat, -1, x_pos[0], ls = '--', lw=.8, color = colors_combined[0])
            for num, k in enumerate(list_conds[1:]):
                ax[1][n].vlines(pse_tmps[k], -1, x_pos[num+1], ls = '--', lw=.8, color = cond_colors[k])
            x_toplot = list(np.linspace(-2, 6, 9)) + list(pse_tmps.values())            
            pses = [i for i in pse_colors.keys()]
            pses.sort()
            x_toplot.sort()
            colors_ticks = ['k']*len(x_toplot)
            for num, ps in enumerate(x_toplot):
                if ps in pses:
                    colors_ticks[num] = pse_colors[ps] 
            arrow_start = (-.2, 1.5)
            arrow_end   = (-1.6, 1.5)
            text_coord  = (-1.8, 1.7)

        # Calculate the slope of the line defined by the endpoints
        slope = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])

        # Extrapolate the corresponding values
        dva_ticks = y_range[0] + slope * (np.array(x_toplot) - x_range[0])

        ax[1][n].set_xticks(x_toplot)  # Include PSE in y-ticks

        labels_ = [f'{value:.2f}' for value in dva_ticks]
        ax[1][n].set_xticklabels(labels_, fontsize=25, rotation = 45)
        ax[1][n].set_xlabel('Space - dva', fontsize=30)

        # Loop through the ytick labels and set their colors individually
        for label, color in zip(ax[1][n].get_xticklabels(), colors_ticks):
            label.set_color(color)    

#         ax[n][1].legend(loc = 'lower right', fontsize = 15) 
        ax[1][n].set_title(f'PSE & intercepts at 25%-75%', fontsize = 30, color = 'k')

        arrow = patches.FancyArrowPatch(
            arrow_start,  # Starting point (x, y)
            arrow_end,  # Ending point (x, y)
            arrowstyle='->',  # Arrow style
            mutation_scale=30,  # Adjust the arrow size
            lw = 5,
            color=colors_combined[1]  # Arrow color
        )
        ax[1][n].add_patch(arrow)  # Add the arrow to the current axes
        ax[1][n].text(text_coord[0], text_coord[1], "Motion\nextrapolation", fontsize=20, color=colors_combined[1])#, rotation = 90)
    
    
        # Set the desired size for the image annotation
        # DOWN
        # Create an OffsetImage for annotation
        img_box = offsetbox.OffsetImage(img_nofl_mot, zoom=img_size*.9, resample=True)
        ab = offsetbox.AnnotationBbox(img_box, img_position_down_x, frameon=False, 
                                      xycoords='axes fraction', 
                                      boxcoords="axes fraction", 
                                      pad=0, zorder=1)
        
        # Add the annotation to the plot
        ax[1][n].add_artist(ab)
    
        ax[1][n].spines['top'].set_color('none')
        ax[1][n].spines['right'].set_color('none')


        # Create an OffsetImage for annotation
        img_box = offsetbox.OffsetImage(img_fl_mot, zoom=img_size*.9, resample=True)
        ab = offsetbox.AnnotationBbox(img_box, img_position_up_x, frameon=False, 
                                      xycoords='axes fraction', 
                                      boxcoords="axes fraction", 
                                      pad=0, zorder=1)
        
        # Add the annotation to the plot
        ax[1][n].add_artist(ab)
        if file_name is not None:
            plt.savefig(os.path.join(file_name + '.pdf'), format = 'pdf', dpi =500)
            plt.savefig(os.path.join(file_name + '.png'), format = 'png', dpi =500)
    return pse_dict

