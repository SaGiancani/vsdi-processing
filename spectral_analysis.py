import argparse, blk_file, datetime, os, utils
import cv2 as cv
import data_visualization as dv
import middle_process as md
import matplotlib.pyplot as plt
import numpy as np
import process_vsdi as process
import retinotopy as retino
from scipy.signal import butter, hilbert, filtfilt#, hilbert2, lfilter
from scipy.stats import circmean


def analytic_signal(x):
    sh = x.shape
    assert len(sh) == 3, 'Datacube required'
    xo = np.empty(sh)
    xo[:] = np.nan
    xo_im = np.empty(sh)
    xo_im[:] = np.nan
    amplitude_envelope  = np.empty(sh)
    amplitude_envelope[:] = np.nan
    for i in range(sh[-2]):
        for j in range(sh[-1]):
            sign_trans = hilbert(x[:, i, j])
            xo[:, i, j] = sign_trans.real
            #print(sign_trans.real.shape, sign_trans.imag.shape, np.abs(sign_trans).shape)
            xo_im[:, i, j] = sign_trans.imag
            amplitude_envelope[:, i, j] = np.abs(sign_trans)
    return xo, xo_im, amplitude_envelope

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# bandpass_filter(x, f1, f2, filter_order, Fs)
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)
    return y

def bandpass_filter(data, f1, f2, filter_order, fs):
    sh = data.shape
    assert len(sh) == 3, 'Datacube required'
    xo = np.empty(sh)  
    xo[:] = np.nan  
    for i in range(sh[-2]):
        for j in range(sh[-1]):
            xo[:, i, j] = butter_bandpass_filter(data[:, i, j], f1, f2, fs, filter_order)
    return xo

def instantaneous_frequency(xph, fs):
    sh = xph.shape
    thresh_tol = 0.9
    #ft = np.zeros(np.shape(xph))
    ft = np.empty(sh)  
    ft[:] = np.nan  
    #ft(:,:,1:end-1) = angle( xph(:,:,2:end) .* conj( xph(:,:,1:end-1) ) ) ./ (2*pi/Fs);
    ft[0:-2, :,:] = np.angle( np.multiply(xph[1:-1, :,:],np.conj(xph[0:-2,:,:] )) ) / (2*np.pi/fs)
    
    if (sum(ft[np.where(ft!=0)] > 0 ) / len( ft[np.where(ft!=0)] ) ) > thresh_tol:
        signIF = +1
    elif ( sum( ft[np.where(ft!=0)] < 0 ) / len( ft[np.where(ft!=0)] ) ) > thresh_tol:
        signIF = -1 
    else:
        signIF = np.nan
    return ft, signIF

def phase_gradient_complex_multiplication(xph, pixel_spacing, signIF = 1):
    assert len(xph) == 3, 'Datacube required'
    dim_xph = np.shape(xph)
    pm = np.zeros(dim_xph)
    pd = np.zeros(dim_xph)
    dx = np.zeros(dim_xph)
    dy = np.zeros(dim_xph)
    
    for tt, time_instant in enumerate(xph):
        # dx
        tmp_dx = np.zeros(np.shape(time_instant))
        # Differences on left and right edges
        tmp_dx[:, 0] = np.angle( np.multiply(time_instant[:, 1], np.conj(time_instant[:, 0])) ) / pixel_spacing
        tmp_dx[:, -1] = np.angle( np.multiply(time_instant[ :, -1], np.conj(time_instant[:, -2])) ) / pixel_spacing
        # Differences interior points    
        tmp_dx[:, 1:-2] = np.angle( np.multiply(time_instant[:, 2:-1], np.conj(time_instant[:, 0:-3])) ) / (2*pixel_spacing)
        dx[tt,:,:] = -signIF * tmp_dx
        
        # dy
        tmp_dy = np.zeros(np.shape(time_instant))  
        # Differences on top and bottom edges
        tmp_dy[0, :] = np.angle( np.multiply(time_instant[1, :], np.conj(time_instant[0, :])) ) / pixel_spacing
        tmp_dy[-1, :] = np.angle( np.multiply(time_instant[-1, :], np.conj(time_instant[-2, :])) ) / pixel_spacing         
        # Differences interior points    
        tmp_dy[1:-2, :] = np.angle( np.multiply(time_instant[2:-1, :], np.conj(time_instant[0:-3, :])) ) / (2*pixel_spacing)
        dy[tt,:,:] = -signIF * tmp_dy

    pm = np.sqrt(dx**2 + dy**2)/(2*np.pi)
    pd = np.arctan2(dy, dx) 
    return pm,pd,dx,dy

def recombine_analytical_signal(xo, xo_im):
    return xo + 1j * xo_im

def phase_gradient_complex_multiplication(xph, pixel_spacing, signIF = 1):
    dim_xph = np.shape(xph)
    pm = np.zeros(dim_xph)
    pd = np.zeros(dim_xph)
    dx = np.zeros(dim_xph)
    dy = np.zeros(dim_xph)
    
    for tt, time_instant in enumerate(xph):
        # dx
        tmp_dx = np.zeros(np.shape(time_instant))
        # Differences on left and right edges
        tmp_dx[:, 0] = np.angle( np.multiply(time_instant[:, 1], np.conj(time_instant[:, 0])) ) / pixel_spacing
        tmp_dx[:, -1] = np.angle( np.multiply(time_instant[ :, -1], np.conj(time_instant[:, -2])) ) / pixel_spacing
        # Differences interior points    
        tmp_dx[:, 1:-2] = np.angle( np.multiply(time_instant[:, 2:-1], np.conj(time_instant[:, 0:-3])) ) / (2*pixel_spacing)
        dx[tt,:,:] = -signIF * tmp_dx
        
        # dy
        tmp_dy = np.zeros(np.shape(time_instant))  
        # Differences on top and bottom edges
        tmp_dy[0, :] = np.angle( np.multiply(time_instant[1, :], np.conj(time_instant[0, :])) ) / pixel_spacing
        tmp_dy[-1, :] = np.angle( np.multiply(time_instant[-1, :], np.conj(time_instant[-2, :])) ) / pixel_spacing         
        # Differences interior points    
        tmp_dy[1:-2, :] = np.angle( np.multiply(time_instant[2:-1, :], np.conj(time_instant[0:-3, :])) ) / (2*pixel_spacing)
        dy[tt,:,:] = -signIF * tmp_dy

    pm = np.sqrt(dx**2 + dy**2)/(2*np.pi)
    pd = np.arctan2(dy, dx) 
    return pm,pd,dx,dy

def extract_analytic_signal(yo):
    # Analytic Signal extraction
    start_analytic_signal = datetime.datetime.now().replace(microsecond=0)
    sh = np.shape(yo)
    # Instance storing variables
    z_sign = np.empty(sh)
    z_sign[:] = np.nan  

    xph = np.empty(sh)
    xph[:] = np.nan  
    
    xph_imag = np.empty(sh)
    xph_imag[:] = np.nan  
    
    amps = np.empty(sh)
    amps[:] = np.nan  
    
    for j, i in enumerate(yo):
        # Zscore
        mean_blank = np.nanmean(i, axis=0)
        std_blank = np.nanstd(i, axis=0)
        z_sign[j, :, :, :] = process.zeta_score(i, mean_blank, std_blank, full_seq = True)

        # Analytic Signal
        tmp = analytic_signal(i)
        
        # Storing
        xph[j, :, :, :] = tmp[0] 
        xph_imag[j, :, :, :] = tmp[1] 
        amps[j, :, :, :] = tmp[2]
        print(f'Analogical signal extraction for trial number {j+1}/{len(z_sign)}!')

    print('Analytic signal extraction for filtered condition in: ' +str(datetime.datetime.now().replace(microsecond=0)-start_analytic_signal))         
    return z_sign, xph, xph_imag, amps

# def phase_latency(xph, fs, ft):
#     dims = xph.shape
#     ph = np.angle(xph)
#     ph_flatten = np.reshape(ph, (dims[0], -1)).transpose()
#     ft_flatten = np.reshape(ft, (dims[0], -1)).transpose()
    
#     pl = np.zeros(np.shape(ph_flatten))
#     for ii, (p,f) in enumerate(zip(ph_flatten, ft_flatten)):    
#         ind0 = np.where(p == 0)# First index in which the value is 0
#         if ind0 != 0:
#             ind0 = ind0 - 1 # Pick the index before the actual 0

#         ind = np.where((p[0:-2]*p[1:-1])< 0) # Time when the phase change of sign
#         if ind !=0:
#             ind = ind - 1 
            
#         if abs(p[ind]) >= (np.pi/2): # Check if the phase in absolute value is more than pi/2
#             ind2 = np.where((p[ind+1:-2]*p[ind+1:-1])< 0)
#             ind2 = ind + ind2
#             ind = ind2 #     
        
#         if (ind0 != np.nan) and (ind0<ind):
#             ind = ind0
    
#         pl[ii,0] = (abs(ph[ind]) / (2*np.pi*f[ind]) ) + (ind/fs)
#     pl = pl.transpose()
#     pl = np.reshape(pl, dims)
#     return pl

def phase_latency(xph, fs, ft, time_point = 0, deg_confidence_interval = 5):
    
    dims = xph.shape
    ph = np.angle(xph)
    ph_flatten = np.reshape(ph, (dims[0], -1)).transpose()
    ft_flatten = np.reshape(ft, (dims[0], -1)).transpose()
    pl = np.empty((dims[1]*dims[2]))
    pl[:] = np.nan
    for ii, (p,f) in enumerate(zip(ph_flatten, ft_flatten)):
        try:
            ind0 = np.where((p[(time_point-1):] >= np.deg2rad(-deg_confidence_interval)) & ((p[(time_point-1):] <= np.deg2rad(deg_confidence_interval))))[0]# First index in which the value is 0
            ind0 = time_point-2 + ind0[0] # Pick the index before the actual 0
        except:
            ind0 = np.nan            

        ind = np.where((p[(time_point-1):-1]*p[time_point:])< 0)[0] # Time when the phase change of sign
        try:
            ind = time_point-2 + ind[0]
        except:
            ind = time_point-2

        if abs(p[ind]) >= ((np.pi/2) - np.deg2rad(deg_confidence_interval)): # Check if the phase in absolute value is more than pi/2
            ind2 = np.where((p[ind:-1]*p[(ind+1):])< 0)[0]
            try:
                ind2 = ind + ind2[0]
                ind = ind2 #    
            except:
                pass
  
        if (ind0 != np.nan) and (ind0<ind):
            ind = ind0

        a = np.divide(abs(p[ind]),(2*np.pi*f[ind]) ) + (ind/fs)
        pl[ii] = a

    pl = pl.transpose()
    pl = np.reshape(pl, (dims[1], dims[2]))
    return pl
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Launching phase latency analysis pipeline')

    parser.add_argument('--path_md', 
                        dest='path_md',
                        type=str,
                        required=True,
                        help='The middle process datafolder path')

    parser.add_argument('--filt', 
                        dest='filt_switch',
                        action='store_true')
    parser.add_argument('--no-filt', 
                        dest='filt_switch', 
                        action='store_false')
    parser.set_defaults(filt_switch=True)
    
    parser.add_argument('--retino', 
                        dest='retino_extraction',
                        action='store_true')
    parser.add_argument('--no-retino', 
                        dest='retino_entraction', 
                        action='store_false')
    parser.set_defaults(retino_store=False)

    parser.add_argument('--store_data', 
                        dest='store_switch',
                        action='store_true')
    parser.add_argument('--no-store_data', 
                        dest='store_switch', 
                        action='store_false')
    parser.set_defaults(retino_store=False)

    parser.add_argument('--vis', 
                        dest='data_vis_switch', 
                        action='store_true')
    parser.add_argument('--no-vis', 
                        dest='data_vis_switch', 
                        action='store_false')
    parser.set_defaults(data_vis_switch=False)  

    parser.add_argument('--green_name', 
                        dest='green_name',
                        type=str,
                        default = 'green01.bmp',
                        required=False)  

    parser.add_argument('--ss_label', 
                        dest='single_stroke_label',
                        type=str,
                        default = 'pos',
                        required=False)  

    parser.add_argument('--am_label', 
                        dest='apparent_motion_label',
                        type=str,
                        default = 'am',
                        required=False)  

    parser.add_argument('--down_tw', 
                        dest='bottom_time_window',
                        type=int,
                        default = 20,
                        required=False)

    parser.add_argument('--up_tw', 
                        dest='upper_time_window',
                        type=int,
                        default = 30,
                        required=False)  

    parser.add_argument('--crop_wind', 
                        dest='dim_crop_window',
                        type=int,
                        default = 150,
                        required=False)  

    parser.add_argument('--tc_wind', 
                        dest='dim_tc_wind',
                        type=int,
                        default = 30,
                        required=False)

    parser.add_argument('--zero', 
                        dest='zero_frames',
                        type=int,
                        default = 20,
                        required=False)

    parser.add_argument('--fs', 
                        dest='fs',
                        type=int,
                        default = 110,
                        required=False)

    parser.add_argument('--ord', 
                        dest='filter_order',
                        type=int,
                        default = 4,
                        required=False)

    parser.add_argument('--f1', 
                        dest='f1',
                        type=int,
                        default = 1,
                        required=False)

    parser.add_argument('--f2', 
                        dest='f2',
                        type=int,
                        default = 20,
                        required=False)

    parser.add_argument('--spacing', 
                        dest='pixel_spacing',
                        type=float,
                        default = .0566,
                        required=False)

    parser.add_argument('--cid', 
                        action='append', 
                        dest='conditions_id',
                        default=None,
                        type=int,
                        help='Conditions to analyze: None by default -all the conditions-')     

    parser.add_argument('--an_name', 
                        dest='name_analysis',
                        type=str,
                        default = 'AMnbStrokes_Spectral_Analysis',
                        required=False)   

    # Example of command line python spectral_analysis.py 
    # --path_md /envau/work/neopto/DATA_AnDO/exp-AM_nbStrokes_BEHAV+VSDI/sub-Ziggy/sess-20210812-001-AM23_05degAmp_16degPS/derivatives/spcbin3_timebin1_zerofrms7_strategymae_n_chunk1_movFalse_deblankTrue/ 
    # --vis --green_name green.bmp --retino --store_data --am_label Stroke --down_tw 30 --up_tw 45
     
    start_process_time = datetime.datetime.now().replace(microsecond=0)
    args = parser.parse_args()

    print(args)
    # Store time boundaries
    time_limits_single = ((args.bottom_time_window, args.upper_time_window))

    # Extract path session from md data folder path
    path_session = args.path_md.split('derivatives')[0]

    # Extract metainformation related to condition positions
    RETINO_POS_AM = retino.get_conditions_correspondance(path_session)

    # Useful variable extraction from session object
    session = md.Session(path_session, logs_switch = False, deblank_switch = False)
    
    #conds = list(session.cond_dict.keys())
    blank_id = session.blank_id
    path_md_files = os.path.join(args.path_md,'md_data')
    
    # Session names extraction
    sub_name, experiment_name, session_name = retino.get_session_metainfo(path_session)
    ID_NAME = sub_name + experiment_name + session_name

    # Instantiate variables
    global_centroid, masks, blobs_pos = list(), list(), list()
    tcs_single_pos, tcs_single_pos_avrg, names_cd = list(), list(), list()
    tcs_ams, tcs_ams_avrg, names_cd_ams= list(), list(), list()
    centroids_singlepos, tc_masks = list(), list()
    filtered_tcs, filtered_tcs_avrg = list(), list()
    filtered_tcs_ams, filtered_tcs_avrg_ams = list(), list()

    # Blank condition loading
    cd_blank = md.Condition()
    cd_blank.load_cond(os.path.join(path_md_files, 'md_data_blank'))
    mean_blank = np.nanmean(cd_blank.averaged_df[:20, :,:], axis=0)
    std_blank = np.nanstd(cd_blank.averaged_df[:20, :,:], axis=0)/np.sqrt(np.shape(cd_blank.averaged_df)[0])
    
    # Loading green
    green_path = utils.find_thing(args.green_name, path_session)
    green = cv.imread(green_path[0], cv.IMREAD_UNCHANGED)

    # Resizing green
    (y_size, x_size) = cd_blank.averaged_df[0, :,:].shape
    x_bnnd_size = x_size
    y_bnnd_size = y_size
    tmp = cv.resize(np.array(green, dtype='float64'), (x_bnnd_size, y_bnnd_size), interpolation=cv.INTER_LINEAR)
    green_ = np.copy(tmp)    

    # Loading handmade mask
    try:
        mask = np.load(os.path.join(path_session, 'derivatives','handmade_mask.npy'))
        mask = mask[0:y_bnnd_size, 0:x_bnnd_size]

    except:
        print('No mask present in derivatives folder for session ' + ID_NAME)
        mask = None

    # Two dictionaries, for type of conditions -pos or am-
    single_pos_conds = {k: v for k,v in session.cond_dict.items() if args.single_stroke_label.lower() in v.lower()}
    am_conds = {k: v for k,v in session.cond_dict.items() if (args.single_stroke_label.lower() not in v.lower()) and (v.lower() != 'blank')}

    # Start from the single stroke conditions for storing and afterward showing the positions in AM conditions
    conds_full = {**single_pos_conds, **am_conds}
    
    # Interesect the set of all the conditions with the picked one in the parser 
    if args.conditions_id is not None:
        conds = {k: v for k,v in conds_full.items() if k in args.conditions_id}
    else:
        conds = conds_full
    
    print(conds)
    #Instance variables
    if args.retino_extraction:
        retino_pos_ = dict()    

    # Conditions loading
    for k,v in conds.items():
        print(v + '\n')
        start_cond_time = datetime.datetime.now().replace(microsecond=0)
        
        # Creation/Check of existence data folder for filtered data
        tmp_filt = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis =  os.path.join(args.name_analysis, ID_NAME, v, 'filtered'))

        # Switch for the retino object
        switch_cond = False
        if args.single_stroke_label in v:
            switch_cond = True
            k_s = 'single stroke'
        else:
            k_s = 'multiple stroke'

        # Instance retino object   
        retino_obj = retino.get_retinotopy(v,
                                            path_md_files,
                                            mask,
                                            green_,
                                            time_limits_single,
                                            args.dim_crop_window,
                                            mean_blank = mean_blank,
                                            std_blank = std_blank,
                                            tc_window_dimension = args.dim_tc_wind,
                                            zero_frames = args.zero_frames,
                                            retino_features = switch_cond,
                                            kind_stroke = k_s)
        maps = retino_obj.signal
        print(f'Time limits for retinotopy detection: {retino_obj.time_limits}')

        if args.retino_extraction:
            retino_pos_[v] = retino_obj.retino_pos
            # Colorcoding for retinotopic positions
            if switch_cond:
                print(retino_pos_)
                indeces_colors = [y for y, kj in enumerate(retino_pos_.keys()) if kj==v][0]
                colrs = [dv.COLORS_7[indeces_colors]]
                print(colrs)
                g_centers = [retino_obj.retino_pos]
            else:
                indeces_colors =[list(retino_pos_.keys()).index(i) for i in RETINO_POS_AM[v]]
                colrs =  [dv.COLORS_7[i] for i in indeces_colors]        
                g_centers = [retino_pos_[i] for i in RETINO_POS_AM[v]]
        else:
            colrs, g_centers = [], []

        if args.data_vis_switch:
            dv.whole_time_sequence(maps, 
                                    mask = mask,
                                    name='z_sequence_'+ v + ID_NAME, 
                                    max=80, min=20, 
                                    global_cntrds = g_centers,
                                    colors_centr = colrs,
                                    name_analysis_= os.path.join(args.name_analysis, ID_NAME, v))


        if args.data_vis_switch and args.retino_extraction:
            if switch_cond:
                # Plotting retinotopic positions over averaged maps
                min_bord = np.nanpercentile(retino_obj.map, 15)
                max_bord = np.nanpercentile(retino_obj.map, 98)

                fig, ax = plt.subplots(1,1, figsize=(9,7), dpi=300)
                ax.contour(retino_obj.blob, 4, colors='k', linestyles = 'dotted')
                pc = ax.pcolormesh(retino_obj.map, vmin=min_bord,vmax=max_bord, cmap=utils.PARULA_MAP)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(pc, shrink=1, ax=ax)
                ax.scatter(retino_obj.retino_pos[0],retino_obj.retino_pos[1],color='r', marker = '+', s=150)
                ax.scatter(retino_obj.distribution_positions[0],retino_obj.distribution_positions[1], color=dv.COLORS_7[list(retino_pos_.keys()).index(v)], marker = '.', s=150)
                ax.vlines(retino_obj.retino_pos[0], 0, retino_obj.map.shape[0], color = dv.COLORS_7[list(retino_pos_.keys()).index(v)], lw= 3, ls='--', alpha=1)
                ax.set_title(ID_NAME + ' condition: ' + v )

                # Storing picture
                tmp = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis = os.path.join(args.name_analysis, ID_NAME, v))
                plt.savefig(os.path.join(tmp, 'averagedheatmap_' +v+ '.svg'))
                print('averagedheatmap_' +v+ '.svg'+ ' stored successfully!')
                plt.savefig(os.path.join(tmp, 'averagedheatmap_' +v+ '.png'))
                plt.close('all')

                # Variables for plotting timecourses and averaged heatmap
                tcs_single_pos.append(retino_obj.time_courses)
                tcs_single_pos_avrg.append(retino_obj.average_time_course)
                names_cd.append(v)
                centroids_singlepos.append([retino_obj.retino_pos])
                tc_masks.append(retino_obj.tc_mask)

            elif (not switch_cond) and args.retino_extraction:
                # Time courses for AM conds 
                xs = list(list(zip(*g_centers))[0])
                ys = list(list(zip(*g_centers))[1])
                a, b = retino.get_trajectory(xs, ys, (75, retino_obj.signal.shape[-1] - 75))
                small_mask = retino.get_mask_on_trajectory((retino_obj.signal.shape[-2], retino_obj.signal.shape[-1]), a, b, radius = 15)
                
                # Time courses making
                retino_obj.time_courses = np.array([process.time_course_signal(np.nan_to_num(w, copy = False, nan=0.0000001, posinf=None, neginf=None), abs(small_mask - 1)) for w in retino_obj.df_fz])
                retino_obj.average_time_course = np.nanmean(retino_obj.time_courses, axis = 0)
                
                # Variables for plotting timecourses and averaged heatmap
                tcs_ams.append(retino_obj.time_courses)
                tcs_ams_avrg.append(retino_obj.average_time_course)
                names_cd_ams.append(v)
                #app_ = v.split('AM_')[1]
                #names_cd_ams.append(app_.replace('nbStrokes', 'AM'))
        
            elif (not switch_cond):# and (not args.retino_extraction):
                circ_mask = blk_file.circular_mask_roi(x_bnnd_size, y_bnnd_size)
                retino_obj.time_courses = np.array([process.time_course_signal(i, circ_mask) for i in retino_obj.df_fz])
                retino_obj.average_time_course = np.nanmean(retino_obj.time_courses, axis = 0)

        #check_presence = utils.find_thing('filtered_df_' + v + '.npy', os.path.join(dv.STORAGE_PATH, ''))
    
        if args.filt_switch:# or (v in list(am_conds.values())): # TO REMOVE AFTER FIRST SAVE OF FILTERED SIGNAL FOR AMS
            # Filtered signal
            yo = np.empty((np.shape(retino_obj.df_fz)))
            start_filt_time = datetime.datetime.now().replace(microsecond=0)
            for j, i in enumerate(retino_obj.df_fz):
                yo[j, :, :, :] = bandpass_filter(i, args.f1, args.f2, args.filter_order, args.fs)
                print(f'Trial number {j+1}/{len(retino_obj.df_fz)} filtered!')
            print('Filtering time for one condition: ' +str(datetime.datetime.now().replace(microsecond=0)-start_filt_time))
            
            if args.store_switch:
                # Create folder for storing filtered data
                np.save(os.path.join(tmp_filt, 'filtered_df_' + v), yo)

        else:
            # Load prefiltered signal
            start_filt_time = datetime.datetime.now().replace(microsecond=0)
            yo = np.load(os.path.join(tmp_filt, 'filtered_df_' + v+'.npy'))
            print('Filtered data loading time for one condition: ' +str(datetime.datetime.now().replace(microsecond=0)-start_filt_time))
        
        # Analytic Signal extraction
        z_sign, xph, xph_imag, amps = extract_analytic_signal(yo)
        analysign = recombine_analytical_signal(xph, xph_imag)
        
        # Store analytic signal
        if args.store_switch:
            # Create folder for storing analytic signal
            tmp_an = dv.set_storage_folder(storage_path = dv.STORAGE_PATH, name_analysis =  os.path.join(args.name_analysis, ID_NAME, v, 'analytic_sign'))
            np.save(os.path.join(tmp_an, 'analysign_real' + v), xph)
            np.save(os.path.join(tmp_an, 'analysign_imag' + v), xph_imag)

        # Phase map extraction
        phm = np.angle(analysign)

        # Normalization: from 0 to 2pi
        phm = phm + np.pi
        mean_phase_map = circmean(phm, axis=0)
        
        if args.data_vis_switch:
            # Time elements:
            signal_dim = list(np.shape(yo))
            no_centroids = list(zip([signal_dim[1]*(None,None)][0][:signal_dim[1]], [signal_dim[1]*(None,None)][0][:signal_dim[1]]))
            tm = np.empty((signal_dim[-2], signal_dim[-1]))
            tm[:] = np.nan
            no_blobs =  [tm]*signal_dim[1]

            # Plotting phase maps 
            dv.whole_time_sequence(mean_phase_map[retino_obj.time_limits[0]-5:retino_obj.time_limits[0]+10, :, :], 
                                    n_columns=5,
                                    mask = mask,
                                    mappa = 'jet',
                                    name='mean_phase_map_'+ v + ID_NAME, 
                                    cntrds = no_centroids, 
                                    blbs = no_blobs, 
                                    max_bord=(5/4)*np.pi, min_bord=np.pi/4,
                                    global_cntrds = g_centers,
                                    colors_centr = colrs,
                                    name_analysis_= os.path.join(args.name_analysis, ID_NAME, v))

        # Storing retino object
        if switch_cond and args.retino_extraction and args.store_switch:
            retino_obj.df_fz = None
            retino_obj.time_courses = None
            retino_obj.store_retino(os.path.join(dv.STORAGE_PATH, args.name_analysis, ID_NAME, v))

        print('Condition ' +v+ ' elaboration in : ' +str(datetime.datetime.now().replace(microsecond=0)-start_cond_time))
    print('Filtering time for one condition: ' +str(datetime.datetime.now().replace(microsecond=0)-start_process_time))








