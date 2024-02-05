import middle_process as md
import argparse
import process_vsdi as process
import os

from scipy.ndimage.filters import gaussian_filter, median_filter

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import AxesGrid


def plot_hist(zeros, output):
    fig, ax = plt.subplots(1,1, figsize=(9,7), dpi=300)    
    _ = ax.hist(zeros[:, :, :, :].ravel(), color = 'k', bins = 1500, alpha=0.8)
    ax.set_title(f'{zeros.shape[1]} time bins - Full frame - {zeros.shape[0]} trials', fontsize = 20)
    ax.set_xbound(0.03, -0.03)
    ax.set_ylabel('Count', fontsize = 15)
    ax.set_xlabel('dF/F0', fontsize = 15)
    plt.savefig(os.path.join(output, 'hist_zeros.png'))
    plt.close('all')        
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Launching zeroframes extraction')
    
    parser.add_argument('--in_path', 
                        dest='path_session',
                        type=str,
                        required=True,
                        help='The session path')
    
    parser.add_argument('--out_path', 
                        dest='path_output',
                        type=str,
                        required=True,
                        help='The output path')
    
    parser.add_argument('--zero', 
                        dest='zero_frames',
                        type=int,
                        default = 20,
                        required=False) 
    
    parser.add_argument('--blank_name', 
                        dest='blank_name',
                        type=str,
                        default = 'blank',
                        required=False) 
    
    args = parser.parse_args()
    

    all_conds = [f.name.split('.pickle')[0] for f in os.scandir(args.path_session) if (f.is_file()) and (f.name.endswith(".pickle"))]
    print(all_conds)

    dict_data = dict()
    for cd in all_conds:
        cd_ = md.Condition() 
        cd_.load_cond(os.path.join(args.path_session, cd))
        cond_name = cd.split('data_')[1]
        print(cond_name)
        indeces_cd = np.where(cd_.autoselection == 1)[0]
        dict_data[cond_name] = cd_.binned_data[indeces_cd, :, :, :]
        del cd_
    
    print('\n')
    print(dict_data.keys())
    print('\n')

    zero_of_cond = args.zero_frames

    # BLANK EXTRACTION
    blank_raw = dict_data[args.blank_name]
    blank_dffz = np.array([process.deltaf_up_fzero(i, zero_of_cond, deblank = True, blank_sign=None) for i in blank_raw])

    num_blank_elements = len(blank_raw)
    average_blank_df = np.nanmean(blank_dffz, axis = 0)
    standard_blank_df = np.nanstd(blank_dffz, axis = 0)/np.sqrt(num_blank_elements)

    ## SIGNAL EXTRACTION
    dict_data_dffz = dict()

    for k,v in dict_data.items():
        print(k)
        dict_data_dffz[k] = np.array([process.deltaf_up_fzero(i, zero_of_cond, deblank = True, blank_sign=average_blank_df) for i in v])

    ## ZERO EXTRACTION
    zeros = np.array([i[:zero_of_cond, :, :] for k, v in dict_data_dffz.items() for i in v])

    ## REFRESH SYSTEM FOR STORING AND LAST COMPUTES
    del dict_data_dffz, blank_dffz, blank_raw, dict_data

    ## STANDARD, MEAN AND STORING
    mean_z = np.nanmean(zeros, axis = (0,1))
    stde_z = np.nanstd(zeros, axis = (0,1))#/(len(zeros)*zero_of_cond)
    print(zeros.shape)
    np.save(os.path.join(args.path_output, 'zeros_full.npy'), zeros)
    # Mean over time and trials
    np.save(os.path.join(args.path_output, 'zeros_mean.npy'), mean_z)
    # Standard error
    np.save(os.path.join(args.path_output, 'zeros_stde.npy'), stde_z)

    plot_hist(zeros, args.path_output)
