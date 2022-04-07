import argparse, os
import middle_process as mp

PATH_ENVAU = '/envau/work/neopto/DATA_AnDO/'
PATH_HPC = '/hpc/neopto/DATA_AnDO/'

def get_sessions(path_storage, exp_type):
    paths = list()
    if 'VSDI' in exp_type and 'BEHAV' in exp_type:
        ending_string = "BEHAV+VSDI"
    elif 'VSDI' in exp_type and 'BEHAV' not in exp_type:
        ending_string = "_VSDI"
    elif 'BEHAV' in exp_type and 'VSDI' not in exp_type:
        ending_string = "_BEHAV"
    elif 'IOI' in exp_type:
        ending_string = "_IOI"
    elif 'ALL' in exp_type:
        ending_string = ""
        
    exps_list = [f.name for f in os.scandir(path_storage) if (f.name.endswith(ending_string))]

    print(exps_list)
    for exp in exps_list:
        print('\n'+exp)
        path_exp = os.path.join(path_, exp)
        subjs_list = [f.name for f in os.scandir(path_exp)]
        print(subjs_list)
        for sub in subjs_list:
            print(sub)
            path_sub = os.path.join(path_exp, sub)
            sess_list = [f.name for f in os.scandir(path_sub)]
            print(sess_list)
            for sess in sess_list:  
                print(sess)
                path_sess = os.path.join(path_sub, sess)
                paths.append(path_sess)
    return paths

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

    parser.add_argument('--dtrn', 
                        dest='detrend',
                        action='store_true')
    parser.add_argument('--no-dtrn', 
                        dest='detrend', 
                        action='store_false')
    parser.set_defaults(detrend=False)

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

    parser.add_argument('--dmn', 
                        dest='demean_switch',
                        action='store_true')
    parser.add_argument('--no-dmn', 
                        dest='demean_switch', 
                        action='store_false')
    parser.set_defaults(demean_switch=True)

    parser.add_argument('--store', 
                        dest='path_storage',
                        type=str,
                        required=True,
                        help='hpc or envau')

    # Write the different keywords simply one after other, without commas
    # ex. --exp 'VSDI' 'BEHAV'
    parser.add_argument('--exp', 
                        nargs='+', 
                        dest='exp_type',
                        default=['VSDI'],
                        type=str,
                        help='The type of experiment')


    args = parser.parse_args()
    print(args)
    
    # Check on quality of inserted data
    assert args.spatial_bin > 0, "Insert a value greater than 0"    
    assert args.temporal_bin > 0, "Insert a value greater than 0"    
    assert args.zero_frames > 0, "Insert a value greater than 0" 
    assert len(set(['ALL', 'VSDI', 'IOI', 'BEHAV']).intersection(set(args.exp_type))) == len(args.exp_type), \
         "Choose elements among 'ALL', 'VSDI', 'IOI', 'BEHAV'"

    if args.path_storage == 'hpc':
        path_ = PATH_HPC
    elif args.path_storage == 'envau':
        path_ = PATH_ENVAU

    sessions = get_sessions(path_, args.exp_type)


