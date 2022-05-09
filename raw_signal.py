import argparse, datetime, os, utils
import ana_logs as al
import middle_process as md
import numpy as np

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

    parser.add_argument('--dblnk', 
                        dest='deblank_switch',
                        action='store_true')
    parser.add_argument('--no-dblnk', 
                        dest='deblank_switch', 
                        action='store_false')
    parser.set_defaults(deblank_switch=False)

    parser.add_argument('--raw', 
                        dest='raw_switch',
                        action='store_true')
    parser.add_argument('--no-raw', 
                        dest='raw_switch', 
                        action='store_false')
    parser.set_defaults(raw_switch=True)
    
    parser.add_argument('--cid', 
                    action='append', 
                    dest='conditions_id',
                    default=None,
                    type=int,
                    help='Conditions to analyze: None by default -all the conditions-')

    parser.add_argument('--chunks', 
                        dest='chunks',
                        type=int,
                        default = 1,
                        required=False,
                        help='Number of elements value for autoselection') 

    parser.add_argument('--strategy', 
                        dest='strategy',
                        type=str,
                        default = 'mae',
                        required=False,
                        help='Strategy for the autoselection: choose between mse/mae, statistical, roi -kevin equation-')     

    args = parser.parse_args()
    print(args)
    
    # Check on quality of inserted data
    assert args.spatial_bin > 0, "Insert a value greater than 0"    
    assert args.temporal_bin > 0, "Insert a value greater than 0"    
    start_time = datetime.datetime.now().replace(microsecond=0)
    report = al.get_basereport(args.path_session)
    print(f'The number of all the BLK files for the session is {len(md.get_all_blks(args.path_session))}')
    filt_blks = report.loc[report['behav Correct'] == 1, 'BLK Names']
    filt_blks = filt_blks.dropna(subset=['BLK Names']).tolist()
    print(f'The number of correct behavior BLK files for the same session is {len(filt_blks)}')
    #Loading session
    session = md.Session(**vars(args))
    #np.save('all_blks.npy', np.array(session.all_blks))
    session.get_session()
    #Creating a storing folder
    folder_path = os.path.join(session.header['path_session'], 'derivatives/raw_data_matlab')  
    pos_blks = list(set(session.all_blks).intersection(set(filt_blks)))
    pos_ids = [session.all_blks.index(i) for i in pos_blks]
    #pick_blks = np.array(session.all_blks)[[session.all_blks.index(i) for i in pos_blks]].tolist()
    if not os.path.exists(folder_path):
    #if not os.path.exists( path_session+'/'+session_name):
        os.makedirs(folder_path)
        #os.mkdirs(path_session+'/'+session_name)
    print(f'The number of all BLK indeces {len(session.all_blks)}')
    print(f'The number of selected indeces {len(pos_ids)}')
    #Storing a raw_data matrix per each condition
    for i in np.unique(session.conditions):
        print(f'Condition: {i}')
        ids = np.where(session.conditions == i)[0].tolist()
        common_ids = list(set(ids).intersection(set(pos_ids)))
        print('Considered ids: \n')
        print(common_ids)
        tmp_matrix = session.raw_data[common_ids]
        #np.save(os.path.join(folder_path, f'raw_data_cd{i}.npy'), tmp_matrix)
        utils.socket_numpy2matlab(folder_path, tmp_matrix, substring=f'cd{i}')


    print('Time for raw signal storing: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
