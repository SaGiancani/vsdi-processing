import matplotlib.pyplot as plt
import numpy as np
import os
import datetime, process, utils

STORAGE_PATH = '/envau/work/neopto/USERS/GIANCANI/Analysis/'

def set_storage_folder(storage_path = STORAGE_PATH, name_analysis = 'prova'):    
    folder_path = os.path.join(storage_path, name_analysis)               
    if not os.path.exists(folder_path):
    #if not os.path.exists( path_session+'/'+session_name):
        os.makedirs(folder_path)
        #os.mkdirs(path_session+'/'+session_name)
    return folder_path

def latency_error_bars(a, title, name_anls, labels = None, store_path = STORAGE_PATH):
    err = list(zip(*a))[1]
    mean = list(zip(*a))[0]
    success = list(zip(*a))[2]
    x = np.arange(len(mean))+1
    if labels is None:
        labels = x
    fig = plt.figure
    plt.rcParams["figure.autolayout"] = True
    plt.xticks(x)
    fig, ax1 = plt.subplots()
    color = 'tab:orange'
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.set_xlabel('Conditions', color='black')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.errorbar(x, mean, yerr=err, label='both limits (default)', fmt="o", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim((min(mean) - 30, max(mean)+30))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels=labels,rotation=90)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Success Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, success,'o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((0, 1))
    fig.suptitle(title, color='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels=labels,rotation=45)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    tmp = set_storage_folder(storage_path = store_path, name_analysis = name_anls)
    plt.savefig(os.path.join(tmp, title +'.png'))
    #plt.savefig((path_session+'/'session_name +'/'+ session_name+'_roi_0'+str(cd_i)+'.png')
    plt.close('all')
    return

def time_sequence_visualization(start_frame, n_frames_showed, end_frame, data, titles, title_to_print, header, path_, c_ax_= None, circular_mask = True, log_ = None, max_trials = 15):
    start_time = datetime.datetime.now().replace(microsecond=0)
    #session_name = header['path_session'].split('/')[-2]+'-'+header['path_session'].split('/')[-3].split('-')[1]
    comp = os.path.normpath(header['path_session']).split(os.sep)
    session_name = comp[-2]+'-'+comp[-3].split('exp-')[1]    
    # Array with indeces of considered frames: it starts from the last considerd zero_frames
    considered_frames = np.round(np.linspace(start_frame-1, end_frame-1, n_frames_showed))
    # Borders for caxis
    if c_ax_ is None:
        max_bord = np.nanpercentile(data, 85)
        min_bord = np.nanpercentile(data, 10)
    elif c_ax_ is not None:
        max_bord = c_ax_[1]
        min_bord = c_ax_[0]
        
    if log_ is not None:
        print(f'Start frame {start_frame}, {n_frames_showed} frames showed and end frame {end_frame}')
        print(f'Max value heatmap: {max_bord}')
        print(f'Min value heatmap: {min_bord}')
    else:
        log_.info(f'Start frame {start_frame}, {n_frames_showed} frames showed and end frame {end_frame}')
        log_.info(f'Max value heatmap: {max_bord}')
        log_.info(f'Min value heatmap: {min_bord}')
    # Implementation for splitting big matrices for storing
    pieces = int(np.ceil(len(data)/max_trials))
    separators = np.linspace(0, len(data), pieces+1, endpoint=True, dtype=int)
    print(separators)
    count = 0
    for i, n in enumerate(separators):
        if i != 0:
            fig = plt.figure(constrained_layout=True, figsize = (n_frames_showed-2, len(data[separators[i-1]:n, :, :, :])), dpi = 80)
            fig.suptitle(f'Session {session_name}')# Session name
            subfigs = fig.subfigures(nrows=len(data[separators[i-1]:n, :, :, :]), ncols=1)
            for sequence, subfig in zip(data[separators[i-1]:n, :, :, :], subfigs):
                subfig.suptitle(f'{titles[count]}')
                axs = subfig.subplots(nrows=1, ncols=n_frames_showed)

                # Showing each frame
                for df_id, ax in zip(considered_frames, axs):
                    Y = sequence[int(df_id), :, :]
                    if circular_mask:
                        mask = utils.sector_mask(Y.shape, (Y.shape[0]//2, Y.shape[1]//2), (np.min(np.shape(Y)))*0.40, (0,360) )
                        Y[~mask] = np.NAN
                    ax.axis('off')
                    pc = ax.pcolormesh(Y, vmin=min_bord, vmax=max_bord, cmap=utils.PARULA_MAP)
                    del Y
                subfig.colorbar(pc, shrink=1, ax=axs)#, location='bottom')
                count +=1
                
            tmp = path_
            if not os.path.exists(os.path.join(tmp,'activity_maps')):
                os.makedirs(os.path.join(tmp,'activity_maps'))
            plt.savefig(os.path.join(tmp,'activity_maps', session_name+'_piece0'+str(i)+'_'+str(title_to_print)+'.png'))
            #del subfigs
            #del fig
            plt.close('all')
    if log_ is not None:
        log_.info('Plotting heatmaps time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    else:
        print('Plotting heatmaps time: ' +str(datetime.datetime.now().replace(microsecond=0)-start_time))
    return  

def chunk_distribution_visualization(coords, m_norm, l, cd_i, header, tc, indeces_select, mask_array, path):
    strategy = header['strategy']
    #session_name = header['path_session'].split('/')[-2]+'-'+header['path_session'].split('/')[-3].split('-')[1]
    comp = os.path.normpath(header['path_session']).split(os.sep)
    session_name = comp[-2]+'-'+comp[-3].split('exp-')[1]    
    colors_a = utils.COLORS
    xxx=np.linspace(0.001,np.max(list(zip(*coords))[1]),1000)
    #print(len(l))
    title = f'Condition #{cd_i}' 
    fig = plt.figure(constrained_layout = True, figsize=(25, 10))
    fig.suptitle(title)# Session name
    #plt.title(f'Condition {cond_num}')
    subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[2,1.25])
    axs = subfigs[0].subplots(nrows=1, ncols=3)#, sharey=True)

    for i,j in enumerate(l):
        axs[2].plot(xxx, process.log_norm(xxx, j[1], j[2]), color = colors_a[i], alpha = 0.5)
        axs[2].plot(list(zip(*coords))[1], list(zip(*coords))[0], "k", marker=".", markeredgecolor="red", ls = "")

        # Median + StdDev
        # Median + StdDev
        mean_o = np.exp(j[1] + j[2]*j[2]/2.0)
        median_o = np.exp(j[1])
        median_o_std = (median_o + 2*np.sqrt((np.exp(j[2]*j[2])-1)*np.exp(j[1]+j[1]+j[2]*j[2])))
        mean_o_std = mean_o + 2*np.sqrt((np.exp(j[2]*j[2])-1)*np.exp(j[1]+j[1]+j[2]*j[2]))
        #plt.axvline(x=median_o, color = colors_a[-i], linestyle='--')
        axs[2].axvline(x=median_o_std, color = colors_a[i], linestyle='-')
        # Mean + StdDev
        #plt.axvline(x=mean_o, color = colors_a[i+1], linestyle='--')
        axs[2].axvline(x=mean_o_std, color = colors_a[i], linestyle='-')


            # We can set the number of bins with the *bins* keyword argument.
        #axs[0].hist(dist1, bins=n_bins)
        #axs[1].hist(dist2, bins=n_bins)
        #plt.gca().set_title()
        axs[0].set_ylabel(strategy)
        axs[0].set_xlabel('Trials')
        #plt.plot(range(len(mae[i, :])), mae[i, :], marker="o", markeredgecolor="red", markerfacecolor="green", ls="-")    
        axs[0].plot(range(len(m_norm[i])), m_norm[i], marker="o", markeredgecolor="red", markerfacecolor=colors_a[i], ls="")#, marker="o", markeredgecolor="red", markerfacecolor="green")
        #plt.plot(range(len(mse[i, :])), [np.mean(mse[i, :])-0.5*np.std(mse[i, :])]*len(mse[i, :]),  ls="--", color = colors_a[i])
        axs[0].plot(range(len(m_norm[i])), [mean_o_std]*len(m_norm[i]),  ls="-", color = colors_a[i])
        #plt.plot(range(len(mae[i, :])), [median_o_std]*len(mae[i, :]),  ls="-", color = colors_a[-i])
        
        #mse[i, :] 
        #plt.plot(range(0, np.max(mse[0])), )
        axs[1].set_ylabel('Count')
        axs[1].set_xlabel(strategy)
        #plt.gca().set_title(f'Histogram for Condition {cond_num}')
        axs[1].hist(m_norm[i], bins = 50, color=colors_a[i], alpha=0.8)

    axs = subfigs[1].subplots(nrows=1, ncols=2)#, sharey=True)

    unselected = []
    for l, (i, sel) in enumerate(zip(tc, mask_array)):
        if sel == 0:
            col = 'crimson'
            alp = 1
            tmp_u = i
            unselected.append(l)
        #else:
            #col = 'grey'
            #alp = 1
            #tmp_s = i
            axs[0].plot(i, color = col, linewidth = 0.5, alpha = alp)
    #axs[0].plot(np.arange(60),tmp_s, color = 'grey', label = 'Selected trials' )
    shapes = np.shape(tc)
    axs[0].plot(np.arange(shapes[1]), tmp_u, color = 'crimson', linewidth = 0.5, label = 'Unselected trials')
    axs[0].plot(np.arange(shapes[1]), np.mean(tc[indeces_select], axis=0), color = 'k', linewidth = 2, label = 'Average among selected trials')
    axs[0].plot(np.arange(shapes[1]), np.mean(tc[unselected], axis=0), color = 'red', linewidth = 2, label = 'Average among unselected trials')
    axs[0].legend(loc = 'upper left')
    axs[0].set_ylim(np.min(tc[indeces_select]) - (np.max(tc[indeces_select]) - np.min(tc[indeces_select]))*0.05, np.max(tc[indeces_select]) + (np.max(tc[indeces_select]) - np.min(tc[indeces_select]))*0.05)
    #plt.subplot(2,3,5)
    for k, i in enumerate(tc[indeces_select[:-1]]):
        axs[1].plot(i, 'gray', linewidth = 0.5)
    axs[1].plot(tc[indeces_select[-1]], 'gray', linewidth = 0.5, label = 'Trials')
    axs[1].plot(np.arange(shapes[1]), np.mean(tc[indeces_select], axis=0), color = 'k', linewidth = 2, label = 'Average among selected trials')
    axs[1].plot(np.arange(shapes[1]), np.mean(tc[unselected], axis=0), color = 'red', linewidth = 2, label = 'Average among unselected trials')
    axs[1].set_ylim(np.min(tc[indeces_select]) - 0.0005, np.max(tc[indeces_select]) + 0.0005)    
    axs[1].legend(loc = 'upper left')
        
    tmp = path
    if not os.path.exists(os.path.join(tmp,'chunks_analysis')):
        os.makedirs(os.path.join(tmp,'chunks_analysis'))
    plt.savefig(os.path.join(tmp,'chunks_analysis', session_name+'_chunks_0'+str(cd_i)+'.png'))
    plt.close('all')
    return