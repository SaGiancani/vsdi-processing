import datetime, utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import process_vsdi as process
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm

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
    session_name = comp[-2].split('sub-')[1]+'-'+comp[-3].split('exp-')[1] + '_' + comp[-1].split('-')[1]    
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
    session_name = comp[-2].split('sub-')[1]+'-'+comp[-3].split('exp-')[1] + '_' + comp[-1].split('-')[1] 
    colors_a = utils.COLORS
    xxx=np.linspace(0.001,np.max(list(zip(*coords))[1]),1000)
    #print(len(l))
    title = f'Condition #{cd_i}' 
    fig = plt.figure(constrained_layout = True, figsize=(25, 10))
    fig.suptitle(title)# Session name
    #plt.title(f'Condition {cond_num}')
    subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[2,1.25])
    axs = subfigs[0].subplots(nrows=1, ncols=3)#, sharey=True)
    # Instance variables
    tmp_u = None
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
    if tmp_u is not None:
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

def retino_pos_visualization(x, y, center, titles, green):
    colors = ['royalblue', 'gold', 'crimson', 'lime', 'black', 'darkorchid']
    fig, axScatter = plt.subplots(figsize=(10, 10))
    pc = axScatter.pcolormesh(green, cmap= 'gray')
    
    for i, (x_, y_) in enumerate(zip(x,y)):
        # the scatter plot:
        axScatter.scatter(x_, y_, color = colors[i], label = titles[i], alpha=0.8)
    
    massx = np.max([j for i in x for j in i])
    massy = np.max([j for i in y for j in i])
    
    axScatter.set_ylim(0, green.shape[0])
    axScatter.set_xlim(0, green.shape[1])
    axScatter.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.5, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.5, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)
    
    for i, (x_, y_) in enumerate(zip(x,y)):
        # now determine nice limits by hand:
        hist, bins, _ = axHistx.hist(x_, bins=40, color = colors[i], alpha=0.8)
        # Plot the PDF.
        #xmin, xmax = [0.5e-3, 1.5e-3] #plt.xlim()
        mu_x = np.mean(x_)
        std_x = np.std(x_)
        # changes here
        p = norm.pdf(bins, mu_x, std_x)           
        axHistx.plot(bins, p/p.sum() * hist.sum(), color='k', alpha=1, lw=1)
        f = p/p.sum() * hist.sum()
        indx = np.argmin(abs(bins-mu_x))
        axHistx.plot(mu_x, f[indx], 'red', marker = 'x')
        axHistx.spines['top'].set_visible(False)
        axHistx.spines['right'].set_visible(False)
        # mu value on the upper histogram distribution
        #axHistx.vlines(mu_x, 0, np.max(hist)+1, color = colors[i], ls = '--', lw=1.5, label = title_center)

        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        # Plot the histogram.
        hist, bins, _ = axHisty.hist(y_, bins=40, orientation='horizontal', color = colors[i], alpha=0.8)
        # Plot the PDF.
        #xmin, xmax = [0.5e-3, 1.5e-3] #plt.xlim()
        mu_y = np.mean(y_)
        std_y = np.std(y_)
        # changes here
        p = norm.pdf(bins, mu_y, std_y)           
        #axHistx.vlines(center[0], 0, np.max([a[0]])+1, color = 'grey', ls = '--', lw=2.5, label = title_center)
        # mu value on the right histogram distribution
        #axHisty.hlines(mu_y, 0, np.max(hist)+1, color = colors[i], ls = '--', lw=1.5)
        axHisty.plot(p/p.sum() * hist.sum(), bins , color='k', lw=1)
        f = p/p.sum() * hist.sum()
        indx = np.argmin(abs(f-mu_y))
        axHisty.plot(f[indx], mu_y, 'red', marker = 'x')
        #if i == len(x)-1:
        #    axHistx.vlines(center[0], 0, np.max([a[0]])+1, color = 'grey', ls = '--', lw=2.5, label = title_center)
        #    axHisty.hlines(center[1], 0, np.max([a[0]])+1, color = 'grey', ls = '--', lw=2.5, label = title_center)
        #else:
        #    axHistx.vlines(center[0], 0, np.max([a[0]])+1, color = 'grey', ls = '--', lw=2.5)
        #    axHisty.hlines(center[1], 0, np.max([a[0]])+1, color = 'grey', ls = '--', lw=2.5)
        axHisty.spines['top'].set_visible(False)
        axHisty.spines['right'].set_visible(False)
        # mu values for the distributions -on the green image-
        axScatter.vlines(mu_x, mu_y, mu_y + massy, color = colors[i], ls = '--', lw=1.5, alpha = 1)
        axScatter.hlines(mu_y, mu_x, mu_x + massx, color = colors[i], ls = '--', lw=1.5, alpha = 1 )#
        if i == len(x)-1:
            lab = r'$\mu$ of distributions'
        else:
            lab= ''
        axScatter.scatter(mu_x, mu_y, marker = '+', color = 'red', s=150, label = lab)


    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.
    axScatter.legend(loc='lower left')
    
    #axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    #axHistx.set_yticks([0, 3, 6])## TO MODIFY

    #axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    #axHisty.set_xticks([])## TO MODIFY
    plt.draw()
    plt.show()
    plt.close('all')
    return

def whole_time_sequence(data, cntrds = None, blbs = None, max=80, min=10, mask = None, name = None, blur = True, adaptive_vm = False, n_columns = 10, store_path = STORAGE_PATH, name_analysis_ = 'RetinotopicPositions'):
    fig = plt.figure(figsize=(20,20))
    fig.subplots_adjust(bottom=0.2)
    #plt.viridis()
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(int(np.ceil(np.shape(data)[0]/n_columns)), n_columns),
                    axes_pad=0.3,
                    share_all=True,
                    label_mode="L",
                    cbar_mode= None#'edge', 'each','single'
                    )
    
    # One max-min value for all the colormap. If True, each colormap the values are recomputed
    if not adaptive_vm:
        max_bord = np.nanpercentile(data, max)
        min_bord = np.nanpercentile(data, min)

    #fig.suptitle(name, fontsize=16)
    for i, (ax, l) in enumerate(zip(grid, data)):
        
        if adaptive_vm:
            max_bord = np.nanpercentile(l, max)
            min_bord = np.nanpercentile(l, min)

        if blur:
            blurred = gaussian_filter(np.nan_to_num(l, copy=False, nan=0.000001, posinf=None, neginf=None), sigma=1)
        else:
            blurred = l

        if mask is not None:
            blurred[~mask] = np.NAN
        
        p=ax.pcolor(blurred, vmin=min_bord,vmax=max_bord, cmap=utils.PARULA_MAP)
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(name)
        #ax.set_xlabel(key, size=28)
        #ax.set_ylabel(key, size=28)

        # If centroids and blobs are provided, it avoids this computation
        if (cntrds is None) and (blbs is None):
            _, centroids, blobs = process.detection_blob(blurred)
        else:
            centroids = [cntrds[i]]
            blobs = blbs[i]

        ax.contour(blobs, 4, colors='k', linestyles = 'dotted')
        for j in centroids:
            ax.scatter(j[0],j[1],color='r', marker = 'X')
            
    if name is not None:
        tmp = set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)
        plt.savefig(os.path.join(tmp, name +'.png'))
        plt.close('all')

    return


def plot_retinotopic_positions(dictionar, distribution_shown = False, name = None, name_analysis_ = 'RetinotopicPositions', store_path = STORAGE_PATH):
    # 
    fig, axs = plt.subplots(1,len(list(dictionar.keys())), figsize=(10*len(list(dictionar.keys())),7))
    if len(list(dictionar.keys()))>1:
        for (ax, (k, v)) in zip(axs, dictionar.items()):
            ax.contour(v[1], 4, colors='k', linestyles = 'dotted')
            pc = ax.pcolormesh(v[3], vmin=v[0][0],vmax=v[0][1], cmap=utils.PARULA_MAP)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(pc, shrink=1, ax=ax)
            if distribution_shown:
                ax.scatter(v[4][0], v[4][1],color='purple', marker = 'x')#, label = 'Single trial retinotopy')
            for l, j in enumerate(v[2]):
                #if l == len(v[2])-1:
                #    ax.scatter(j[0],j[1],color='r', marker = '+', s=150, legend = 'Averaged retinotopy')
                #else:
                ax.scatter(j[0],j[1],color='r', marker = '+', s=150)
            ax.set_title(k)
            ax.legend()
    else:
        ax.contour(v[1], 4, colors='k', linestyles = 'dotted')
        pc = ax.pcolormesh(v[3], vmin=v[0][0],vmax=v[0][1], cmap=utils.PARULA_MAP)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(pc, shrink=1, ax=ax)
        if distribution_shown:
            ax.scatter(v[4][0], v[4][1],color='purple', marker = 'x')#, label = 'Single trial retinotopy')
        for l, j in enumerate(v[2]):
            #if l == len(v[2])-1:
            #    ax.scatter(j[0],j[1],color='r', marker = '+', s=150, legend = 'Averaged retinotopy')
            #else:
            ax.scatter(j[0],j[1],color='r', marker = '+', s=150)
        ax.set_title(k)
        ax.legend()

    if name is not None:
        tmp = set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)
        plt.savefig(os.path.join(tmp, name +'.png'))
        plt.close('all')
    return