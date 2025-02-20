import datetime, utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import numpy as np
import os
import process_vsdi as process
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.stats import norm

STORAGE_PATH = '/envau/work/neopto/USERS/GIANCANI/Analysis/'

COLORS_7 = ['crimson', 'tomato', 'magenta', 'darkorange', 'burlywood', 'palevioletred', 'chocolate', 'black', 'white', 'gray']

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
    axs[0].plot(np.arange(shapes[1]), np.nanmean(tc[indeces_select], axis=0), color = 'k', linewidth = 2, label = 'Average among selected trials')
    axs[0].plot(np.arange(shapes[1]), np.nanmean(tc[unselected], axis=0), color = 'red', linewidth = 2, label = 'Average among unselected trials')
    axs[0].legend(loc = 'upper left')
    axs[0].set_ylim(np.nanmin(tc[indeces_select]) - (np.nanmax(tc[indeces_select]) - np.nanmin(tc[indeces_select]))*0.05, 
                    np.nanmax(tc[indeces_select]) + (np.nanmax(tc[indeces_select]) - np.nanmin(tc[indeces_select]))*0.05)
    #plt.subplot(2,3,5)
    for k, i in enumerate(tc[indeces_select[:-1]]):
        axs[1].plot(i, 'gray', linewidth = 0.5)
    axs[1].plot(tc[indeces_select[-1]], 'gray', linewidth = 0.5, label = 'Trials')
    axs[1].plot(np.arange(shapes[1]), np.nanmean(tc[indeces_select], axis=0), color = 'k', linewidth = 2, label = 'Average among selected trials')
    axs[1].plot(np.arange(shapes[1]), np.nanmean(tc[unselected], axis=0), color = 'red', linewidth = 2, label = 'Average among unselected trials')
    axs[1].set_ylim(np.nanmin(tc[indeces_select]) - 0.0005, np.nanmax(tc[indeces_select]) + 0.0005)    
    axs[1].legend(loc = 'upper left')
        
    tmp = path
    if not os.path.exists(os.path.join(tmp,'chunks_analysis')):
        os.makedirs(os.path.join(tmp,'chunks_analysis'))
    plt.savefig(os.path.join(tmp,'chunks_analysis', session_name+'_chunks_0'+str(cd_i)+'.png'))
    plt.close('all')
    return

def retino_pos_visualization(x, y, center, titles, green, name = 'Prova', ext = 'svg', store_path = STORAGE_PATH, name_analysis_ = 'RetinotopicPositions', colors = ['royalblue', 'gold', 'crimson', 'darkorchid','lime', 'black'], lims = [-0.3, 0.3], axis_titles = None):#center):
    fig, axScatter = plt.subplots(figsize=(10, 10))
    if green is not None:
        pc = axScatter.pcolormesh(green, cmap= 'gray')
        axScatter.set_ylim(0, green.shape[0])
        axScatter.set_xlim(0, green.shape[1])
        axScatter.set_aspect(1.)
        shap = green.shape

    else:
        axScatter.set_ylim(lims[0], lims[1])
        axScatter.set_xlim(lims[0], lims[1])
        axScatter.set_aspect(1.)
        shap = (lims[0], lims[1])
        
    

    for i, (x_, y_) in enumerate(zip(x,y)):
        # the scatter plot:
        axScatter.scatter(x_, y_, color = colors[i], label = titles[i], alpha=0.8)
    
    massx = np.nanmax([j for i in x for j in i])
    massy = np.nanmax([j for i in y for j in i])
    
    if axis_titles is not None:
        axScatter.set_xlabel(axis_titles[0])
        axScatter.set_ylabel(axis_titles[1])
    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.5, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.5, pad=0.1, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)
    print(len(x), len(y))
    for i, (x_, y_) in enumerate(zip(x,y)):
        # now determine nice limits by hand:
        hist, bins, _ = axHistx.hist(x_, bins=40, color = colors[i], alpha=0.8)
        # Plot the PDF.
        #xmin, xmax = [0.5e-3, 1.5e-3] #plt.xlim()
        mu_x = np.nanmean(x_)
        std_x = np.nanstd(x_)
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
        mu_y = np.nanmean(y_)
        std_y = np.nanstd(y_)
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
        axScatter.vlines(mu_x, mu_y, abs(shap[0]), color = colors[i], ls = '--', lw=1.5, alpha = 1)
        axScatter.hlines(mu_y, mu_x,  abs(shap[1]), color = colors[i], ls = '--', lw=1.5, alpha = 1 )#
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
    if name is not None:
        tmp = set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)
        plt.savefig(os.path.join(tmp, name + '.png'), format = 'png', dpi =500)
        plt.rc('figure', max_open_warning = 0)
        plt.rcParams.update({'font.size': 12})
        plt.savefig(os.path.join(tmp, name + '.'+ext), format=ext, dpi =500)
        #plt.savefig(os.path.join(tmp, 'Bretz_pos2inAM3_SingleTrial_distrib' + '.pdf'), format='pdf', dpi =500)
        print(name + ext+ ' stored successfully!')
    
    plt.draw()
    plt.show()
    plt.close('all')
    return

def whole_time_sequence3d(data, num_rows = 1, num_cols = 10, vmax=10 , vmin = -1):
    # Adjust the figure size based on the number of subplots
    fig_width = 16
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height), dpi = 150)

    frames_data = data

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8), subplot_kw={'projection': '3d'}, 
                             gridspec_kw={'hspace': 0.5, 'wspace': -0.6}, dpi = 300)

    # Loop through the frames and plot each one in a subplot
    for i, ax in enumerate(axes.flat):
        frame_data = frames_data[i, :]
        # Plot the heatmap for the current frame
        x = np.arange(frame_data.shape[1])
        y = np.arange(frame_data.shape[0])
        x, y = np.meshgrid(x, y)
        frame_data[~mask_] = np.NAN
        heatmap = ax.plot_surface(x, y, frame_data, cmap=utils.PARULA_MAP, vmax = vmax, vmin = vmin)
        # Remove the frame and grid
        ax.set_facecolor('none')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.axis('off')    
        ax.view_init(elev=-60, azim=270)  

    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to remove space

    cbar_ax = fig.add_axes([.95, 0.45, 0.02, 0.12])  # Adjust position and size as needed
    # x, y, width, height 
    fig.colorbar(heatmap, cax=cbar_ax)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()
    return

def whole_time_sequence(data, 
                        global_cntrds = None, 
                        colors_centr = ['black', 'purple', 'aqua'], 
                        centroids_labeling = 'dotted circles',
                        width_line = 1.5,
                        width_contour = .3,
                        cntrds = None, 
                        blbs = None, 
                        max=80, min=10, 
                        mask = None, 
                        name = None, 
                        blur = True, 
                        adaptive_vm = False, 
                        n_columns = 10, 
                        store_path = STORAGE_PATH,
                        handle_lims_blobs = ((97.72, 100)), 
                        name_analysis_ = 'RetinotopicPositions',
                        max_bord = None,
                        min_bord = None,
                        ext= 'png',
                        mappa = utils.PARULA_MAP,
                        titles = None,
                        titles_rows = None,
                        pixel_spacing = None,
                        second_contour = None,
                        kern_median = 5,
                        color_text = 'white',
                        manual_thresh = None,
                        flag_simple_thres = False,
                        render_flag = False,
                        padding_axes = .05,
                        y_title = 1,
                        x_title = 1,
                        coord_title_row = None,
                        font_size = 20,
                        color_contour = 'k'):

    fig = plt.figure(figsize=(15,15), dpi=500)
    fig.subplots_adjust(bottom=0.2)
    #plt.viridis()
    #fig.canvas.draw()

    if titles is None:
        titles = ['']*len(data)

    nrows = int(np.ceil(np.shape(data)[0]/n_columns))  
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(nrows, n_columns),
                    axes_pad=padding_axes,
                    share_all=True,
                    label_mode="L",
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )

    
    # One max-min value for all the colormap. If True, each colormap the values are recomputed
    if (not adaptive_vm) and ((max_bord is None) or (min_bord is None)):
        max_bord = np.nanpercentile(data, max)
        min_bord = np.nanpercentile(data, min)

    # If centroids and blobs are provided, it avoids this computation
    if (cntrds is None) and (blbs is None):# and (mask is not None):
        # Significant threshold for blob thresholding
        bottom_limit = np.nanpercentile(data, 80)
        upper_limit = np.nanpercentile(data, 100)
        # Significant threshold for blob thresholding
        ad_t = False

        if manual_thresh is None:
            manual_th = handle_lims_blobs[0]
            _, centroids, blobs = process.detection_blob(data,
                                                         min_lim = bottom_limit,
                                                         max_lim = upper_limit,
                                                         min_2_lim = manual_th, 
                                                         max_2_lim = handle_lims_blobs[1],  
                                                         adaptive_thresh = ad_t)
            
        elif (manual_thresh is not None) and (not flag_simple_thres):
            a = [process.manual_thresholding(i, manual_thresh) for i in data]
            centroids = list(zip(*a))[1]
            blobs = list(zip(*a))[2]

        elif (manual_thresh is not None) and (flag_simple_thres):
            centroids = []
            blobs = None    
            
        if len(centroids)>0:
            new_centroids = []

            for c in centroids:
                cntrds = []

                if len(c)>0:
                    for x,y, in c:
                        if mask[y, x]:
                            cntrds.append((x,y))

                new_centroids.append(cntrds)
            centroids = new_centroids
            
    else:
        centroids = cntrds
        blobs = blbs

        
    counter_title = 0

    #fig.suptitle(name, fontsize=16)
    for i, (ax, l) in enumerate(zip(grid, data)):
        
        if adaptive_vm:
            max_bord = np.nanpercentile(l, max)
            min_bord = np.nanpercentile(l, min)

        if blur:
            #blurred = gaussian_filter(np.nan_to_num(l, copy=False, nan=np.nanmin(l), posinf=None, neginf=None), sigma=1)
            blurred = median_filter(np.nan_to_num(l, copy=False, nan=np.nanmin(l), posinf=None, neginf=None), (kern_median,kern_median))
        else:
            blurred = l

        if mask is not None:
            blurred[~mask] = np.NAN
        
        p=ax.pcolor(blurred, vmin=min_bord,vmax=max_bord, cmap=mappa)

        if second_contour is not None:
            second_contour = second_contour*mask
            ax.contour(second_contour, 15, colors='white', ls = 'dotted', lw = width_contour)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        # Title for each frame
        ax.annotate(titles[i], xy=(0.5, 1), xytext=(x_title, y_title), 
                    textcoords='offset points', ha='center', 
                    fontsize=font_size, color=color_text)
        ax.set_title("")  # Remove the default title

        if (titles_rows is not None) and ((i%n_columns)==0):
            if coord_title_row is None:
                x_row_title = x_title + 100
                y_row_title = y_title + 100           
            else:
                if isinstance(coord_title_row, tuple):
                    x_row_title, y_row_title = coord_title_row                 
                elif isinstance(coord_title_row, (int, float)):
                    x_row_title = y_row_title = coord_title_row                 
                else:
                    print('Wrong dimension for title coordinates')

            ax.annotate(titles_rows[counter_title], xy=(0.5, 1), xytext=(x_row_title, y_row_title), 
                        textcoords='offset points', ha='center', 
                        fontsize=round(font_size + font_size*.2), color='k')     
            counter_title += 1
                   
        if (manual_thresh is not None) and (flag_simple_thres) and (blobs is None):
            blobs_ = np.zeros(blurred.shape, dtype = bool)
            blobs_[np.where(blurred>manual_thresh)] = 1
            # If there is a mask, it looks for maximi inside the blob
            if (mask is not None) and (np.sum(blobs_) > 20).all():
                blobs_ = blobs_*mask
            if np.nansum(blobs_)>0:
                (x_, y_) = process.find_highest_sum_area(blurred*blobs_, 20)
                centroids.append([(y_, x_)])
            else:
                centroids.append([])

        elif blobs is not None:
            if mask is not None:
                blobs_ = blobs[i]*mask
            else:
                blobs_ = blobs[i]                    
                
        ax.contour(blobs_, width_contour, colors=color_contour, levels=[1])

        if centroids is not None:
            if len(centroids[i])>0:
                for j in centroids[i]:
                    ax.scatter(j[0],j[1],color='r', marker = 'X')

        if global_cntrds is not None:
            for k, cc in zip(global_cntrds, colors_centr):
                # ax.vlines(i[0], 0, blurred.shape[0], color = cc, lw= 1.5)
                if centroids_labeling == 'dotted circles':
                    mask_single_dot = utils.sector_mask(l.shape, 
                                                        (k[1], k[0]), 
                                                        25, 
                                                        (0,360))
                    ax.contour(mask_single_dot, 10, colors=cc, linestyles = 'dotted', lw=width_line)
                elif centroids_labeling == 'vlines':
                    ax.vlines(k[0], 0, blurred.shape[0], color = cc, lw= width_line)

    
    # print(centroids)


    cbar = ax.cax.colorbar(p)
    cbar = grid.cbar_axes[0].colorbar(p)


    if pixel_spacing is not None:
        #fig, ax = plt.subplots()
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(ax.transData,
                                    round(2/pixel_spacing), '2mm', 'lower right', #'upper right' 
                                    pad=0.1,
                                    color='crimson',
                                    frameon=False,
                                    size_vertical=2,
                                    fontproperties=fontprops)

        ax.add_artist(scalebar)

    print(f'Limits values for heatmaps: {max_bord} - {min_bord}')   
    if name is not None:
        tmp = set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)
        #plt.savefig(os.path.join(tmp, name +ext), dpi=1000)
        print(os.system('/usr/bin/sync'))
        plt.savefig(os.path.join(tmp, name + '.'+ext), format = ext, dpi =500)
        print(os.system('/usr/bin/sync'))
        plt.rc('figure', max_open_warning = 0)
        plt.rcParams.update({'font.size': 12})
        # plt.savefig(os.path.join(tmp, name + '.'+ext), format=ext, dpi =500)
        print(name + ext+ ' stored successfully!')

    if render_flag:
        plt.show()
        plt.pause(1)
    plt.close('all')
    return centroids

def plot_lines(*args, titles=None, num_cols=3, y_lim=None, fontsize=12, axis_labels=None, fig_title=None):
    num_lines = len(args)
    
    num_rows = int(np.ceil((num_lines) / num_cols) + 1)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows), sharey=True)
    
    all_data = np.concatenate(args)
    if y_lim is None:
        min_val = np.min(all_data)
        max_val = np.max(all_data)
    else:
        min_val = y_lim[0]
        max_val = y_lim[1]
    
    for i, line_data in enumerate(args):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        ax.plot(line_data)
        
        if titles is None:
            ax.set_title(f'Line {i+1}', fontsize=fontsize)
        else:
            ax.set_title(f'{titles[i]}', fontsize=fontsize)
            
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        
        # Set ylabel only for the first plot of the row
        if col == 0:
            if axis_labels is None:
                ax.set_ylabel('Y Label', fontsize=fontsize)
            else:
                ax.set_ylabel(axis_labels[1], fontsize=fontsize)
        else:
            ax.set_ylabel('')
        
        ax.set_xlabel('')
        ax.set_ylim(min_val, max_val)
    
    # Create a larger subplot at the end for combined plot of all lines
    ax_all = plt.subplot2grid((num_rows, num_cols), (num_rows-1, 0), colspan=num_cols)
    avg_line = np.mean(np.array(args), axis=0)
    for line_data in args:
        ax_all.plot(line_data, alpha=0.5)
    ax_all.plot(avg_line, lw=2, color='red', label='Average')
    ax_all.legend(fontsize=fontsize)
    ax_all.set_title('All Lines with Average', fontsize=fontsize)    
    
    ax_all.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_all.yaxis.label.set_size(fontsize)
    ax_all.set_ylim(min_val, max_val)
    
    # Set x label for the last subplot only
    if axis_labels is None:
        ax_all.set_xlabel('X Label', fontsize=fontsize)
        ax_all.set_ylabel('Y Label', fontsize=fontsize)

    else:
        ax_all.set_xlabel(axis_labels[0], fontsize=fontsize)
        ax_all.set_ylabel(axis_labels[1], fontsize=fontsize)
    
    try:
        # Remove empty subplots if necessary
        if num_lines < num_rows * num_cols:
            for i in range(num_lines, num_rows*num_cols - 1):
                fig.delaxes(axs.flatten()[i])
    except:
        pass

    plt.tight_layout()
    if fig_title:
        fig.suptitle(fig_title, fontsize=fontsize+6, y=1.005)  # Adjust y value for padding
        plt.savefig(os.path.join(fig_title + '.png' ))

    else:
        fig_title = 'Fig_Title'
        plt.show()
    plt.close('all')
    return


def plot_retinotopic_positions(dictionar, titles = ['Inferred centroids', 'Single stroke centroids'], distribution_shown = False, name = None, name_analysis_ = 'RetinotopicPositions', store_path = STORAGE_PATH, ext = '.svg'):#, labs = [ 'Single trial retinotopy', 'Averaged retinotopy']):
    # 
    fig, axs = plt.subplots(1,len(list(dictionar.keys())), figsize=(10*len(list(dictionar.keys())),7))
    if len(list(dictionar.keys()))>1:
        for (ax, (k, v)) in zip(axs, dictionar.items()):
            a = ax.contour(v[1], 4, colors='purple', linestyles = 'dotted')
            #a.collections[0].set_label('Inferred pos2: AM12-pos1')
            pc = ax.pcolormesh(v[3], vmin=v[0][0],vmax=v[0][1], cmap=utils.PARULA_MAP)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(pc, shrink=1, ax=ax)
            if distribution_shown:
                b = ax.scatter(list(v[4][0]), list(v[4][1]),color='purple', marker = 'x', alpha = 0.5, label = titles[0])#, label = 'Single trial retinotopy')
                #b.collections[0].set_label(labs[0])
            for j in v[2]:
                #if l == len(v[2])-1:
                #    ax.scatter(j[0],j[1],color='r', marker = '+', s=150, legend = 'Averaged retinotopy')
                #else:
                c = ax.scatter(j[0],j[1],color='purple', marker = '+', s=150)
            #c.collections[0].set_label(labs[1])
            try:
                a = ax.contour(v[10], 4, colors='k', linestyles = 'dotted')
                #a.collections[0].set_label('Inferred pos2: AM12-pos1')
                for i,  j in enumerate(v[9]):
                    #if l == len(v[2])-1:
                    #    ax.scatter(j[0],j[1],color='r', marker = '+', s=150, legend = 'Averaged retinotopy')
                    #else:
                    if i == len(v[9])-1:
                        titolo = titles[1]
                    else:
                        titolo = None
                    c = ax.scatter(j[0],j[1],color='k', marker = '+', s=150, label=titolo)

            except:
                pass

            ax.set_title(k)
            ax.legend()
    else:
        a = ax.contour(dictionar.values()[1], 4, colors='k', linestyles = 'dotted')
        pc = ax.pcolormesh(dictionar.values()[3], vmin=dictionar.values()[0][0],vmax=dictionar.values()[0][1], cmap=utils.PARULA_MAP)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(pc, shrink=1, ax=ax)
        if distribution_shown:
            b = ax.scatter(list(v[4][0]), list(v[4][1]),color='purple', marker = 'x')#, label = 'Single trial retinotopy')
            #b.collections[0].set_label(labs[0])
        for j in dictionar.values()[2]:
            #if l == len(v[2])-1:
            #    ax.scatter(j[0],j[1],color='r', marker = '+', s=150, legend = 'Averaged retinotopy')
            #else:
            c = ax.scatter(j[0],j[1],color='r', marker = '+', s=150)
        #c.collections[0].set_label(labs[1])
        ax.set_title(k)
        #ax.legend()

    if name is not None:
        tmp = set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)
        #plt.savefig(os.path.join(tmp, name + ext), dpi=1000)
        plt.savefig(os.path.join(tmp, name + ext))
        print(name + ext+ ' stored successfully!')
        plt.close('all')
    return

def plot_averaged_map(name_cond, retino_obj, map, center, min_bord, max_bord, color, session_name, col_distr, name_analysis_ = 'RetinotopicPositions', store_path = STORAGE_PATH, store_pic = True):
    # Plotting retinotopic positions over averaged maps
    fig, ax = plt.subplots(1,1, figsize=(9,7), dpi=300)
    ax.contour(retino_obj.blob, 4, colors='k', linestyles = 'dotted')
    pc = ax.pcolormesh(map, vmin=min_bord,vmax=max_bord, cmap=utils.PARULA_MAP)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(pc, shrink=1, ax=ax)
    ax.scatter(retino_obj.retino_pos[0],retino_obj.retino_pos[1],color='r', marker = '+', s=150)
    ax.scatter(retino_obj.distribution_positions[0],retino_obj.distribution_positions[1], color=col_distr, marker = '.', s=150)
    ax.vlines(center[0], 0, map.shape[0], color = color, lw= 3, ls='--', alpha=1)
    ax.set_title(session_name + ' condition: ' + name_cond )

    if store_pic:
        # Storing picture
        tmp = set_storage_folder(storage_path = store_path, name_analysis = name_analysis_)#os.path.join(name_analysis_, ID_NAME, v))
        # plt.savefig(os.path.join(tmp, 'averagedheatmap_' +name_cond+ '.svg'))
        # print('averagedheatmap_' +name_cond+ '.svg'+ ' stored successfully!')
        plt.savefig(os.path.join(tmp, 'averagedheatmap_' +name_cond+ '.png'))
        plt.close('all')
    else:
        plt.show()
    return

def plot_zmask(Mask, U, cutoff, path_folder, filename = None):

    # Plot the mask
    plt.figure()
    plt.imshow(Mask, cmap='viridis')
    plt.colorbar()
    plt.title("Z-Score Mask")
    plt.savefig(os.path.join(path_folder, title_mask))
    plt.close()

    # Compute histogram with np.histogram
    val_mean = np.nanmean(U)
    U_filled = np.nan_to_num(U, nan=val_mean)  # Replace NaN with the mean of non-NaN values
    U_filled[~np.isfinite(U_filled)] = val_mean  # Replace inf with the mean of non-NaN values
    hist_values, bin_edges = np.histogram(U_filled.ravel(), bins=1500)

    # Plot the histogram using computed values
    plt.figure()
    plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), align='edge', edgecolor='black')

    # Add the cutoff line, ensuring it aligns with the histogram's x-axis scale
    plt.axvline(cutoff, color='r', linewidth=2, label=f'Cutoff: {cutoff:.2f}')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram with Cutoff")
    plt.legend()

    # Save histogram plot
    if filename is not None:
        title_hist = f'histogram_cutoff_{filename}.png'
        title_mask = f'zmask_{filename}.png'
    else:
        title_hist = f'histogram_cutoff.png'
        title_mask = f'zmask.png'

    plt.savefig(os.path.join(path_folder, title_hist))
    plt.close()


import st_builder as st


def plot_st(profilemap,  
            threshold_contour, 
            traj_mask,
            pixel_spacing,
            retinotopic_pos = None,
            retinotopic_time = None, 
            map_type = utils.PARULA_MAP,
            st_title = None,
            onset_time = 4,
            colors_retinotopy = ['crimson', 'tomato', 'magenta'],
            draw_peak_traj = True,
            is_delay = 30,#ms
            sampling_fq = 100,#Hz
            high_level = 5,
            color_peak = 'teal',
            low_level = -1,
            store_path = None):
    
    # Safety checks
#     if (retinotopic_pos is not None) and (retinotopic_time is not None):
#         assert len(retinotopic_pos) == len(colors_retinotopy), 'Mismatch in retinotopic positions numbers and colors available'
    space, time  = profilemap.shape
    timing_frame = int((1/sampling_fq)*1000)
    assert (is_delay/timing_frame)>1, 'Something weird: sampling frequency and timing of a frame incompatible'
    isi_frames   = int(is_delay/timing_frame)

    # Plot colormap
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    fig.set_facecolor('white')
    pc_ = ax.pcolormesh(profilemap, cmap= map_type, vmax = high_level, vmin=low_level)
    
    # Plot intensity contour
    blobs = np.zeros(profilemap.shape, dtype = bool)
#     blobs[np.where(median_filter(profilemap, size=(5,5))>=threshold_contour)] = 1
    blobs[np.where(profilemap>=threshold_contour)] = 1
    ax.contour(blobs, 4, colors='k', alpha = .5, levels=[1])
    
    blobs_ = np.copy(blobs)
    blobs_[np.where(median_filter(profilemap, size=(5,5))>=threshold_contour)] = 1
    
    # Draw peak's trajectory
    if draw_peak_traj:
        a = st.maximi_inda_blob(profilemap, blobs_)
        ax.scatter(list(list(zip(*a))[1]), list(list(zip(*a))[0]), marker = '.', color = 'k')
        ax.plot(list(list(zip(*a))[1]), list(list(zip(*a))[0]), ls = '-', color = 'k', alpha = .3)
    
    if (retinotopic_pos is not None) and (retinotopic_time is not None):
        number_strokes = len(retinotopic_pos)
        # Plot timelines and retinotopic positions
        for n in range(number_strokes):
            print(f'stroke\'s peak {retinotopic_time[n]+n*isi_frames} coordinate')
            ax.scatter(retinotopic_time[n]+n*isi_frames, retinotopic_pos[n], marker = 'o', color = colors_retinotopy[n], s= 100)

    else:
        colors_retinotopy = [color_peak]
        number_strokes = 1
        
    for n in range(number_strokes):
        plt.vlines(onset_time+n*isi_frames, 
                   np.where(traj_mask != 0)[1].min(), 
                   np.where(traj_mask != 0)[1].max(), 
                   color = colors_retinotopy[n], ls ='--', lw=2)
    
    # Plot highest spot
    if len(retinotopic_pos)>0:
        a, b = process.find_highest_sum_area(profilemap*blobs_, 5, None, None, onset_time, 45)
        ax.scatter(b,a, marker = 'o', color = color_peak, s= 100)
        print(a, b)

    # Custom axis
    strokes_onset_times = [onset_time+i*isi_frames for i in range(number_strokes)]
    strokes_onset_times.sort()
    print(strokes_onset_times)
    start_time_instants = [0] + strokes_onset_times
    tmp = start_time_instants + list(np.linspace(start_time_instants[-1], time, (2+(time-start_time_instants[-1])//10)))

    print(tmp)
    ax.set_xticks(tmp)
    labels_ = [item.get_text() for item in ax.get_xticklabels()]
    # x_tmp = np.arange((zero_of_cond-12), (zero_of_cond+30+12), len(tmp))
    list_x = list()
    for i, x in zip(labels_, tmp):
        list_x.append(f'{int((x-(onset_time))*timing_frame)}')
    ax.set_xticklabels(list_x, fontsize = 12)
    ax.set_xlabel('Time - ms', fontsize = 15)

    tmp_y = np.linspace(0, space-10, 9) 
    ax.set_yticks(tmp_y)
    labels_ = [item.get_text() for item in ax.get_yticklabels()]
    list_y = list()
    for y in np.linspace(0, (pixel_spacing*space) , 9):
        list_y.append(f'{y:.1f}')
    ax.set_yticklabels(list_y, fontsize = 12)
    ax.set_ylabel('Space - mm', fontsize = 15)
    ax.set_ylim((np.where(traj_mask != 0)[1].min(), np.where(traj_mask != 0)[1].max()))
    fig.colorbar(pc_) 
                   
    if st_title is not None:
        plt.title(st_title, fontsize = 15)
        if store_path is not None:
            plt.savefig(os.path.join(store_path+ '.pdf'), format = 'pdf', dpi =500)
            plt.savefig(os.path.join(store_path+ '.png'), format = 'png', dpi =500)
    plt.show()
    return (a,b)
