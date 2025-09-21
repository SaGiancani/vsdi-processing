import datetime, utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

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
                        width_line = 2.5,
                        width_contour = 1,
                        cntrds = None, 
                        blbs = None, 
                        max=80, min=10, 
                        mask = None, 
                        name = None, 
                        blur = True, 
                        adaptive_vm = False, 
                        time_series_data = None,
                        time_series_info = (None, None, None),
                        title_plot = None,
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
                        fixed_contour = None,
                        kern_median = 5,
                        color_text = 'white',
                        manual_thresh = None,
                        second_manual_thresh = None,
                        flag_simple_thres = False,
                        render_flag = False,
                        frame_w  = 1, 
                        frame_h = 2.5,
                        padding_axes = (0.2, 0.2), #height and width
                        y_title = 1,
                        x_title = 1,
                        window_maximi = 20, 
                        coord_title_row = None,
                        font_size = 20,
                        color_contour = 'k',
                        white_background = True):

    if titles is None:
        titles = ['']*len(data)

    # total_subplots = np.shape(data)[0]# + 1  # Extra slot for time course
    # nrows = int(np.ceil(total_subplots / n_columns))
    # if time_series_data is not None:
    #     add_height = 2
    # else:
    #     add_height = 0
        
    # fig_height = 5 + nrows * 2 + add_height  # Base height + extra height per row
    # fig = plt.figure(figsize=(15, fig_height), dpi=500)
    # # Adjust bottom spacing dynamically
    # bottom_padding = 0.1 + (add_height / fig_height)  # More rows → more bottom space
    # fig.subplots_adjust(bottom=bottom_padding)
    
    # if white_background:
    #     fig.patch.set_facecolor('white')    

    # grid = AxesGrid(fig, 111,
    #                 nrows_ncols=(nrows, n_columns),
    #                 axes_pad=padding_axes,
    #                 share_all=True,
    #                 label_mode="L",
    #                 cbar_mode='single',
    #                 cbar_location='right',
    #                 cbar_pad=0.1
    #                 )


    # Fixed frame and padding sizes (in inches)
    pad_h, pad_w = padding_axes       # horizontal & vertical padding between frames

    total_subplots = np.shape(data)[0]
    nrows = int(np.ceil(total_subplots / n_columns))

    # Compute figure size from fixed frame & pad sizes
    fig_w = n_columns * frame_w + (n_columns - 1) * pad_w
    fig_h = nrows * frame_h + (nrows - 1) * pad_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=500)

    if white_background:
        fig.patch.set_facecolor('white')    

    # Build grid with absolute inch-based padding
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(nrows, n_columns),
        axes_pad=(pad_w / frame_w, pad_h / frame_h),  # normalize pad relative to frame size
        share_all=True,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.05)

    
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

        if fixed_contour is not None:
            fixed_contour = fixed_contour*mask
            cs = ax.contour(fixed_contour, 15, colors='white', lw = width_contour)
            for line in cs.collections:
                line.set_alpha(0.3)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        # Title for each frame
        # ax.annotate(titles[i], xy=(0.5, 1), xytext=(x_title, y_title), 
        #             textcoords='offset points', ha='center', 
        #             fontsize=font_size, color=color_text)
        ax.text(0.5, 1.,  # normalized position: center x, slightly above top
                titles[i],
                ha='center', va='bottom',
                transform=ax.transAxes,  # interpret coords relative to Axes
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
                (x_, y_), _ = process.find_highest_sum_area(blurred*blobs_, window_maximi)
                centroids.append([(y_, x_)])
            else:
                centroids.append([])

        elif blobs is not None:
            if mask is not None:
                blobs_ = blobs[i]*mask
            else:
                blobs_ = blobs[i]  

        if pixel_spacing  is not None:
            print(f'Blob surface {np.sqrt(np.nansum(blobs_.ravel())*pixel_spacing):.3f} mm^2')
        ax.contour(blobs_, linewidths=width_contour, colors=color_contour, levels=15)

        if (second_manual_thresh is not None):
            blobs_second = np.zeros(blurred.shape, dtype = bool)
            blobs_second[np.where(blurred>second_manual_thresh)] = 1
            blobs_second = median_filter(blobs_second, (kern_median*2,kern_median*2))
            cs = ax.contour(blobs_second,
                            levels=15, # specify levels first
                            colors='w',
                            linewidths=width_contour, # use linewidths (plural)
                            alpha=0.5)
            # for c in cs.collections:
            #     c.set_alpha(0.5)  # manually set alpha


        if centroids is not None:
            if len(centroids[i])>0:
                for j in centroids[i]:
                    print(j)
                    ax.scatter(j[0],j[1],color='r', marker = 'X', s = 50)

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
                elif centroids_labeling == 'hlines':
                    ax.hlines(k[1], 0, blurred.shape[1], color = cc, lw= width_line)      

        if (pixel_spacing is not None) and (i == len(data)-1):
            #fig, ax = plt.subplots()
            fontprops = fm.FontProperties(size=14)
            scalebar  = AnchoredSizeBar(ax.transData,
                                        round(2/pixel_spacing), '2mm', 'lower right', #'upper right' 
                                        pad=0.1,
                                        color='k',
                                        frameon=False,
                                        size_vertical=5,
                                        fontproperties=fontprops)

            ax.add_artist(scalebar)                                  

    # print(centroids)

    cbar = ax.cax.colorbar(p)
    cbar = grid.cbar_axes[0].colorbar(p)

    # ---- ADDITIONAL ROW FOR TIME COURSES ----
    if time_series_data is not None:
        zero, time_interval, time_bins = time_series_info

        ax_time = fig.add_axes([0.1, 0.15, 0.8, 0.35-(fig_h/150)])    #Left, bottom, width, height
        
        ax_time.spines[['top', 'right']].set_visible(False)

        #x_tc = np.arange(25,45,1)
        if (zero is not None) and  (time_interval is not None) and  (time_bins is not None):
            x_tc = np.arange(-zero*time_interval, (time_bins*time_interval)-zero*time_interval, time_interval)
        else:
            x_tc = np.arange(len(time_series_data[0]))  # Assume all time series have the same length

        ax_time.fill_between(x_tc, 
                             np.nanpercentile(time_series_data, 95, axis = 0), 
                             np.nanpercentile(time_series_data, 10, axis = 0), color = 'k', alpha = 0.1)
        ax_time.plot(x_tc, np.nanmean(time_series_data, axis=0), label = 'Average', color = 'k', lw = 3 )
        ax_time.vlines(0, np.nanpercentile(data, 15), np.nanpercentile(data, 95), ls = '--', lw = 2, color = 'gold')
        ax_time.set_ylim(np.nanpercentile(data, 15), np.nanpercentile(data, 95))
    #         ax.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))

        #ax.legend()
        if title_plot is not None:
            ax_time.set_title(f'{title_plot}', fontsize = 20)
        ax_time.tick_params(axis='both', which='major', labelsize=16)
        ax_time.set_xlabel('Time - ms', fontsize=18)
        ax_time.set_ylabel('Signal', fontsize=18)


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
    return centroids, blobs_

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

def plot_averaged_map(name_cond, blob, 
                      retino_pos, distribution_positions, 
                      map, center, 
                      min_bord, max_bord,
                      color, session_name, 
                      col_distr, kern_median = 5,
                      second_thresh = None, name_analysis_ = 'RetinotopicPositions', 
                      store_path = STORAGE_PATH, store_pic = True):
    # Plotting retinotopic positions over averaged maps
    fig, ax = plt.subplots(1,1, figsize=(9,7), dpi=300)
    if blob is not None:
        cs1 = ax.contour(blob,
                        levels=15, # specify levels first
                        colors='w',
                        linewidths=1, # use linewidths (plural)
                        alpha=0.5)
    if second_thresh is not None:
        blobs_second = np.zeros(map.shape, dtype = bool)
        blobs_second[np.where(map>second_thresh)] = 1
        blobs_second = median_filter(blobs_second, (kern_median,kern_median))
        cs2 = ax.contour(blobs_second,
                         levels=15, # specify levels first
                         colors='k',
                         linewidths=1, # use linewidths (plural)
                         alpha=0.5)

    pc = ax.pcolormesh(map, vmin=min_bord,vmax=max_bord, cmap=utils.PARULA_MAP)
    # ax.set_xticks([])
    # ax.set_yticks([])
    fig.colorbar(pc, shrink=1, ax=ax)
    if retino_pos is not None:
        ax.scatter(retino_pos[0], retino_pos[1],color='r', marker = '+', s=150)
    if distribution_positions is not None:
        ax.scatter(distribution_positions[0], distribution_positions[1], color=col_distr, marker = '.', s=150)
    if center is not None:
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
    
    # Save histogram plot
    if filename is not None:
        title_hist = f'histogram_cutoff_{filename}.png'
        title_mask = f'zmask_{filename}.png'
    else:
        title_hist = f'histogram_cutoff.png'
        title_mask = f'zmask.png'

    # Plot the mask
    plt.figure()
    plt.imshow(Mask, cmap='viridis')
    plt.colorbar()
    plt.title("Z-Score Mask")
    plt.savefig(os.path.join(path_folder, title_mask))
    plt.close()

    try:
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

        plt.savefig(os.path.join(path_folder, title_hist))
        plt.close()
    except:
        print('Unable to store histogram of pixels')

def plot_x_medians(summary_df, conditions=None, figsize=(10, 6), output_path=None, multi_session=False):
    """
    Plot x_median values with shaded x_widths across repetitions.
    Labels are written directly at the end of each line (or session dots if multi_session=True).

    Parameters:
    - summary_df: pandas DataFrame from summary_to_dataframe
    - conditions: list of condition names to filter (optional)
    - figsize: size of the plot
    - output_path: if provided, saves the figure
    - multi_session: if True, scatter points per session instead of averaging
    """
    df = summary_df.copy()

    if conditions is not None:
        df = df[df['condition'].isin(conditions)]

    def get_color(cond):
        if cond.startswith('AM2'):
            return 'turquoise'
        elif cond.startswith('AM3'):
            return 'teal'
        else:
            return 'k'

    fig, ax = plt.subplots(figsize=figsize)

    grouped = df.groupby('condition')

    for cond, cond_df in grouped:
        print(cond)
        color = get_color(cond)

        if multi_session:
            # Scatter each session’s repetitions
            for session_id, sess_df in cond_df.groupby(['session']):
                print(session_id)
                sess_df = sess_df.sort_values('repeatition')
                reps = sess_df['repeatition'].values
                meds = sess_df['x_median'].values

                ssub = session_id[0].split('AM')[0]
                date = session_id[0].split('VSDI')[1][0:7]
                
                # Plot as dots, optionally with connecting line per session
                ax.plot(reps, meds, marker='o', linestyle='-', linewidth=0.5, markersize=6, color=color, alpha=0.7)

                # Optionally add session ID for debugging
                ax.text(reps[-1] + 0.1, meds[-1], f"{ssub}_{date}_{cond}", fontsize=8, alpha=0.6)
        else:
            # Aggregate for single-session style
            agg_df = cond_df.groupby('repeatition').agg({
                'x_median': 'mean',
                'x_width': 'mean'
            }).reset_index().sort_values('repeatition')

            reps = agg_df['repeatition'].values
            meds = agg_df['x_median'].values
            widths = agg_df['x_width'].values

            # Median line
            ax.plot(reps, meds, marker='o', color=color)

            # Optional shaded area
            # lower = meds - widths / 2
            # upper = meds + widths / 2
            # ax.fill_between(reps, lower, upper, alpha=0.2, color=color)

            # Label only once at the last point
            label_text = 'pos' if 'pos' in cond else cond
            ax.text(reps[-1] + 0.1, meds[-1], label_text, color=color,
                    fontsize=12, verticalalignment='center', horizontalalignment='left')

    # Axis formatting
    tmp = df['x_median'].abs().max() * 1.5
    n_reps = df['repeatition'].nunique()
    ax.set_ylim(-tmp, tmp)
    ax.set_xlim(-0.5, n_reps - 0.5 + 1)
    ax.set_xticks(np.arange(0, n_reps, 1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.hlines(0, 0, n_reps - 1, ls='--', color='k', alpha=0.8)

    session_title = df['session'].iloc[0] if not multi_session else "MULTI-SESSION"
    ax.set_title(f"X-Medians — Session {session_title}", fontsize=20)
    ax.set_xlabel("Repetition", fontsize=18)
    ax.set_ylabel("Space (dva)", fontsize=18)
    plt.tight_layout()
    plt.show()

    if output_path is not None:
        sess_id = df['session'].iloc[0] if not multi_session else "multi"
        output_path = f'median_summary_AM_{sess_id}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.pause(.1)

def plot_normalized_distributions(dict_norm_dist, cond_dict, meta_data_cd):
    for session, conds_dict in dict_norm_dist.items():
        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        color_map = {
            2: ('turquoise', 'gray'),
            3: ('teal', 'black')
        }
        
        for cond, vv in cond_dict[session].items():
            if cond not in conds_dict:
                continue  # Skip if AM condition is missing

            x__am, y__am = list(), list()
            for n, i in enumerate(conds_dict[cond]):
                x_amlast, y_amlast = i
    
                num_trials = len(vv)
                color, color_sub = color_map.get(num_trials, ('red', 'darkred'))  # fallback colors
    
                # Plot AM condition
                ax.scatter(x_amlast, y_amlast, color=color, s=15, alpha=0.5)
                ax.scatter(np.nanmedian(x_amlast), np.nanmedian(y_amlast), color=color, marker='+', s=150)
                ax.text(np.nanmedian(x_amlast) + 0.02, np.nanmedian(y_amlast) + 0.02, f'{meta_data_cd[cond]}', fontsize=8 + (n+2))
                x__am.append(np.nanmedian(x_amlast))
                y__am.append(np.nanmedian(y_amlast))
            ax.plot(x__am, y__am, ls = '--', color = 'k')    
            # Plot corresponding SUB condition if available
            sub_cond = next((k for k in conds_dict if k.startswith(f'{cond}-')), None)
            if sub_cond:
                x__, y__ = list(), list()
                for i in conds_dict[sub_cond]:
                    x_sub, y_sub = i
                    ax.scatter(x_sub, y_sub, color=color_sub, s=15, alpha=0.5)
                    ax.scatter(np.nanmedian(x_sub), np.nanmedian(y_sub), color=color_sub, marker='+', s=150)
                    ax.text(np.nanmedian(x_sub) + 0.02, np.nanmedian(y_sub) + 0.02, f'{meta_data_cd[sub_cond]}', fontsize=8 + (n+2))
                    x__.append(np.nanmedian(x_sub))
                    y__.append(np.nanmedian(y_sub))
            else:
                print(f'{cond} has no possible subtraction')

        # Plot origin and labels
        ax.scatter(0, 0, color='gold', marker='o', s=150)
        ax.set_xlabel('Space - dva', fontsize=22)
        ax.set_ylabel('Space - dva', fontsize=22)

        # Legend
        legend_elements = [
            Patch(facecolor='turquoise', edgecolor='turquoise', label='AM len=2'),
            Patch(facecolor='gray', edgecolor='gray', label='SUB len=2'),
            Patch(facecolor='teal', edgecolor='teal', label='AM len=3'),
            Patch(facecolor='black', edgecolor='black', label='SUB len=3'),
            Patch(facecolor='gold', edgecolor='gold', label='Center')
        ]
        ax.legend(handles=legend_elements, fontsize=16, loc='upper right')

        plt.pause(0.1)

from scipy.stats import norm, iqr

def freedman_diaconis_bins(data):
    """Compute optimal number of bins using Freedman–Diaconis rule."""
    data = np.asarray(data)
    h = 2 * iqr(data) / (len(data) ** (1/3))
    if h == 0:  # Handle zero spread
        return 10
    bins = int(np.ceil((data.max() - data.min()) / h))
    return max(bins, 5)

def plot_two_distributions(
    data1, data2,
    bins='auto',
    color1='skyblue', color2='salmon',
    label1='Dist 1', label2='Dist 2',
    xlim=None,  # <-- NEW
    ylim=None,  # <-- NEW
    save_figure = True):
    """
    Plots two histograms normalized to % of total per dataset,
    with Gaussian fits scaled to match percentages.
    """
    combined = np.concatenate([data1, data2])
    if bins == 'auto':
        bins = freedman_diaconis_bins(combined)

    edges = np.linspace(combined.min(), combined.max(), bins + 1)
    bin_width = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2

    counts1, _ = np.histogram(data1, bins=edges)
    counts2, _ = np.histogram(data2, bins=edges)
    counts1 = counts1 / len(data1) * 100
    counts2 = counts2 / len(data2) * 100

    mu1, std1 = norm.fit(data1)
    mu2, std2 = norm.fit(data2)
    x = np.linspace(combined.min(), combined.max(), 500)
    pdf1 = norm.pdf(x, mu1, std1) * bin_width * 100
    pdf2 = norm.pdf(x, mu2, std2) * bin_width * 100

    plt.bar(centers, counts1, width=bin_width, alpha=0.5, color=color1, label=label1)
    plt.bar(centers, counts2, width=bin_width, alpha=0.5, color=color2, label=label2)

    plt.plot(x, pdf1, color=color1, lw=2)
    plt.plot(x, pdf2, color=color2, lw=2)

    plt.xlabel("Value")
    plt.ylabel("Percentage per bin (%)")
    plt.legend()
    plt.title(f"Two Normalized Distributions with Gaussian Fits (bins={bins})")

    if xlim is not None:   # <-- apply fixed limits
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_figure:
        plt.savefig(f'{label1}_{label2}.svg', dpi=300)
        plt.savefig(f'{label1}_{label2}.png', dpi=300)
    plt.show()


from matplotlib.lines import Line2D
from collections import defaultdict
import math
import scipy.stats as stats

class PeakPlotter:
    def __init__(self, results_dict):
        """
        Initialize the plotter with your results dictionary
        
        Parameters:
        results_dict: dict containing session data with 'prediction' and 'subtraction' keys
        """
        self.results_dict = results_dict
        
        # Default marker styles for different conditions
        self.default_marker_styles = {
            ('nonlin', 2, 0.5): 'o',    # Circle
            ('nonlin', 2, 1): 's',      # Square
            ('nonlin', 3, 0.5): '^',    # Triangle up
            ('nonlin', 3, 1): 'v',      # Triangle down
            ('lin', 2, 0.5): 'p',       # Pentagon
            ('lin', 2, 1): 'P',         # Plus (filled)
            ('lin', 3, 0.5): '*',       # Star
            ('lin', 3, 1): 'X',         # X (filled)
        }
        
        # Default colors
        self.pos_color = 'gold'
        self.neg_color = 'royalblue'
    
    def extract_peaks_data(self, sessions=None, methods=None, n_pos_values=None, 
                          isi_values=None, dir_values=None):
        """
        Extract peaks data from MapExtractionResults objects
        
        Parameters:
        sessions: list of session keys to include (None = all sessions)
        methods: list of methods to include ['prediction', 'subtraction'] (None = both)
        n_pos_values: list of n_pos values to include (None = all available)
        isi_values: list of isi values to include (None = all available)  
        dir_values: list of direction keys to include (None = all available, e.g., [-1, 1])
        
        Returns:
        tuple: (pos_peaks_data, neg_peaks_data, pos_values_data, neg_values_data)
        """
        if sessions is None:
            sessions = list(self.results_dict.keys())
        if methods is None:
            methods = ['prediction', 'subtraction']
        
        # Initialize nested dictionaries
        pos_peaks_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        neg_peaks_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        pos_values_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        neg_values_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for session in sessions:
            if session not in self.results_dict:
                print(f"Warning: Session {session} not found in results")
                continue
                
            for method in methods:
                if method not in self.results_dict[session]:
                    print(f"Warning: Method {method} not found in session {session}")
                    continue
                
                results = self.results_dict[session][method]
                
                # Map method names to plot categories
                kind = 'nonlin' if method == 'prediction' else 'lin'
                
                      
                # Iterate through available n_pos and isi values
                if hasattr(results, 'pos_peaks'):
                    for n_pos in results.pos_peaks.keys():
                        if n_pos_values is None or n_pos in n_pos_values:
                            for isi in results.pos_peaks[n_pos].keys():
                                if isi_values is None or isi in isi_values:
                                    # Extract positive peaks for each direction
                                    for direction in results.pos_peaks[n_pos][isi].keys():
                                        if dir_values is None or direction in dir_values:
                                            try:
                                                # Get coordinate pairs for this direction
                                                coord_pairs = [i[0] for i in results.pos_peaks[n_pos][isi][direction]]
                                                pos_value  =  [i[1] for i in results.pos_peaks[n_pos][isi][direction]]
                                                pos_values_data[kind][n_pos][isi].extend(pos_value)
                                                
                                                # Get reference value from results.peaks and subtract from y-coordinates
                                                if hasattr(results, 'peaks') and direction in results.peaks[n_pos][isi]:
                                                    reference_value = results.peaks[n_pos][isi][direction]
                                                    if isinstance(reference_value, (list, tuple)):
                                                        reference_value = reference_value[0]  # Use first value if it's a list
                                                    
                                                    # Subtract reference from y-coordinate (first dimension) of each peak
                                                    normalized_value = np.array(list(zip(*coord_pairs))[0]) - reference_value 
                                                    pos_peaks_data[kind][n_pos][isi].extend(zip(normalized_value, list(zip(*coord_pairs))[1]))
                                                else:
                                                    # Use y-coordinates directly if no reference available
                                                    y_coords = [coord_pair[0] for coord_pair in coord_pairs]
                                                    x_coords = [coord_pair[1] for coord_pair in coord_pairs]
                                                    
                                                    pos_peaks_data[kind][n_pos][isi].extend((x_coords, y_coords))
                                                        
                                            except Exception as e:
                                                print(f"Warning: Could not extract pos_peaks for {session}, {method}, n_pos={n_pos}, isi={isi}, dir={direction}: {e}")
                                    
                # Extract negative peaks for each direction
                if hasattr(results, 'neg_peaks'):
                    for n_pos in results.neg_peaks.keys():
                        if n_pos_values is None or n_pos in n_pos_values:
                            for isi in results.neg_peaks[n_pos].keys():
                                if isi_values is None or isi in isi_values:
                                    # Extract positive peaks for each direction
                                    for direction in results.neg_peaks[n_pos][isi].keys():
                                        if dir_values is None or direction in dir_values:
                                            try:
                                                # Get coordinate pairs for this direction
                                                coord_pairs = [i[0] for i in results.neg_peaks[n_pos][isi][direction]]
                                                neg_value  =  [i[1] for i in results.neg_peaks[n_pos][isi][direction]]
                                                neg_values_data[kind][n_pos][isi].extend(neg_value)
                    
                                                # Get reference value from results.peaks and subtract from y-coordinates
                                                if hasattr(results, 'peaks') and direction in results.peaks[n_pos][isi]:
                                                    reference_value = results.peaks[n_pos][isi][direction]
                                                    if isinstance(reference_value, (list, tuple)):
                                                        reference_value = reference_value[0]  # Use first value if it's a list
                                                    
                                                    # Subtract reference from y-coordinate (first dimension) of each peak
                                                    normalized_value = np.array(list(zip(*coord_pairs))[0]) - reference_value 
                                                    neg_peaks_data[kind][n_pos][isi].extend(zip(normalized_value, list(zip(*coord_pairs))[1]))
                                                else:
                                                    # Use y-coordinates directly if no reference available
                                                    y_coords = [coord_pair[0] for coord_pair in coord_pairs]
                                                    x_coords = [coord_pair[1] for coord_pair in coord_pairs]
                                                    neg_peaks_data[kind][n_pos][isi].extend((x_coords, y_coords))
                                                        
                                            except Exception as e:
                                                print(f"Warning: Could not extract neg_peaks for {session}, {method}, n_pos={n_pos}, isi={isi}, dir={direction}: {e}")
        
        # IMPORTANT: Return statement moved outside all loops!
        return pos_peaks_data, neg_peaks_data, pos_values_data, neg_values_data

    def inspect_data(self, session_name=None, method='prediction', n_pos=3, isi=0.5):
        """
        Inspect the data structure for debugging
        """
        if session_name is None:
            session_name = list(self.results_dict.keys())[0]
        
        results = self.results_dict[session_name][method]
        
        print(f"Inspecting {session_name}, {method}, n_pos={n_pos}, isi={isi}")
        print(f"pos_peaks: {results.pos_peaks[n_pos][isi]}")
        print(f"neg_peaks: {results.neg_peaks[n_pos][isi]}")
        print(f"peaks: {results.peaks[n_pos][isi]}")
        
        # Show what gets extracted
        pos_data, neg_data, pos_vals, neg_vals = self.extract_peaks_data(
            sessions=[session_name], methods=[method], 
            n_pos_values=[n_pos], isi_values=[isi]
        )
        
        kind = 'nonlin' if method == 'prediction' else 'lin'
        print(f"\nExtracted data:")
        print(f"pos_peaks_data: {pos_data[kind][n_pos][isi]}")
        print(f"pos_values_data: {pos_vals[kind][n_pos][isi]}")
        print(f"neg_peaks_data: {neg_data[kind][n_pos][isi]}")  
        print(f"neg_values_data: {neg_vals[kind][n_pos][isi]}")
    
    def plot_peaks(self, sessions=None, methods=None, n_pos_values=None, isi_values=None, 
                   dir_values=None, marker_styles=None, threshold_alpha=2, dim_mod_scatter=50,
                   scale_factor=0.1, figsize=(10, 8), xlim=(-1, 30), ylim=(-20, 20),
                   title="Visualization of Point Coordinates by Condition"):
        """
        Plot peaks data with customizable parameters
        
        Parameters:
        sessions: list of session keys to include (None = all sessions)
        methods: list of methods to include ['prediction', 'subtraction'] (None = both)
        n_pos_values: list of n_pos values to include (None = all available)
        isi_values: list of isi values to include (None = all available)
        dir_values: list of dir values to include (None = all available)
        marker_styles: dict mapping (kind, n_pos, isi) to marker symbols
        threshold_alpha: threshold for alpha transparency
        dim_mod_scatter: size multiplier for scatter points
        scale_factor: scaling factor for y coordinates
        figsize: figure size tuple
        xlim, ylim: axis limits
        title: plot title
        
        Returns:
        fig, ax: matplotlib figure and axis objects
        """
        
        # Extract data
        pos_peaks_data, neg_peaks_data, pos_values_data, neg_values_data = self.extract_peaks_data(sessions, methods, 
                                                                                                   n_pos_values, isi_values, 
                                                                                                   dir_values)

        # Use default marker styles if none provided
        if marker_styles is None:
            marker_styles = self.default_marker_styles
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        poss = defaultdict(list)
        negs = defaultdict(list)
        # Plot positive peaks (Facilitation)
        for kind in pos_peaks_data.keys():
            for n_pos in pos_peaks_data[kind].keys():
                for isi in pos_peaks_data[kind][n_pos].keys():
                    points = pos_peaks_data[kind][n_pos][isi]
                    values = pos_values_data[kind][n_pos][isi]

                    if points:
                        y, x = zip(*points)
                        y_scaled = [yi * scale_factor for yi in y]
                        
                        # Get marker style
                        marker_key = (kind, n_pos, isi)
                        marker = marker_styles.get(marker_key, 'o')

                        if isinstance(y_scaled, list):
                            poss[kind].extend(y_scaled)
                        else:
                            poss[kind].append(y_scaled)
                            
                        # Set alpha based on threshold and handle negative values
                        if values:
                            alphas = [1 if abs(v) >= threshold_alpha else 0.3 for v in values]
                            sizes = np.abs(np.array(values)) * dim_mod_scatter  # Use absolute value for size
                        else:
                            alphas = 1
                            sizes = 20  # Default size if no values
                        
                        label = f'{kind} {n_pos} dots {isi} spacing (Facilitation)'
                        ax.scatter(x, y_scaled, 
                                   c=self.pos_color, 
                                   marker=marker,
                                   label=label, 
                                   s=sizes, 
                                   alpha=alphas)
        
        # Plot negative peaks (Suppression)
        for kind in neg_peaks_data.keys():
            for n_pos in neg_peaks_data[kind].keys():
                for isi in neg_peaks_data[kind][n_pos].keys():
                    points = neg_peaks_data[kind][n_pos][isi]
                    values = neg_values_data[kind][n_pos][isi]
                    
                    if points:
                        y, x = zip(*points)
                        y_scaled = [yi * scale_factor for yi in y]
                        # Get marker style
                        marker_key = (kind, n_pos, isi)
                        marker = marker_styles.get(marker_key, 's')
            
                        if isinstance(y_scaled, list):
                            negs[kind].extend(y_scaled)
                        else:
                            negs[kind].append(y_scaled)
                        
                        # Set alpha based on threshold and handle negative values
                        if values:
                            alphas = [1 if abs(v) >= threshold_alpha else 0.3 for v in values]
                            sizes = np.abs(np.array(values)) * dim_mod_scatter  # Use absolute value for size
                        else:
                            alphas = 1
                            sizes = 20  # Default size if no values
                        
                        label = f'{kind} {n_pos} dots {isi} spacing (Suppression)'
                        ax.scatter(x, y_scaled, c=self.neg_color, marker=marker,
                                 label=label, s=sizes, alpha=alphas)
        
        # Customize plot
        ax.set_xlabel('Time - ms', fontsize=15)
        ax.set_ylabel('Space - mm', fontsize=15)
        ax.set_title(title, fontsize=16, pad=20)
        
        # Add reference lines
        ax.hlines(0, xlim[0], xlim[1], ls='--', color='k', linewidth=1, label='Last dot pos')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Add grid
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Create custom legend
        legend_elements = []
        
        # Add marker styles to legend
        used_combinations = set()
        for kind in list(pos_peaks_data.keys()) + list(neg_peaks_data.keys()):
            for n_pos in list(pos_peaks_data.get(kind, {}).keys()) + list(neg_peaks_data.get(kind, {}).keys()):
                for isi in list(pos_peaks_data.get(kind, {}).get(n_pos, {}).keys()) + list(neg_peaks_data.get(kind, {}).get(n_pos, {}).keys()):
                    combination = (kind, n_pos, isi)
                    if combination not in used_combinations:
                        marker = marker_styles.get(combination, 'o')
                        legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                                      label=f'{kind} {n_pos}d {isi}s',
                                                      markerfacecolor='gray', markersize=10))
                        used_combinations.add(combination)
        
        # Add color legend
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Facilitation',
                                    markerfacecolor=self.pos_color, markersize=10))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label='Suppression',
                                    markerfacecolor=self.neg_color, markersize=10))
       
        ax.legend(handles=legend_elements, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

        # --- Perform statistical tests ---
        tmp_dim = np.nanmin([len(negs['lin']), len(negs['nonlin'])])
        tmp_dim_ = np.nanmin([len(poss['lin']), len(poss['nonlin'])])
        
        t_stat_s, p_ttest_s = stats.ttest_rel(negs['lin'], poss['lin'])
        t_stat_cross_n, p_ttest_cross_n = stats.ttest_rel(negs['lin'][:tmp_dim], negs['nonlin'][:tmp_dim])
        t_stat_p, p_ttest_p = stats.ttest_rel(negs['nonlin'], poss['nonlin'])
        t_stat_cross_p, p_ttest_cross_p = stats.ttest_rel(poss['lin'][:tmp_dim_], poss['nonlin'][:tmp_dim_])
        
        w_stat_s, p_wilcoxon_s = stats.wilcoxon(negs['lin'], poss['lin'])
        w_stat_cross_n, p_wilcoxon_cross_n = stats.wilcoxon(negs['lin'][:tmp_dim], negs['nonlin'][:tmp_dim])
        w_stat_p, p_wilcoxon_p = stats.wilcoxon(negs['nonlin'], poss['nonlin'])
        w_stat_cross_p, p_wilcoxon_cross_p = stats.wilcoxon(poss['lin'][:tmp_dim_], poss['nonlin'][:tmp_dim_])
    
        print(f"Paired t-test lin-subtraction: t = {t_stat_s:.3f}, p = {p_ttest_s:.4f}")
        print(f"Paired t-test nonlin-lin negs: t = {t_stat_cross_n:.3f}, p = {p_ttest_cross_n:.4f}")
        print(f"Paired t-test nonlin-predictions: t = {t_stat_p:.3f}, p = {p_ttest_p:.4f}")
        print(f"Paired t-test nonlin-lin poss: t = {t_stat_cross_p:.3f}, p = {p_ttest_cross_p:.4f}")
        
        print(f"Wilcoxon signed-rank test lin-subtraction: W = {w_stat_s:.3f}, p = {p_wilcoxon_s:.4f}")
        print(f"Wilcoxon signed-rank test nonlin-lin negs:: W = {w_stat_cross_n:.3f}, p = {p_wilcoxon_cross_n:.4f}")
        print(f"Wilcoxon signed-rank test nonlin-predictions: W = {w_stat_p:.3f}, p = {p_wilcoxon_p:.4f}")
        print(f"Wilcoxon signed-rank test nonlin-lin poss: W = {w_stat_cross_p:.3f}, p = {p_wilcoxon_cross_p:.4f}")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig, ax

    def visualize_map_with_peaks_fullportion_ax(self, q, baseline, peak, direction, pos_peak, neg_peak, ax, 
                                                clims=0.1, cmap='viridis'):
        """
        Helper method for visualizing maps with peaks on a given axis
        """
        if clims is None:
            clims = np.nanpercentile(q, 95)

        q_proc = q[:, :]
        baseline_proc = baseline[:, :]

        pcm = ax.pcolormesh(q_proc, cmap=cmap, vmin=-clims, vmax=clims)
        
        thr_up = np.nanpercentile(baseline_proc, 75)
        thr_low = np.nanpercentile(baseline_proc, 25)

        mask_up = q_proc >= thr_up
        mask_low = q_proc <= thr_low

        ax.contour(mask_up, levels=[0.5], colors='black', linewidths=1, alpha=0.7)
        ax.contour(mask_low, levels=[0.5], colors='red', linewidths=1, alpha=0.7)

        ax.hlines(peak, 0, q.shape[1], color='magenta', linestyle='--')
        ax.scatter(pos_peak[1], pos_peak[0], color='white', edgecolor='black', s=80)
        ax.scatter(neg_peak[1], neg_peak[0], color='crimson', edgecolor='black', s=80)

        ax.set_xlabel('Time')
        ax.set_ylabel('Spatial Pos')
        
        return pcm

    def plot_maps_with_peaks(self, dirs=None, map_idxs=None, time_keys=None, trial_idx=0, 
                           cols=3, figsize_per_plot=(5, 5), cmap='viridis', 
                           save_path='peakcompute_OUT_class.png', dpi=150):
        """
        Plot maps with peaks for all sessions and conditions with a shared colorbar
        
        Parameters:
        dirs: list of directions (default: [-1, 1])
        map_idxs: list of map indices (default: [3, 2])
        time_keys: list of time keys (default: [0.5, 1])
        trial_idx: trial index to plot (default: 0)
        cols: number of columns in subplot grid (default: 3)
        figsize_per_plot: size of each subplot (default: (5, 5))
        square_wind: square window size for peak finding (default: 5)
        end_col: end column for peak finding (default: 15)
        cmap: colormap to use (default: 'viridis')
        save_path: path to save the figure (default: 'peakcompute_OUT_class.png')
        dpi: DPI for saved figure (default: 150)
        
        Returns:
        fig, axs: matplotlib figure and axes objects
        """
        if dirs is None:
            dirs = [-1, 1]
        if map_idxs is None:
            map_idxs = [3, 2]
        if time_keys is None:
            time_keys = [0.5, 1]
        
        plot_idx = 0
        sess_names = list(self.results_dict.keys())
        total_plots = len(sess_names) * len(dirs) * len(map_idxs) * len(time_keys) * 2  # *2 for prediction and subtraction
        rows = math.ceil(total_plots / cols)

        # Create figure with space for colorbar
        fig, axs = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows))
        axs = axs.flatten() if total_plots > 1 else [axs]
        
        # Collect all data to determine global color limits
        all_matrices = []
        valid_plots = []
        
        # First pass: collect all matrices and valid plot information
        for sess_name in sess_names:
            for map_type in ['prediction', 'subtraction']:
                for dir_ in dirs:
                    for map_idx in map_idxs:
                        for time_key in time_keys:
                            try:
                                data = self.results_dict[sess_name][map_type]
                                matrix = data.matrices[map_idx][time_key][dir_][trial_idx]
                                baseline = data.baselines[map_idx][time_key][dir_][trial_idx]
                                peak = self.results_dict[sess_name][map_type].peaks[map_idx][time_key][dir_][trial_idx]
                                pos_peak =  self.results_dict[sess_name][map_type].pos_peaks[map_idx][time_key][dir_][trial_idx]
                                neg_peak =  self.results_dict[sess_name][map_type].neg_peaks[map_idx][time_key][dir_][trial_idx]
                                
                                all_matrices.append(matrix)
                                valid_plots.append({'sess_name': sess_name,
                                                    'map_type': map_type,
                                                    'dir_': dir_,
                                                    'map_idx': map_idx,
                                                    'time_key': time_key,
                                                    'matrix': matrix,
                                                    'baseline': baseline,
                                                    'peak': peak,
                                                    'pos_peak': pos_peak[0],
                                                    'neg_peak': neg_peak[0]})
                                
                            except Exception as e:
                                # print(f"Skipping {sess_name} - {map_type} - dir:{dir_} - n_pos:{map_idx} - isi:{time_key} due to error: {e}")
                                continue
        
        # Calculate global color limits
        if all_matrices:
            global_clim = np.nanpercentile(np.concatenate([m.flatten() for m in all_matrices]), 95)
        else:
            global_clim = 0.1
            
        # Second pass: create plots with consistent color scale
        for plot_info in valid_plots:
            if plot_idx >= len(axs):
                break
                
            ax = axs[plot_idx]
            
            pcm = self.visualize_map_with_peaks_fullportion_ax(plot_info['matrix'], plot_info['baseline'], plot_info['peak'], 
                                                               plot_info['dir_'], plot_info['pos_peak'], plot_info['neg_peak'], 
                                                               ax, clims = global_clim, cmap=cmap)
            
            # Create title
            name_title = plot_info['sess_name'].split('VSDI')[1].split('-001')[0] if 'VSDI' in plot_info['sess_name'] else plot_info['sess_name']
            ax.set_title(f"{name_title} - {plot_info['map_type']} - dir:{plot_info['dir_']} - n_pos:{plot_info['map_idx']} - isi:{plot_info['time_key']}")
            plot_idx += 1

        # Hide unused axes
        for j in range(plot_idx, len(axs)):
            fig.delaxes(axs[j])

        # Add colorbar with proper positioning
        if plot_idx > 0:
            # Adjust subplot parameters to make room for colorbar
            plt.subplots_adjust(right=0.85)
            
            # Create colorbar on the right side
            cbar_ax = fig.add_axes([1.05, 0.6, 0.03, 0.35])  # [left, bottom, width, height]
            cbar = fig.colorbar(pcm, cax=cbar_ax)
            cbar.set_label('Signal Intensity', rotation=270, labelpad=50)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return fig, axs
