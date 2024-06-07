import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import time
import os
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from functools import partial

from blueetl.constants import *
from blueetl.parallel import call_by_simulation


import cortex_etl as c_etl



def plot_simulation_raster(simulation_row, filtered_dataframes, analysis_config, raster_option_combination):
        
    plot_raster(simulation_row, 
                        filtered_dataframes['windows'].iloc[0], 
                        filtered_dataframes['spikes'], 
                        filtered_dataframes['neurons'], 
                        filtered_dataframes['neuron_classes'], 
                        raster_option_combination,
                        analysis_config,
                        simulation_histograms=filtered_dataframes['histograms'])


def plot_rasters(a, custom_file_path=None, simulation_filter={}):

    a.repo.simulations.df["SummaryPNG"] = a.repo.simulations.df['rasters_dir'] / (a.repo.simulations.df['simulation_string'].astype(str) + "_SUMMARY.png")

    for dict_to_unpack in a.analysis_config.custom['raster_windows']:

        window_str = list(dict_to_unpack.keys())[0]
        fig_dims = list(dict_to_unpack.values())[0]

        if (type(fig_dims) == list):
            fig_width = fig_dims[0]
            fig_height = fig_dims[1]
        else:
            fig_width = fig_dims
            fig_height = fig_width * 0.6
        
        raster_option_combinations = [
                                    RasterOptions(a, window_str=window_str, neuron_group_y_axis_equal=False, use_spikes=True, smoothing_type='Gaussian', hist_bin_size=3.0, kernel_sd=1.0, neuron_classes=c_etl.LAYER_EI_NEURON_CLASSES, neuron_class_groupings=c_etl.NEURON_CLASS_NO_GROUPINGS, fig_width=fig_width, fig_height=fig_height, custom_file_path=custom_file_path),
                                    RasterOptions(a, window_str=window_str, neuron_group_y_axis_equal=True, use_spikes=False, smoothing_type='Gaussian', hist_bin_size=3.0, kernel_sd=1.0, neuron_classes=['ALL'], neuron_class_groupings=[['ALL']], extra_string='All', fig_width=fig_width, fig_height=fig_height, custom_file_path=custom_file_path),
                                    RasterOptions(a, window_str=window_str, neuron_group_y_axis_equal=True, use_spikes=False, smoothing_type='Gaussian', hist_bin_size=3.0, kernel_sd=1.0, neuron_classes=c_etl.LAYER_EI_NEURON_CLASSES + ['ALL_EXC', 'ALL_INH'], neuron_class_groupings=c_etl.LAYER_EI_NEURON_CLASS_GROUPINGS, extra_string='LayerEI', fig_width=fig_width, fig_height=fig_height, custom_file_path=custom_file_path),
                                    RasterOptions(a, window_str=window_str, neuron_group_y_axis_equal=True, use_spikes=False, smoothing_type='Gaussian', hist_bin_size=5.0, kernel_sd=1.0, neuron_classes=c_etl.LAYER_EI_NEURON_CLASSES + ['ALL_EXC', 'ALL_INH'], neuron_class_groupings=c_etl.LAYER_EI_NEURON_CLASS_GROUPINGS, extra_string='LayerEI', fig_width=fig_width, fig_height=fig_height, custom_file_path=custom_file_path, lw=0.4, seperator_lw=0.2)
                                    ]

        if (a.analysis_config.custom['plot_rasters']):

            print(f"\n----- Plot rasters, window: {window_str} -----")

            for roc in raster_option_combinations:

                dataframes={
                        "spikes": a.repo.report.df.etl.q(neuron_class=roc.neuron_classes, window=roc.window_str, trial=0),
                        "windows": a.repo.windows.df.etl.q(window=roc.window_str), 
                        "neurons": a.repo.neurons.df,
                        "neuron_classes": a.repo.neuron_classes.df.etl.q(neuron_class=roc.neuron_classes),
                        "histograms": a.features.histograms.df.etl.q(neuron_class=roc.neuron_classes, window=roc.window_str, bin_size=roc.hist_bin_size, smoothing_type=roc.smoothing_type, kernel_sd=roc.kernel_sd)
                        }
            
                results = call_by_simulation(a.repo.simulations.df.etl.q(simulation_filter), 
                                                dataframes, 
                                                func=partial(plot_simulation_raster, analysis_config=a.analysis_config.custom, raster_option_combination=roc), how="series")

        if (a.analysis_config.custom['create_raster_videos']):

            print(f"\n----- Create raster videos, window: {window_str} -----")

            for roc in raster_option_combinations:
                if (a.repo.windows.df.etl.q(window=window_str).iloc[0]['window_type'] == "spontaneous"):
                    for mask_key in ['', 'bursting', 'bursting_or_fr_gt_threshold_or_ei_corr_r_out_of_range']:
                        roc.create_video(a, mask_key=mask_key)

                elif (a.repo.windows.df.etl.q(window=window_str).iloc[0]['window_type'] == "evoked_stimulus_onset_zeroed"):
                    for mask_key_and_invert_mask_bool in [['', False], ['overly_sustained_response', False], ['overly_sustained_response', True], ['higher_secondary_peak', False], ['higher_secondary_peak', True], ['too_much_trial_to_trial_variance', False], ['too_much_trial_to_trial_variance', True], ['evoked_mask', False], ['evoked_mask', True]]:
                        roc.create_video(a, mask_key=mask_key_and_invert_mask_bool[0], invert_mask=mask_key_and_invert_mask_bool[1])




class RasterOptions(object):

    def __init__(self, a, window_str='', neuron_group_y_axis_equal=True, use_spikes=True, smoothing_type='', hist_bin_size=3.0, kernel_sd=1.0, neuron_classes=[], neuron_class_groupings=[], extra_string='', fig_width=20, fig_height=15, custom_file_path=None, lw=0.2, seperator_lw=0.2):

        self.window_str = window_str
        self.neuron_group_y_axis_equal = neuron_group_y_axis_equal
        self.use_spikes = use_spikes
        self.smoothing_type = smoothing_type
        self.hist_bin_size = hist_bin_size
        self.kernel_sd = kernel_sd
        self.neuron_classes = neuron_classes
        self.neuron_class_groupings = neuron_class_groupings
        self.extra_string = extra_string
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.lw = lw
        self.seperator_lw = seperator_lw

        
        windows = pd.merge(a.repo.simulations.df.reset_index().drop(["index"], axis=1), a.repo.windows.df.reset_index()).set_index(['index'])
        if (custom_file_path == None):
            custom_file_path = windows['rasters_dir'].astype(str)

        self.options_str = "{spikes}".format(spikes="S_" if use_spikes == True else "NS_") + smoothing_type + "_" + str(hist_bin_size) + "_" + str(kernel_sd) + "_" + "{yax}".format(yax="YNE_" if neuron_group_y_axis_equal == True else "YE_") + extra_string
        self.df_file_path_key = self.options_str + '_rasters_path_png'
        self.df_file_path_pdf_key = self.options_str + '_rasters_path_pdf'

        a.repo.windows.df.loc[windows.index, self.df_file_path_key] = custom_file_path + (windows['window'].astype(str) + "_" + self.options_str + "_RASTER.png")
        a.repo.windows.df.loc[windows.index, self.df_file_path_pdf_key] = custom_file_path+ (windows['window'].astype(str) + "_" + self.options_str + "_RASTER.pdf")


    def create_video(self, a, mask_key='', invert_mask=False):

        windows_df = a.repo.windows.df.etl.q(window=self.window_str)
        
        if (mask_key != ''):
            windows_with_stats_df = pd.merge(windows_df, a.custom['by_simulation'])
            q = {mask_key: invert_mask}
            windows_df = windows_with_stats_df.etl.q(q)

        raster_videos_window_dir = str(a.figpaths.raster_videos) + "/" + self.window_str + "/" + mask_key + str(invert_mask) + "/"
        os.makedirs(raster_videos_window_dir, exist_ok=True)

        video_fn = raster_videos_window_dir + self.window_str + "_" + self.options_str + '_' + mask_key + ":" + str(invert_mask) + ".mp4"

        c_etl.video_from_image_files(windows_df[self.df_file_path_key].astype(str).tolist(), video_fn)


def renormalise_psth(psth):
    new_hist = psth  - np.min(psth)
    new_hist = new_hist / np.max(new_hist)
    return new_hist

def plot_raster(simulation_row, window_row, window_spikes, circuit_neurons, neuron_classes, raster_option_combination, analysis_config, simulation_histograms=None, spont_ei_corr_rval=-5.0):

    sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3,
            "axes.titlesize": 7, "axes.spines.right": False, "axes.spines.top": False})

    start_time = time.time()
    plt.figure(figsize=(raster_option_combination.fig_width, raster_option_combination.fig_height))
    ax = plt.gca()


    # SET NEURON CLASS COLOURS 
    neuron_classes = neuron_classes.copy()
    neuron_classes.loc[:, 'c'] = neuron_classes.apply(lambda row : c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[row['neuron_class']]["color"], axis=1)

    # SET NEURON CLASS START 
    if (not raster_option_combination.neuron_group_y_axis_equal):
        neuron_classes['cum_sum'] = neuron_classes[COUNT].cumsum()
        neuron_classes['start_pos'] = neuron_classes['cum_sum'].shift().fillna(0)
    else:
        neuron_classes['start_pos'] = 0
        neuron_classes['cum_sum'] = 0
        for neuron_class_index, neuron_class in neuron_classes.iterrows():            
            for neuron_class_grouping_index, neuron_class_grouping in enumerate(raster_option_combination.neuron_class_groupings):
                if neuron_class["neuron_class"] in neuron_class_grouping:
                    neuron_classes.loc[neuron_class_index, 'start_pos'] = neuron_class_grouping_index * 5000
                    neuron_classes.loc[neuron_class_index, 'cum_sum'] = neuron_class_grouping_index * 5000 + 5000


    # PLOT SPIKES
    if (raster_option_combination.use_spikes):
        window_spikes = pd.merge(neuron_classes, window_spikes)
        window_spikes = window_spikes.set_index([CIRCUIT_ID, NEURON_CLASS, GID])
        circuit_neurons = circuit_neurons.set_index([CIRCUIT_ID, NEURON_CLASS, GID])
        window_spikes = circuit_neurons.join(window_spikes, how='inner')

        shuffled_within_neuron_class = True
        if (shuffled_within_neuron_class):
            for neuron_class_index, neuron_class in neuron_classes.iterrows(): 
                nc_w_spikes = window_spikes.etl.q(neuron_class=neuron_class[NEURON_CLASS])
                nc_random_map = np.arange(neuron_class[COUNT])
                np.random.shuffle(nc_random_map)
                shuffled_neuron_class_indices = nc_w_spikes.neuron_class_index.map(lambda x: nc_random_map[x])
                neuron_scatter_pos = nc_w_spikes['start_pos'] + shuffled_neuron_class_indices
                ax.scatter(nc_w_spikes[TIME], neuron_scatter_pos, s=0.1, c=nc_w_spikes['c'], linewidths=0) #, facecolors='c', s=0.2

        else:
            neuron_scatter_pos = window_spikes['start_pos'] + window_spikes['neuron_class_index']
            ax.scatter(window_spikes[TIME], neuron_scatter_pos, s=0.1, c=window_spikes['c'], linewidths=0)



    # OPTIONALLY LOAD INVIVO_HISTOGRAMS
    if (window_row['window_type'] in ['evoked_stimulus_onset_zeroed', 'evoked_cortical_onset_zeroed']):
        vivo_df = pd.read_feather(analysis_config['vivo_df']).reset_index()
        vivo_neuron_classes = vivo_df["neuron_class"].unique()

    # PLOT DIVIDERS AND HISTOGRAMS
    for neuron_class_index, neuron_class in neuron_classes.iterrows():
        plt.plot([window_row['t_start'], window_row['t_stop']], [neuron_class['start_pos'], neuron_class['start_pos']], lw=raster_option_combination.seperator_lw, c='k')

        if (simulation_histograms is not None):

            bin_indices, hist_array = c_etl.hist_elements(simulation_histograms.etl.q(simulation_id=window_row[SIMULATION_ID], 
                                                                                        neuron_class=neuron_class[NEURON_CLASS], 
                                                                                        window=raster_option_combination.window_str, 
                                                                                        bin_size=raster_option_combination.hist_bin_size, 
                                                                                        smoothing_type=raster_option_combination.smoothing_type, 
                                                                                        kernel_sd=raster_option_combination.kernel_sd))


            if (hist_array.shape[0] != 0):
                hist_max = np.max(hist_array)
                if (hist_max != 0.0):
                    max_normalised_hist = hist_array / np.max(hist_array)

                    if (not raster_option_combination.neuron_group_y_axis_equal):
                        plt.plot(window_row['t_start'] + (bin_indices * raster_option_combination.hist_bin_size), neuron_class['cum_sum'] - neuron_class[COUNT]*max_normalised_hist, c=neuron_class['c'], lw=raster_option_combination.lw)
                    else:
                        plt.plot(window_row['t_start'] + (bin_indices * raster_option_combination.hist_bin_size), neuron_class['cum_sum'] - 5000*(max_normalised_hist), c=neuron_class['c'], lw=raster_option_combination.lw)

                    if (window_row['window_type'] in ['evoked_stimulus_onset_zeroed', 'evoked_cortical_onset_zeroed']):
                        if (neuron_class[NEURON_CLASS] in list(c_etl.vivo_neuron_class_map.keys())):
                            in_vivo_neuron_class = c_etl.vivo_neuron_class_map[neuron_class[NEURON_CLASS]]
                            if in_vivo_neuron_class in vivo_neuron_classes:
                                nc_data = vivo_df[(vivo_df["neuron_class"] == in_vivo_neuron_class) & (vivo_df["barrel"] == "C2")].iloc[0]
                                nc_mean = nc_data["psth_mean"]
                                # nc_sd = nc_data["psth_sd"]

                                x = window_row['t_start'] - 50.0 + (1.0 * np.asarray(range(len(nc_mean))))

                                if (not raster_option_combination.neuron_group_y_axis_equal):
                                    y = neuron_class['cum_sum'] - neuron_class[COUNT]*nc_mean
                                else:
                                    y = neuron_class['cum_sum'] - 5000*(nc_mean)


                                plt.plot(x, y, c=neuron_class['c'], lw=raster_option_combination.lw, linestyle='--')




    # PLOT OPTIONS AND SAVE
    x_tick_distance = 5
    duration = window_row['t_stop'] - window_row['t_start']
    if (duration > 50):
        x_tick_distance = 10
    if (duration > 100):
        x_tick_distance = 100
    if (duration > 1000):
        x_tick_distance = 1000

    ax.set_yticks(neuron_classes['start_pos'] + (neuron_classes[COUNT]/2.0), minor=False)
    if (raster_option_combination.use_spikes):
        ax.set_yticklabels([c_etl.neuron_class_label_map[nc] for nc in neuron_classes['neuron_class']], minor=False)
    else:
        ax.set_yticks(neuron_classes['start_pos'] + (5000.0/2.0), minor=False)
#         print([nc_str.split('_')[0] for nc_str in neuron_classes['neuron_class']])
        ax.set_yticklabels([nc_str.split('_')[0] for nc_str in neuron_classes['neuron_class']], minor=False)

    ax.set_xlim([window_row['t_start'], window_row['t_stop']])
    ax.set_ylim([0, neuron_classes['cum_sum'].max()])
    ax.set_ylabel('')
    # ax.set_xlabel('Time from window start (ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_axisbelow(True)
    title_str = str(simulation_row['simulation_id']) + " " + simulation_row['simulation_string'] #+ "    " + raster_option_combination.options_str
    if (spont_ei_corr_rval != -5.0):
        title_str += "  spont_ei_corr_rval: " + str(np.around(spont_ei_corr_rval, decimals=3))
    ax.set_title(title_str)

    ax.xaxis.set_major_locator(MultipleLocator(x_tick_distance))
    ax.xaxis.set_minor_locator(MultipleLocator(x_tick_distance))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.invert_yaxis()

    plt.savefig(window_row[raster_option_combination.df_file_path_key], bbox_inches='tight', dpi=600)
    plt.savefig(window_row[raster_option_combination.df_file_path_pdf_key], bbox_inches='tight')
    plt.close()

    print("Raster generated: ", "{:.2f}".format(time.time() - start_time), 's')