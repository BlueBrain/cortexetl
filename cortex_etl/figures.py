import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import linregress


import seaborn as sns
from matplotlib import colors
from matplotlib import cm

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle

from itertools import chain

import cortex_etl as c_etl


############################################################################ 
### Heatmap + lineplot
############################################################################ 
def lineplot_settings(analysis_key):
    cmap_name = 'PiYG'
    cmap_buffer = 0
    if (analysis_key in ['ei_corr_rval', 'std_of_neuron_class_proportions_of_vivo', 'cv_of_neuron_class_proportions_of_vivo', 'euc_dist_to_desired_proportion_of_in_vivo_FRs', 'euc_dist_to_scaled_in_vivo_FRs', 'difference_between_mean_proportion_and_target_proportion']):
        cmap_name = 'cividis'
        cmap_buffer = 0

    return cmap_name, cmap_buffer

    return r_dict
import pandas as pd

def heatmap_settings(analysis_key):

    hm_label = analysis_key
    cmap = sns.cm.rocket
    cmap_center = None
    fixed_vmin = []
    fixed_vmax = []
    additional_options = {}
    reverse_vertical = False
    annot=True
    square=True
    lw=2.0

    if (analysis_key in ["ei_corr_rval", "ei_corr_shuffled_rval"]):
        hm_label = 'E/I correlation r-value'
#         cmap = sns.cm.vlag
        cmap = cm.get_cmap('cividis')
        cmap_center = 0.25
#         fixed_vmin = [-1.0]
        fixed_vmin = [0.0]
        fixed_vmax = [0.7]

    if (analysis_key in ["25_decay_diff_rp", "50_decay_diff_rp"]):
        cmap = sns.cm.vlag
        cmap_center = 0.0
        fixed_vmin = [-1.0]
        fixed_vmax = [1.0]

    if (analysis_key in ["mean_ratio_difference"]):
        cmap = sns.color_palette("seismic", as_cmap=True)
        cmap_center = 0.0

    if (analysis_key in ["30ms_decay_point", "60ms_decay_point"]):
        hm_label = 'Decay fraction'
        cmap = cm.get_cmap('winter')

    if (analysis_key in ["log_euc_dist_to_vivo_evoked_ratios"]):
        cmap = sns.cm.vlag

    if (analysis_key in ['latency_silico', 'latency_diff', 'decay_silico', 'decay_diff', '100pDiffToRP', '75pDiffToRP', '50pDiffToRP', '25pDiffToRP']):
        cmap = cm.get_cmap('winter')

    if (analysis_key in ["mean_pairwise_first_spike_r_value"]):
        hm_label = 'mean_pairwise_first_spike_r_value'
        cmap = sns.cm.vlag
        cmap_center = 0.0
        fixed_vmin = [-1.0]
        fixed_vmax = [1.0]

    if (analysis_key == "simulation_id"):
        hm_label = 'Simulation ID'
        additional_options = {'fmt':'g'}

    if (analysis_key == "power"):
        hm_label = 'FFT Power'
        cmap = cm.get_cmap('Greys')
        reverse_vertical = True
        fixed_vmin = [0.0]
        annot=False
        square=False
        lw=0.0

    if (analysis_key == "log_power"):
        hm_label = 'FFT Log Power'
        cmap = cm.get_cmap('Greys_r')
        reverse_vertical = False
        fixed_vmin = [0.0]
        annot=False
        square=False
        lw=0.0

    if (analysis_key == "euc_dist_to_desired_proportion_of_in_vivo_FRs"):
        hm_label = 'Euc dist to\ntarget $P_{FR}$'
        
        
    if (analysis_key == "euc_dist_to_scaled_in_vivo_FRs"):
        hm_label = 'Euc dist to\ntarget FRs'

    if (analysis_key == "desired_connected_proportion_of_invivo_frs"):
        hm_label = '$P_{FR}$'

    if (analysis_key == "vpm_pct"):
        hm_label = '$F_{P}$'

    d = {
        'cmap': cmap,
        'cmap_center': cmap_center,
        'hm_label': hm_label,
        'fixed_vmin': fixed_vmin,
        'fixed_vmax': fixed_vmax,
        'additional_options': additional_options,
        'reverse_vertical': reverse_vertical,
        'annot': annot,
        'square': square,
        'lw': lw
    }

    return d




def unique_vals_in_df(df, key, sort=True, sort_reverse=False):

    if (key != 'none'):
        unique_vals = df[key].unique()
        if sort_reverse:
            return np.flip(np.sort(unique_vals))
        elif sort:
            return np.sort(unique_vals)
        return unique_vals
    else:
        return [None]


def label_for_key(key, val=None):

    label = key

    if (key in c_etl.parameter_constants.keys()):

        label = c_etl.parameter_constants[key]['axis_label']

        if (val is not None):
            label += ": " + str(val)

        if (c_etl.parameter_constants[key]['unit_string'] != ''):
            label += " (" + c_etl.parameter_constants[key]['unit_string'] + ")"

    else:
        label = key + ": " + str(val)


    return label



def set_hor_and_ver_labels_for_axis(ax, ver_i, hor_i, uniq_ver, uniq_hor, x_key, y_key, hor_val, ver_val, hor_key, ver_key):

    if (ver_i == len(uniq_ver) - 1):
        ax.set_xlabel(label_for_key(x_key), fontsize=9)
    if (hor_i == 0):
        ax.set_ylabel(label_for_key(y_key), fontsize=9)
    if (hor_i == len(uniq_hor) - 1):
        if (ver_val != 'none'):
            ax.annotate(label_for_key(ver_key, ver_val), xy=(0, 0), xytext=(1.03, 0.5), xycoords=ax.yaxis.label, textcoords='axes fraction', size=9, ha='left', va='center', rotation=90)
    if (ver_i == 0):
        if (hor_val != 'none'):
            ax.annotate(label_for_key(hor_key, hor_val), xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction', textcoords='offset points', size=9, ha='center', va='baseline')




import matplotlib
from matplotlib import colors
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
def heatmap(hms_df, stat_key, file_name, hor_key="none", ver_key="none", x_key="none", y_key="none", mask_key="", highlight_false_key="", figsize=(), show_colorbar=False, override_cmap=''):

    print("Create heatmap: " + stat_key)
    
    hm_keys = [hor_key, ver_key, x_key, y_key, stat_key]
    if ("none" in hm_keys):
        hm_keys = hm_keys.remove("none")

    vertical_reverse = False
    if (stat_key in ['power', 'log_power']):
        vertical_reverse = True

    uniq_hor = c_etl.unique_vals_in_df(hms_df, hor_key)
    uniq_ver = c_etl.unique_vals_in_df(hms_df, ver_key, sort_reverse=vertical_reverse)

    uniq_x = c_etl.unique_vals_in_df(hms_df, x_key)
    uniq_y = c_etl.unique_vals_in_df(hms_df, y_key)

    masked_hms_df = hms_df
    if (mask_key != ''):
        mask_dict = {mask_key:False}
        masked_hms_df = hms_df.etl.q(mask_dict)

    if (len(figsize) == 0):
        figsize = (4*len(uniq_hor), 4.0*len(uniq_ver))
        
    fig, axes = plt.subplots(ncols=len(uniq_hor), nrows=len(uniq_ver), sharex=True, sharey=True, figsize=figsize)

    if ((len(uniq_ver) == 1) & (len(uniq_hor) == 1)):
        axes = [[axes]]
    if ((len(uniq_ver) == 1) & (len(uniq_hor) > 1)):
        axes = [axes]
    if ((len(uniq_ver) > 1) & (len(uniq_hor) == 1)):
        axes = [[ax] for ax in axes]

    d = heatmap_settings(stat_key)

    vmin = masked_hms_df[stat_key].min()
    vmax = masked_hms_df[stat_key].max()
    
    if len(d['fixed_vmin']):
        vmin = d['fixed_vmin'][0]
    if len(d['fixed_vmax']):
        vmax = d['fixed_vmax'][0]

    for hor_i, hor_val in enumerate(uniq_hor):
        for ver_i, ver_val in enumerate(uniq_ver):

            ax = axes[ver_i][hor_i]

            q_dict = {}
            if (hor_val is not None):
                q_dict[hor_key] = hor_val

            if (ver_val is not None):
                q_dict[ver_key] = ver_val

            hm_df = hms_df.etl.q(q_dict)

            hm = []
            hm_mask = []
            hm_highlight = []
            for x_i, x_val in enumerate(uniq_x):
                sub_l = []
                mask_sub_l = []
                for y_i, y_val in enumerate(uniq_y):

                    q_dict = {}
                    if (x_val is not None):
                        q_dict[x_key] = x_val

                    if (y_val is not None):
                        q_dict[y_key] = y_val

                    vals_list = hm_df.etl.q(q_dict)[stat_key].tolist()
                    if (len(vals_list) > 0):
                        if (len(vals_list) > 1):
                            print("Warning: More than 1 value found for query")
                        sub_l.append(vals_list[0])
                        if (mask_key != ""):
                            mask_val = hm_df.etl.q(q_dict)[mask_key].tolist()[0]
                            mask_sub_l.append(mask_val)
                        else:
                            mask_sub_l.append(False)

                        if ((highlight_false_key != "") and (not hm_df.etl.q(q_dict)[highlight_false_key].tolist()[0])):
                            ax.add_patch(Rectangle((x_i+.02,y_i+.02),.96,.96, fill=False, edgecolor='g', facecolor='None', lw=d['lw'], clip_on=True, zorder=10))

                    else:
                        sub_l.append(-1)
                        mask_sub_l.append(True)

                hm.append(sub_l)
                hm_mask.append(mask_sub_l)

            hm = np.asarray(hm).T
            hm_mask = np.asarray(hm_mask).T
            
            cmap = override_cmap
            if (override_cmap==''):
                cmap=d['cmap']
            sns.heatmap(hm, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=d['cmap_center'], cbar=False, square=d['square'], xticklabels=uniq_x, yticklabels=uniq_y, annot=d['annot'], annot_kws={"fontsize":5}, mask=hm_mask, linewidths=0.0, linecolor=(0.0, 0.0, 0.0, 0.0), **d['additional_options'])
            ax.set_yticklabels(uniq_y, rotation=0)

            if (stat_key in ['power']):
                
                max_x = round(np.max(uniq_x))
                ax.set_xlim([0.0, len(uniq_x)/2])
                ax.xaxis.set_major_locator(MultipleLocator(len(uniq_x)/2))
#                 ax.xaxis.set_minor_locator(MultipleLocator(len(uniq_x) / 20.0))
                ax.set_xticks([0.0, 50.0], labels=["0", "10"], rotation=0)
                ax.set_yticks([0, 9], labels=["0", "1"], rotation=0)
                ax.tick_params(axis=u'both', which=u'both',length=0)

            ax.invert_yaxis()
            ax.set_facecolor("lightgrey")

            c_etl.set_hor_and_ver_labels_for_axis(ax, ver_i, hor_i, uniq_ver, uniq_hor, x_key, y_key, hor_val, ver_val, hor_key, ver_key)

#             for tick in ax.xaxis.get_major_ticks():
#                 tick.label.set_fontsize(9)

#             for tick in ax.yaxis.get_major_ticks():
#                 tick.label.set_fontsize(9)

    if (show_colorbar):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if d['cmap_center'] != None:
            lim = max(abs(vmin), abs(vmax))
            norm = colors.Normalize(vmin=-lim, vmax=lim)
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=d['cmap']), ax=axes, label=d['hm_label'], location='bottom', pad=0.4)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(d['hm_label'], fontsize=9)

    if (mask_key != ''):
        file_name += '_M-' + mask_key

    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


import matplotlib
from matplotlib import cm
import numpy as np
def lineplot(hms_df, stat_key, file_name, hor_key="none", ver_key="none", x_key="none", colour_var_key="none", linestyle_key="none", mask_key="", highlight_false_key="", marker='x', figsize=(), custom_ylim=[], remove_intermediate_labels=True, autosize_scaling_factor=4): #, also_without_values=False

    print("Create lineplot: " + stat_key)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    
    
    hm_keys = [hor_key, ver_key, x_key, colour_var_key, linestyle_key, stat_key]
    if "none" in hm_keys:
        hm_keys.remove("none")

    if (mask_key != ''):
        mask_dict = {mask_key:False}
        hms_df = hms_df.etl.q(mask_dict)

    if (len(hms_df) > 0):

        uniq_hor = c_etl.unique_vals_in_df(hms_df, hor_key)
        uniq_ver = c_etl.unique_vals_in_df(hms_df, ver_key)
        uniq_x = c_etl.unique_vals_in_df(hms_df, x_key)
        uniq_colour_vars = c_etl.unique_vals_in_df(hms_df, colour_var_key, sort=False)
        unique_linestyles = c_etl.unique_vals_in_df(hms_df, linestyle_key)

        vmin = hms_df[stat_key].min()
        vmax = hms_df[stat_key].max()
        
        if (len(figsize) == 0):
            figsize = (autosize_scaling_factor*len(uniq_hor), autosize_scaling_factor*1.2*len(uniq_ver))

        fig, axes = plt.subplots(ncols=len(uniq_hor), nrows=len(uniq_ver), sharex=True, sharey=True, figsize=figsize)

        if ((len(uniq_ver) == 1) & (len(uniq_hor) == 1)):
            axes = [[axes]]
        if ((len(uniq_ver) == 1) & (len(uniq_hor) > 1)):
            axes = [axes]
        if ((len(uniq_ver) > 1) & (len(uniq_hor) == 1)):
            axes = [[ax] for ax in axes]

        cmap_name, cmap_buffer = c_etl.lineplot_settings(stat_key)
        cmap = cm.get_cmap(cmap_name, len(uniq_colour_vars) + 1 + cmap_buffer)  # n discrete colors

        max_y_vals = []
        min_y_vals = []

        for hor_i, hor_val in enumerate(uniq_hor):
            for ver_i, ver_val in enumerate(uniq_ver):

                ax = axes[ver_i][hor_i]

                q_dict = {}
                if (hor_val is not None):
                    q_dict[hor_key] = hor_val

                if (ver_val is not None):
                    q_dict[ver_key] = ver_val

                hm_df = hms_df.etl.q(q_dict)

                hm = []
                hm_mask = []
                hm_highlight = []
                
                for linestyle_var_i, linestyle_var_val in enumerate(unique_linestyles):
                    
                    q_dict = {}
                    if (linestyle_var_val is not None):
                        q_dict[linestyle_key] = linestyle_var_val
                    ls_df = hm_df.etl.q(q_dict)

                    for colour_var_i, colour_var_val in enumerate(uniq_colour_vars):

                        xvals = []
                        sub_l = []

                        for x_i, x_val in enumerate(uniq_x):
                            
                            mask_sub_l = []

                            q_dict = {}
                            if (x_val is not None):
                                q_dict[x_key] = x_val

                            if (colour_var_val is not None):
                                q_dict[colour_var_key] = colour_var_val

                            vals_list = ls_df.etl.q(q_dict)[stat_key].tolist()
                            if (len(vals_list) > 0):
                                sub_l.append(vals_list[0])
                                xvals.append(x_val)
                                max_y_vals.append(np.max(sub_l))
                                min_y_vals.append(np.min(sub_l))
                        
                        ax.plot(xvals, sub_l, color=cmap(colour_var_i + 1), lw=0.5, markersize=3, marker='x', ls=linestyles[linestyle_var_i % len(linestyles)], label=str(colour_var_val))
                        ax.set_xticks(xvals)
                
                if ((hor_i==0) & (ver_i == 0)):
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper center')

                c_etl.set_hor_and_ver_labels_for_axis(ax, ver_i, hor_i, uniq_ver, uniq_hor, x_key, stat_key, hor_val, ver_val, hor_key, ver_key)
            
                ax.set_xlabel(c_etl.heatmap_settings(x_key)['hm_label'], labelpad=-3)
                if (hor_i == 0):
                    ax.set_ylabel(c_etl.heatmap_settings(stat_key)['hm_label'], labelpad=-5)
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)
                
                if ((len(custom_ylim) == 0) and (len(max_y_vals))):
                    max_val = np.max(max_y_vals)
                    if (np.max(max_y_vals) > 0.0):
                        max_val = 1.0
                    ax.set_ylim([np.min([np.min(min_y_vals), 0.0]), np.max([np.max(max_y_vals), max_val])])
                elif len(max_y_vals):
                    # print(custom_ylim)
                    ax.set_ylim(custom_ylim)
                
                if remove_intermediate_labels:
                    c_etl.remove_intermediate_labels(ax.xaxis.get_major_ticks())    
                    c_etl.remove_intermediate_labels(ax.yaxis.get_major_ticks())    

            
        # print(file_name + '_' + mask_key + '.pdf')
        plt.savefig(file_name + '_' + mask_key + '.pdf', bbox_inches='tight')
        plt.close()


############################################################################ 
### Compare two conditons by neuron class
############################################################################ 
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import cm
import numpy as np
def plot_two_conditions_comparison_by_neuron_class(df1, df2, stat_key_1, stat_key_2, colour_key, comparison_pair, fig_path, fit_and_plot_exponential=False, neuron_classes=''):

    # df1 = df1.etl.q(depol_stdev_mean_ratio=[0.2, 0.3, 0.4, 0.5])
    # df2 = df2.etl.q(depol_stdev_mean_ratio=[0.2, 0.3, 0.4, 0.5])

    fig, axes = plt.subplots(1, len(neuron_classes), figsize=(24, 2))

    unique_cas = df1[colour_key].unique()
    cmap = cm.get_cmap('viridis', len(unique_cas))    # n discrete colors

    slope_dict = {}
    for ax_i, ax, neuron_class in zip(list(range(len(axes))), axes, neuron_classes):
        nc_df1 = df1.etl.q(neuron_class=neuron_class)
        nc_df2 = df2.etl.q(neuron_class=neuron_class)

        limit_filter = False
        if (limit_filter):
            nc_df2["invivo_fr"] = nc_df2['desired_connected_fr'] / nc_df2['desired_connected_proportion_of_invivo_frs']
            where_good = nc_df2['mean_of_mean_firing_rates_per_second'] < nc_df2['invivo_fr'] * 1.05
            nc_df2 = nc_df2[where_good]
            nc_df1 = nc_df1[where_good]

        stat1 = nc_df1[stat_key_1].to_numpy()
        stat2 = nc_df2[stat_key_2].to_numpy()

        if (len(stat1) & len(stat2)):
            if ((not np.all(stat1 == stat1[0])) & (not np.all(stat2 == stat2[0]))):
                slope_dict[c_etl.bluepy_neuron_class_map[neuron_class]] = np.around(linregress(stat1, stat2).slope, 3)


        ca_x_range_maxes = []
        for ca in nc_df1['ca'].unique():

            c = cmap(list(unique_cas).index(ca))

            ca_nc_df1 = nc_df1.etl.q(ca=ca)
            ca_nc_df2 = nc_df2.etl.q(ca=ca)

            ca_stat1 = ca_nc_df1[stat_key_1].to_numpy()
            ca_stat2 = ca_nc_df2[stat_key_2].to_numpy()

            ignore_basline_samples = False
            if (ignore_basline_samples):
                if (neuron_class in ["L23_INH", "L4_INH"]):
                    if (neuron_class == "L23_INH"):
                        baseline = 0.2

                    if (neuron_class == "L4_INH"):
                        baseline = 0.16

                    where_above = ca_stat2 > baseline
                    if (len(ca_stat1[where_above]) > 2):

                        ca_stat1 = ca_stat1[where_above]
                        ca_stat2 = ca_stat2[where_above]



            if (len(ca_stat1) & len(ca_stat2)):

                if ((not np.all(ca_stat1 == ca_stat1[0])) & (not np.all(ca_stat2 == ca_stat2[0]))):

                    ca_lr = linregress(ca_stat1, ca_stat2)
                    ca_lr_slope = np.around(ca_lr.slope, 3)

                    ca_x_range = np.arange(start=0.0, stop=ca_stat1.max(), step=0.005, dtype=float)
                    ca_y_range = ca_lr_slope * ca_x_range + ca_lr.intercept

                    if (fit_and_plot_exponential):
                        if (len(ca_stat1) > 2):
                            expon, expon_error = c_etl.fit_exponential(ca_stat1, ca_stat2)
            
                            step = 0.005
                            ca_x_range = np.arange(start=step, stop=ca_stat1.max() + step, step=step, dtype=float)
                            ca_x_range_maxes.append(np.max(ca_x_range))

                            ca_y_range = expon[0] * np.exp(expon[1] * ca_x_range) + expon[2]
                            ax.plot(ca_x_range, ca_y_range, color=c, lw=.5)



                    ax.set_xlim([0.0, np.max([stat1.max(), stat2.max()])])

            ax.scatter(ca_stat1, ca_stat2, color=c, s=2)    
        
        ax.set_title(neuron_class)

        if (len(stat1) > 0):
            maxes = [stat1.max(), stat2.max()]
            if 'ca_x_range_maxes' in locals():
                if (len(ca_x_range_maxes)):
                    maxes.append(np.max(ca_x_range_maxes))

            lims = [
            0.0,  # min of both axes
            np.max(maxes),  # max of both axes
            ]
            lims[1] = lims[1] * 1.1

            # now plot both limits against eachother
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
            ax.set_aspect('equal', 'box')
            ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            if (lims[0] != lims[1]):
                ax.set_xlim(lims)
                ax.set_ylim(lims)

        ax.set_xlabel(comparison_pair[0] + ' MFR \n(spikes/s)')
        if (ax_i == 0):
            ax.set_ylabel(comparison_pair[1] + ' MFR \n(spikes/s)')

    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return slope_dict

def compare_firing_rates_for_condition_pairs_by_neuron_class(a, custom_by_neuron_class_df, stat_filter, fit_and_plot_exponential=False, neuron_classes=''):

    for comparison_pair in a.analysis_config.custom['fr_comparison_pairs']:

        print(comparison_pair)

        if (len(comparison_pair) == 3):
            fit_and_plot_exponential = comparison_pair[2]

        stat_key_1 = 'mean_of_mean_firing_rates_per_second'; stat_key_2 = 'mean_of_mean_firing_rates_per_second'
        window_key_1 = comparison_pair[0]; window_key_2 = comparison_pair[1]

        window_key_placeholder = "conn_spont"

        prespecified_keys = ['predicted_fr', 
                             'adjusted_unconnected_target_fr', 
                             'desired_connected_fr', 
                             'desired_unconnected_fr', 
                             'connection_fr_increase', 
                             'connection_fr_error', 
                             'recorded_proportion_of_in_vivo_FR', 
                             'desired_connected_proportion_of_invivo_frs']
        
        if (comparison_pair[0] in prespecified_keys):
            stat_key_1 = comparison_pair[0]
            window_key_1 = window_key_placeholder
        if (comparison_pair[1] in prespecified_keys):
            stat_key_2 = comparison_pair[1]
            window_key_2 = window_key_placeholder
            
        df1 = custom_by_neuron_class_df.etl.q(window=window_key_1, neuron_class=neuron_classes)
        df2 = custom_by_neuron_class_df.etl.q(window=window_key_2, neuron_class=neuron_classes)

        label = ''
        if (len(list(stat_filter.keys()))):
            df1 = df1.etl.q(stat_filter)
            df2 = df2.etl.q(stat_filter)

            label = list(stat_filter.keys())[0]
            
        plt.figure()
        plt.scatter(df1[stat_key_1], df2[stat_key_2], c=df1['ca'], s=2.)
        fn = 'frs_' + comparison_pair[0] + '_VS_' + comparison_pair[1] + '_' + label + '.pdf'
        plt.savefig(str(a.figpaths.fr_condition_comparisons) + '/' + fn, bbox_inches='tight')
        plt.close()


        slope_dict = plot_two_conditions_comparison_by_neuron_class(df1, 
                                                    df2, 
                                                    stat_key_1, 
                                                    stat_key_2, 
                                                    'ca',
                                                    comparison_pair, 
                                                    str(a.figpaths.fr_condition_comparisons) + '/' + 'nc_frs_' + comparison_pair[0] + '_VS_' + comparison_pair[1] + '_' + label + '.pdf',
                                                    fit_and_plot_exponential=fit_and_plot_exponential,
                                                    neuron_classes=neuron_classes)





############################################################################ 
### Plot P_FR validations
############################################################################ 
def plot_nc_proportion_of_invivo_for_multiple_sims(a, by_neuron_class, stat_filter, file_name):

    plt.figure(figsize=(10, 5))
    
    cmap_nc = cm.get_cmap('viridis', len(c_etl.LAYER_EI_NEURON_CLASSES))

    unique_depol_stdev_mean_ratios = by_neuron_class['depol_stdev_mean_ratio'].unique()
    cmap = cm.get_cmap('viridis', len(unique_depol_stdev_mean_ratios))

    prev_ca = -1.0
    xtick_positions = []
    xtick_labels = []

    filtered_custom_by_neuron_class_df = by_neuron_class.etl.q(window="conn_spont", neuron_class=c_etl.LAYER_EI_NEURON_CLASSES)
    filtered_custom_by_neuron_class_df = filtered_custom_by_neuron_class_df.etl.q(stat_filter)

    for filtered_sim_index, simulation_id in enumerate(filtered_custom_by_neuron_class_df["simulation_id"].unique()):

        subset_df = filtered_custom_by_neuron_class_df.etl.q(simulation_id=simulation_id)
        
        if (len(subset_df) > 0):

            plt.scatter([filtered_sim_index], subset_df['desired_connected_proportion_of_invivo_frs'].iloc[0], c='k', s=4)

            ca = subset_df['ca'].iloc[0]
            depol_stdev_mean_ratio = subset_df['depol_stdev_mean_ratio'].iloc[0]

            if (ca != prev_ca):
                if (prev_ca != -1.0):
                    plt.plot([filtered_sim_index - 0.5, filtered_sim_index - 0.5], [0.0, 1.1], c='k')

                num_ca_sims = (len(filtered_custom_by_neuron_class_df.etl.q(ca=ca)) / len(c_etl.LAYER_EI_NEURON_CLASSES))
                xtick_loc = filtered_sim_index
                if (num_ca_sims > 1):
                    xtick_loc = filtered_sim_index + (num_ca_sims / 2.0) - 0.5

                xtick_positions.append(xtick_loc)
                xtick_labels.append(str(ca))

                prev_ca = ca

            xs = [filtered_sim_index for i in range(len(subset_df))]
            ys = []
            for subset_index, subset_row in subset_df.iterrows():
                ys.append(subset_row['mean_of_mean_firing_rates_per_second'] / a.analysis_config.custom['vivo_frs'][subset_row['neuron_class']])

            scat_cols = ['r' if 'EXC' in nc else 'b' for nc in c_etl.LAYER_EI_NEURON_CLASSES]
            nc_markers = [c_etl.LAYER_EI_NEURON_CLASS_MARKERS[nc] for nc in c_etl.LAYER_EI_NEURON_CLASSES]

            for x, y, m, c in zip(xs, ys, nc_markers, scat_cols):
                plt.scatter(x, y, c=c, marker=m, s=2, zorder=4)

            line_c = cmap(unique_depol_stdev_mean_ratios.tolist().index(depol_stdev_mean_ratio))
            plt.plot(xs, ys, c=line_c, lw=.5, zorder=3)

    plt.gca().set_ylim([0.0, 1.1])
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.gca().set_xticks(xtick_positions, xtick_labels)

    plt.gca().set_xlabel(c_etl.parameter_constants['ca']['axis_label'])
    plt.gca().set_ylabel('Proportion of\nin vivo MFR')

    for file_type in ['.pdf', '.png']:
        plt.savefig(str(a.figpaths.root) + "/" + file_name + file_type)
    plt.close()


def plot_nc_proportion_of_invivo_for_single_param_set(a, by_neuron_class, stat_filter, file_name):

    plt.figure(figsize=(10.35*0.35, 10.35*0.35))

    filtered_custom_by_neuron_class_df = by_neuron_class.etl.q(window="conn_spont", neuron_class=c_etl.LAYER_EI_NEURON_CLASSES).etl.q(stat_filter)

    for simulation_id in filtered_custom_by_neuron_class_df["simulation_id"].unique():

        subset_df = filtered_custom_by_neuron_class_df.etl.q(simulation_id=simulation_id)
        
        if (len(subset_df) > 0):

            xs = subset_df['desired_connected_proportion_of_invivo_frs']
            ys = [subset_row['mean_of_mean_firing_rates_per_second'] / a.analysis_config.custom['vivo_frs'][subset_row['neuron_class']] for _, subset_row in subset_df.iterrows()]
            scat_cols = ['r' if 'EXC' in nc else 'b' for nc in c_etl.LAYER_EI_NEURON_CLASSES]
            nc_markers = [c_etl.LAYER_EI_NEURON_CLASS_MARKERS[nc] for nc in c_etl.LAYER_EI_NEURON_CLASSES]
            for x, y, m, c in zip(xs, ys, nc_markers, scat_cols):
                plt.scatter(x, y, c=c, marker=m, s=10, zorder=5, alpha=0.5, linewidth=0)


    if (len(filtered_custom_by_neuron_class_df['desired_connected_proportion_of_invivo_frs'])):

        max_prop = filtered_custom_by_neuron_class_df['desired_connected_proportion_of_invivo_frs'].max()
        min_prop = filtered_custom_by_neuron_class_df['desired_connected_proportion_of_invivo_frs'].min()
        plt.plot([0.0, 1.0], [0.0, 1.0], c='grey', linewidth=0.5)

        plt.gca().set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.gca().set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.gca().set_xlim([0.0, 1.005])
        plt.gca().set_ylim([0.0, 1.1])
        
        plt.gca().spines.right.set_visible(False)
        plt.gca().spines.top.set_visible(False)
        plt.gca().set_xlabel('Target $P_{FR}$')
        plt.gca().set_ylabel('Observed $P_{FR}$')

        plt.gca().set_aspect('equal', 'box')

    for file_type in ['.pdf', '.png']:
        plt.savefig(str(a.figpaths.root) + "/" + file_name + file_type, bbox_inches='tight')
    plt.close()           
            


############################################################################ 
### Plot Sim Stat Lines
############################################################################ 
from itertools import chain
import matplotlib
from matplotlib import colors
import numpy as np
import seaborn as sns
import math
def plot_sim_stat_lines_all_sims(a, stats_df, stat_key, stat_filter, file_name, neuron_class_groupings, neuron_class_grouping_cmaps, show_in_vivo_FRs=False, label_map=None, custom_x_lims=[], show_colorbar=False, figsize=(2,3.5), major_loc=None, minor_loc=None, file_types=['.pdf', '.png']):

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    unique_desired_conn = np.sort(stats_df['desired_connected_proportion_of_invivo_frs'].unique())
    flattend_neuron_classes = list(chain.from_iterable(neuron_class_groupings))
    num_neuron_classes = len(flattend_neuron_classes)
    
    ax.set_ylim([-0.25, num_neuron_classes - 1 + 0.25])
    
    max_stats_used = []
    for neuron_classes, cmap_name in zip(neuron_class_groupings, neuron_class_grouping_cmaps):
        
        cmap = sns.color_palette(cmap_name, len(unique_desired_conn), as_cmap=True)
        color_vals = cmap(np.linspace(0, 1, len(unique_desired_conn) + 1))

        for simulation_id in a.repo.simulations.df['simulation_id']:
            ng_stats = stats_df.etl.q(simulation_id=simulation_id, window="conn_spont", neuron_class=neuron_classes)
            filtered_stats = ng_stats.etl.q(stat_filter)
            stats = filtered_stats[stat_key]
            
            neuron_class_labels = neuron_classes
            if label_map != None:
                neuron_class_labels = [label_map[nc] for nc in neuron_classes]
            
            if (len(stats) > 0):

                max_s = np.nanmax([v for v in stats if not (math.isinf(v) or math.isnan(v))])
                max_stats_used.append(max_s)

                ordered_stats = [filtered_stats.etl.q(neuron_class=nc)[stat_key].iloc[0] for nc in neuron_classes]
                
                c = color_vals[np.where(unique_desired_conn == ng_stats['desired_connected_proportion_of_invivo_frs'].iloc[0])[0][0] + 1]
                plt.plot(ordered_stats, neuron_class_labels, c=c, lw=0.25, zorder=2)
                
                if (show_in_vivo_FRs):
                    vivo_c = 'g'
                    invivo_frs = [a.analysis_config.custom['vivo_frs'][nc] for nc in neuron_classes]
                    
                    plt.plot(invivo_frs, neuron_class_labels, label='In vivo', c=vivo_c, lw=2, zorder=-3, marker='.', ms=5)
#                     invivo_ng_marker_dict = {"L1_INH": 'v', "L23_EXC": 'P', "L23_INH": 'P', "L4_EXC": 'P', "L4_INH": 'P', "L5_EXC": 'P', "L5_INH": 'P', "L6_EXC": 's', "L6_INH": 'v'}
#                     for ng in invivo_ngs:
#                         plt.scatter(a.analysis_config.custom['vivo_frs'][ng], [c_etl.bluepy_neuron_class_map[ng]], marker=invivo_ng_marker_dict[ng], color=vivo_c, zorder=4)
                
    matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
#     ax.spines.bottom.set_visible(False)
#     ax.spines.top.set_visible(True)
#     ax.set_xlim(left=0)
    ax.set_xlabel(c_etl.label_for_key(stat_key), labelpad=0)
    if (len(custom_x_lims)):
        ax.set_xlim(custom_x_lims)
#     ax.xaxis.tick_top()
#     ax.xaxis.set_label_position('top')
    
    if (major_loc != None):
        ax.xaxis.set_major_locator(major_loc)
    if (minor_loc != None):
        ax.xaxis.set_minor_locator(minor_loc)
    else:
        c_etl.remove_intermediate_labels(ax.xaxis.get_major_ticks())
    ax.tick_params(axis='both', pad=1)
    if (show_colorbar):
        
        vmin = 0.0
        vmax = np.max(unique_desired_conn)
        norm = colors.Normalize(vmin=0.0, vmax=vmax)
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='$P_{FR}$', location='right', shrink=0.5, pad=-0.25)
#         cbar.ax.tick_params(labelsize=9)

    for file_type in file_types:
        plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()
    plt.close()

############################################################################ 
### Helper
############################################################################ 
def remove_intermediate_labels(xticks):
    
    n_ticks = len(xticks)
    for i, _ in enumerate(xticks):
        if ((i != 0) & (i != (n_ticks - 1))):
            xticks[i].label1.set_visible(False)
            xticks[i].label2.set_visible(False)


############################################################################ 
### Plot Bursting Ratios
############################################################################ 
def plot_bursting_ratios(a, bursting_ratios):

    plt.figure()
    hist_ns, _, _ = plt.hist(bursting_ratios, bins=40, range=(0.0, 1.0))
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.plot([0.125, 0.125], [0.0, np.max(hist_ns)], c='k', linestyle='--')
    plt.gca().set_xlabel("Bursting ratio")
    plt.gca().set_ylabel("Frequency")
    plt.savefig(str(a.figpaths.root) + "/BurstingRatio.pdf" )
    plt.close()
