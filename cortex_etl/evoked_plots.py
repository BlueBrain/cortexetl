import cortex_etl as c_etl
import numpy as np
import matplotlib.pyplot as plt
def layer_and_pairwise_comparison(nc_stat_and_mask_df, vivo_df, vivo_exp, decay, nc_pairs, path, xlim, xticks):

    flattened_ncs = [item for sublist in nc_pairs for item in sublist]

    plt.figure(figsize=(8.2*0.2, 8.2*0.2))
    for simulation_id in nc_stat_and_mask_df.simulation_id.unique():
        for nc_pair in nc_pairs:
            cs = [c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[nc_pair[0]]['color'], c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[nc_pair[1]]['color']]
            q_df = nc_stat_and_mask_df.etl.q(simulation_id=simulation_id, neuron_class=nc_pair, SimBad100p50p25pDecayOverlySustained=False)
            if len(q_df):
                x_to_plot =[q_df.etl.q(neuron_class=nc_pair[0]).iloc[0][decay] + np.random.normal(0.0, 0.04, 1), q_df.etl.q(neuron_class=nc_pair[1]).iloc[0][decay] + np.random.normal(0.0, 0.04, 1)]
                y_to_plot = [flattened_ncs.index(nc_pair[0]) + 1, flattened_ncs.index(nc_pair[1]) + 1]
                plt.plot(x_to_plot, y_to_plot, c='grey', lw=.2)
                plt.scatter(x_to_plot, y_to_plot, c=cs, s=0.1)

    for nc_pair in nc_pairs:
        cs = [c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[nc_pair[0]]['color'], c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[nc_pair[1]]['color']]
        nc_pair_df = vivo_df.etl.q(neuron_class=nc_pair, bin_size=1.0, sigma=1, experiment=vivo_exp)
        if (len(nc_pair_df)):
            x_to_plot = [nc_pair_df.etl.q(neuron_class=nc_pair[0]).iloc[0][decay], nc_pair_df.etl.q(neuron_class=nc_pair[1]).iloc[0][decay]]
            y_to_plot = [flattened_ncs.index(nc_pair[0]) + 1, flattened_ncs.index(nc_pair[1]) + 1]
            plt.plot(x_to_plot, y_to_plot, c='k', lw=1.0)
            plt.scatter(x_to_plot, y_to_plot, c=cs, s=1.0)

    plt.gca().set_xlabel("Time (ms)")
    plt.gca().set_xlim(xlim)
    plt.gca().set_xticks(ticks=xticks)
    plt.gca().set_yticks(ticks=np.arange(len(flattened_ncs)) + 1,labels=[c_etl.neuron_class_label_map[nc] for nc in flattened_ncs])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


import cortex_etl as c_etl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def compare_metrics_in_vivo_vs_in_silico(df, vivo_df, vivo_exp, layers, groups, stat, prefix, figs_dir, upper_lim=20.0, mask_key=''):

    if (mask_key != ''):
        mask_dict = {mask_key:False}
        df = df.etl.q(mask_dict)

    cmap = cm.get_cmap('rocket', 5)
    colors = cmap(np.linspace(0, 1, 5))
    layer_colours = { "L23": colors[0], "L4": colors[1], "L5": colors[2], "L6": colors[3]}
    group_colours = {"EXC": 'r', "INH": 'b', "PV": 'g', "SST": 'k'}

    
    for layer in layers:
        for group in groups:
            neuron_class = layer + "_" + group
            nc_df = df.etl.q(neuron_class=neuron_class)
            vivo_stat_df_1 = vivo_df.etl.q(experiment=vivo_exp, neuron_class=neuron_class, bin_size=1, sigma=1)
            num_points = len(nc_df)
            plt.scatter(np.asarray([vivo_stat_df_1.iloc[0][stat] for i in range(num_points)]) + np.random.normal(0.0, 0.04, num_points), nc_df[stat] + np.random.normal(0.0, 0.04, num_points), s=0.2, marker=c_etl.LAYER_MARKERS[layer], c=[group_colours[group] for i in range(num_points)])

    plt.gca().set_xlim([0.0, upper_lim])
    plt.gca().set_ylim([0.0, upper_lim])
    plt.gca().set_xlabel("In vivo")
    plt.gca().set_ylabel("In silico")
    plt.gca().plot([0.0, upper_lim], [0.0, upper_lim], 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
    plt.savefig(figs_dir + vivo_exp + "COMP_" + prefix + '_' + groups[0] + '_+_' + groups[1] + '_' + stat + '_' + mask_key + '.pdf')
    plt.close()


import cortex_etl as c_etl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
def compare_metrics_of_two_layerwise_groups(df, vivo_df, vivo_experiments, layers, group_1, group_2, stat, prefix, figs_dir, mask_key=''):
    if (mask_key != ''):
        mask_dict = {mask_key:False}
        df = df.etl.q(mask_dict)

    if len(df):

        cmap = cm.get_cmap('rocket', 5)
        colors = cmap(np.linspace(0, 1, 5))
        layer_colours = { "L23": colors[0], "L4": colors[1], "L5": colors[2], "L6": colors[3]}


        group_1_df = df.etl.q(neuron_class=[layer + "_" + group_1 for layer in layers])
        group_2_df = df.etl.q(neuron_class=[layer + "_" + group_2 for layer in layers])

        group_1_df.loc[:, 'layer_colour'] = group_1_df.apply(lambda row: layer_colours[row['layer']], axis = 1)
        layer_df = pd.merge(group_1_df, group_2_df, on=['simulation_id', "layer"])
        layer_df[stat + "_x"] = layer_df[stat + "_x"] + np.random.normal(0.0, 0.05, len(layer_df[stat + "_x"]))
        layer_df[stat + "_y"] = layer_df[stat + "_y"] + np.random.normal(0.0, 0.05, len(layer_df[stat + "_y"]))

        plt.figure()
        plt.scatter(layer_df[stat + "_x"], layer_df[stat + "_y"], c=np.asarray(layer_df['layer_colour']).tolist(), s=0.2)

        # PLOT IN VIVO REFERENCES
        vivo_m = ['s', 'x']
        for vivo_ind, vivo_exp in enumerate(vivo_experiments):
            exp_df = vivo_df.etl.q(experiment=vivo_exp)
            for layer in layers:
                vivo_stat_df_1 = exp_df.etl.q(neuron_class=layer + "_" + group_1, bin_size=1, sigma=1)
                vivo_stat_df_2 = exp_df.etl.q(neuron_class=layer + "_" + group_2, bin_size=1, sigma=1)
                if (len(vivo_stat_df_1) & len(vivo_stat_df_2)):
                    plt.scatter([vivo_stat_df_1.iloc[0][stat]], [vivo_stat_df_2.iloc[0][stat]], c=[layer_colours[layer]], label=layer, s=5, marker=vivo_m[vivo_ind])

        plt.gca().legend()
        plt.gca().set_xlim([0.0, 20.0])
        plt.gca().set_ylim([0.0, 20.0])
        plt.gca().set_xlabel(group_1)
        plt.gca().set_ylabel(group_2)
        plt.gca().plot([0.0, 20.0], [0.0, 20.0], 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
        plt.savefig(figs_dir + prefix + '_' + group_1 + '_v_' + group_2 + '_' + stat + '_' + mask_key + '.pdf')
        plt.close()






def mean_normalise(data):
    normalised_data = data / np.mean(data)
    return normalised_data

import seaborn as sns   
def evoked_ratios_line_plot(a, silico_evoked_ratios, silico_mean_ratios_across_sims, vivo_nc_ratios, neuron_classes, normalisation_type='none'):

    plt.figure(figsize=(8.2*0.3, 8.2*0.3*0.5))
    plt.gca().set_facecolor('lightgrey')
    cmap = sns.color_palette("seismic", as_cmap=True)

    vivo_nc_mean_ratio = np.mean(vivo_nc_ratios)

    silico_mean_ratio_differences = []
    for simulation_id in silico_evoked_ratios.simulation_id.unique():

        simulation_nc_evoked_ratios = silico_evoked_ratios.etl.q(simulation_id=simulation_id, neuron_class=neuron_classes)
        
        if (normalisation_type == "none"):
            silico_mean_ratio_differences.append(np.mean(np.asarray(simulation_nc_evoked_ratios['ratio'])) - vivo_nc_mean_ratio)
        elif (normalisation_type == "mean_normalise"):
            silico_mean_ratio_differences.append(np.mean(mean_normalise(np.asarray(simulation_nc_evoked_ratios['ratio']))) - mean_normalise(vivo_nc_mean_ratio))


    if (normalisation_type == "none"):
        a.custom['all_window_stat_and_mask_df'].loc[a.custom['all_window_stat_and_mask_df'].etl.q(simulation_id=silico_evoked_ratios.simulation_id.unique()).index, "mean_ratio_difference"] = silico_mean_ratio_differences


    abs_max_silico_mean_ratio_differences = np.max(np.abs(silico_mean_ratio_differences))
    range_width = abs_max_silico_mean_ratio_differences * 2

    for i, simulation_id in enumerate(silico_evoked_ratios.simulation_id.unique()):

        simulation_nc_evoked_ratios = silico_evoked_ratios.etl.q(simulation_id=simulation_id, neuron_class=neuron_classes)
        c_val = (silico_mean_ratio_differences[i] / range_width) + 0.5

        if (normalisation_type == "none"):
            plt.plot(range(len(neuron_classes)), simulation_nc_evoked_ratios['ratio'], c=cmap(c_val), lw=.2)
        elif (normalisation_type == "mean_normalise"):
            plt.plot(range(len(neuron_classes)), mean_normalise(simulation_nc_evoked_ratios['ratio']), c='k', lw=.2)

    if (normalisation_type == "none"):
        plt.plot(range(len(neuron_classes)), vivo_nc_ratios, c='w', lw=1, marker='o', ms=3)
        plt.plot(range(len(neuron_classes)), silico_mean_ratios_across_sims.etl.q(neuron_class=neuron_classes), c='k', lw=1, marker='o', ms=3)
        plt.gca().set_ylim([0.0, 200.0])

    elif (normalisation_type == "mean_normalise"):
        plt.plot(range(len(neuron_classes)), mean_normalise(vivo_nc_ratios), c='w', lw=1, marker='o', ms=3)
        plt.plot(range(len(neuron_classes)), mean_normalise(silico_mean_ratios_across_sims.etl.q(neuron_class=neuron_classes)), c='k', lw=1, marker='o', ms=3)
        plt.gca().set_ylim([0.0, 5.0])
        # plt.gca().set_ylim([0.0, 1.0])



    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(50))
    plt.gca().set_xticks(range(len(neuron_classes)), labels=[c_etl.neuron_class_label_map[nc] for nc in neuron_classes])
    plt.gca().set_ylabel('Evok/spont\nratio', labelpad=-10.0)
    plt.tight_layout()
    plt.savefig(str(a.figpaths.evoked) + "/" + normalisation_type + "_EvokedRatios.pdf")

    # plt.close()




# PSTH PLOTS
######################################################################


def renormalise_psth(psth):
    new_hist = psth  - np.min(psth)
    new_hist = new_hist / np.max(new_hist)
    return new_hist

def psth(psth):
    return psth

def colour_for_mask(sim_colour_row, c_for_masked_psths, c_for_unmasked, mask_key):
    c=c_for_unmasked
    zorder=5
    if (sim_colour_row[mask_key]):
        c=c_for_masked_psths
        zorder=4
    return c, zorder



def axis_psths_single_neuron_class(ax, neuron_class, baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, time_key, silico_sigma, silico_bin_size, vivo_bin_size, vivo_sigma, c_for_masked_psths, mask_key, major_locator, minor_locator, is_first_row, is_last_row, is_first_column, override_c_for_unmasked='', override_c_for_vivo=''):

    rp_psth_all_df = rp_psth_df.etl.q(barrel='C2', neuron_class=neuron_class)
    svo_psth_all_df = svo_psth_df.etl.q(neuron_class=neuron_class)
    psths_all = baseline_PSTH_df.etl.q(neuron_class=neuron_class, bin_size=silico_bin_size, sigma=silico_sigma)

    c_for_unmasked = 'g'
    if (override_c_for_unmasked==''):
        c_for_unmasked = c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[neuron_class]['color']
    else:
        c_for_unmasked = override_c_for_unmasked

    c_for_vivo = c_for_unmasked
    if (override_c_for_vivo != ''):
        c_for_vivo = override_c_for_vivo

    for simulation_id in psths_all['simulation_id'].unique():
        sim_colour_df = nc_window_comparison_decay_df.etl.q(simulation_id=simulation_id, neuron_class=neuron_class)
        if (len(sim_colour_df)):
            psth_sim = psths_all.etl.q(simulation_id=simulation_id)

            psth_c, zorder = colour_for_mask(sim_colour_df.iloc[0], c_for_masked_psths, c_for_unmasked, mask_key)
            if (psth_c != 'w'):
                ax.plot(psth_sim[time_key], psth(psth_sim['psth']), lw=0.1, c=psth_c, zorder=zorder)    
        else:
            print(simulation_id)


    if (major_locator != None):
        ax.xaxis.set_major_locator(major_locator)
    if (minor_locator != None):
        ax.xaxis.set_minor_locator(minor_locator)

    # ax.tick_params(axis='both', which='major', labelsize=8)

    if (is_first_row):
        ax.set_title(c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[neuron_class]['synapse_class'])

    if (is_last_row):
        ax.set_xlabel("Time (ms)", labelpad=-7.0)
    else:
        ax.axes.xaxis.set_ticklabels([])

    if (is_first_column):
        ax.set_ylabel(c_etl.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[neuron_class]['layer_string'], labelpad=2.0, rotation=0)
    else:
        ax.axes.yaxis.set_ticklabels([])

    ax.set_yticks(ticks=[0.0, 1.0],labels=['', '']) #, rotation=25,fontsize=8

    filtered_rp_psth_all_df = rp_psth_all_df.etl.q(sigma=vivo_sigma, bin_size=vivo_bin_size)
    ax.plot(filtered_rp_psth_all_df[time_key], filtered_rp_psth_all_df['psth_norm'], c=c_for_vivo, lw=0.3, zorder=6)
    filtered_svo_psth_all_df = svo_psth_all_df.etl.q(sigma=vivo_sigma, bin_size=vivo_bin_size)
    ax.plot(filtered_svo_psth_all_df[time_key], filtered_svo_psth_all_df['psth_norm'], c=c_for_vivo, lw=0.3, linestyle='--', zorder=6)


import matplotlib.ticker as ticker
def plot_psths_neuron_class_pairs(neuron_class_pairs, baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, figs_dir, c_for_masked_psths, xlim, prefix, silico_bin_size, silico_sigma, mask_key, major_locator, minor_locator, override_c_for_unmasked='', override_c_for_vivo=''):

    baseline_PSTH_df = baseline_PSTH_df.etl.q(bin_size=silico_bin_size, sigma=silico_sigma)

    time_key = "time"

    vivo_bin_size = 1
    vivo_sigma = 1

    n_cols = len(neuron_class_pairs[0])

    fig, axs = plt.subplots(ncols=n_cols, nrows=len(neuron_class_pairs), figsize=(8.2*0.66666, 8.2*0.66*0.66), layout="constrained")

    num_rows = len(neuron_class_pairs)

    for row in range(num_rows):
        for col in range(n_cols):
            ax = axs[row,col]

            if (neuron_class_pairs[row][col] != ""):
                axis_psths_single_neuron_class(ax, neuron_class_pairs[row][col], baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, time_key, silico_sigma, silico_bin_size, vivo_bin_size, vivo_sigma, c_for_masked_psths, mask_key, major_locator, minor_locator, row==0, row==(num_rows-1), col==0, override_c_for_unmasked=override_c_for_unmasked, override_c_for_vivo=override_c_for_vivo)
            else:
                ax.set_axis_off()
            ax.set_ylim([0.0, 1.0])
            ax.set_xlim(xlim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5),
            #                        transform=axs[row, col].transAxes,
            #                        ha='center', va='center', fontsize=18,
            #                        color='darkgrey')

    ticker.MultipleLocator(50), ticker.MultipleLocator(10),

    plt.savefig(figs_dir + prefix + "_" + str(silico_bin_size) + '_' + str(silico_sigma) + 'GROUPED' + '_PSTHs.pdf')
    plt.close()


def plot_psths_single_neuron_class(neuron_class, baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, figs_dir, c_for_masked_psths, silico_bin_size, silico_sigma, override_c_for_unmasked='', override_c_for_vivo=''):
    
    baseline_PSTH_df = baseline_PSTH_df.etl.q(bin_size=silico_bin_size, sigma=silico_sigma)

    time_key = "time"

    vivo_bin_size = 1
    vivo_sigma = 1

    plt.figure(figsize=(8.2*0.2, 8.2*0.2*0.5))
    ax = plt.gca()

    axis_psths_single_neuron_class(ax, neuron_class, baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, time_key, silico_sigma, silico_bin_size, vivo_bin_size, vivo_sigma, c_for_masked_psths, 'Bad100p50p25pDecayOverlySustained', None, None, False, True, True, override_c_for_unmasked=override_c_for_unmasked, override_c_for_vivo=override_c_for_vivo)

    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Time (ms)", labelpad=-7.0)
    ax.set_ylabel(c_etl.neuron_class_label_map[neuron_class], labelpad=2.0, rotation=0)
    ax.set_yticks(ticks=[0.0, 1.0],labels=['', ''])


    plt.gca().set_xlim([0.0, 100.0])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    plt.tight_layout()
    plt.savefig(figs_dir + str(silico_bin_size) + '_' + str(silico_sigma) + '_' + neuron_class + '_PSTHs.pdf')

    plt.gca().set_xlim([0.0, 25.0])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig(figs_dir + str(silico_bin_size) + '_' + str(silico_sigma) + '_' + neuron_class + '_ALL_PSTHs_CLOSE.pdf')
    plt.close()


