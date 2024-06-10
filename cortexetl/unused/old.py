import cortexetl as c_etl


def plot_3x3s(a):

	for window_str in a.analysis_config['3x3_windows']:
		features_with_sim_info = pd.merge(a.features.by_neuron_class.df.etl.q(window=window_str).reset_index(), a.repo.simulations.df.reset_index())

		for custom_lims, lims_label in [([], ''), ([0.0, 2.4], '_vivo_lims')]:

			params = (a.analysis_config['vivo_frs'], window_str, 'mean_of_mean_firing_rates_per_second', custom_lims,)
			pairwise_grid_comparison(features_with_sim_info, 
											[a.analysis_config['3x3s'] / (window_str + lims_label + '_COMPARISON.png')], 
											*params, draw_legend=True)

			# file_paths_for_video = []
			# for simulation_id in a.repo.simulations.df['simulation_id']:

			# 	single_sim_features_with_sim_info = features_with_sim_info.etl.q(simulation_id=simulation_id)
			# 	file_path = single_sim_features_with_sim_info.iloc[0]['figures_dir'] / (window_str + lims_label + '_COMPARISON.png')
			# 	file_paths_for_video.append(str(file_path))
			# 	figures.pairwise_grid_comparison(single_sim_features_with_sim_info, 
			# 									[file_path], 
			# 									*params, draw_legend=True)
				

			# video.video_from_image_files(file_paths_for_video, str(a.analysis_config['3x3s'] / (window_str + lims_label + '_COMPARISON.mp4')))


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator

def pairwise_grid_comparison(features_with_sim_info,
                            file_paths, 
                            vivo_fr_dict,
                            window_str, 
                            stat_key, 
                            custom_lims=[],
                            draw_legend=True
                            ):

    fig, axes = plt.subplots(3, 3, figsize=(10,10))
    mins_and_maxs = []
    params = (stat_key, features_with_sim_info, vivo_fr_dict, window_str, ) # p_df = params + ('default_heatmap_params',)

    mins_and_maxs.append(difference_plotter(axes[0][0], ['L23_INH', 'L4_INH'], *params, plot_yticklabels=True, plot_xticklabels=False, draw_legend=draw_legend))
    mins_and_maxs.append(difference_plotter(axes[1][0], ['L23_INH', 'L5_INH'], *params, plot_yticklabels=True, plot_xticklabels=False))
    mins_and_maxs.append(difference_plotter(axes[2][0], ['L4_INH', 'L5_INH'], *params, plot_yticklabels=True, plot_xticklabels=True))

    mins_and_maxs.append(difference_plotter(axes[0][1], ['L23_EXC', 'L4_EXC'], *params, plot_yticklabels=False, plot_xticklabels=False))
    mins_and_maxs.append(difference_plotter(axes[1][1], ['L23_EXC', 'L5_EXC'], *params, plot_yticklabels=False, plot_xticklabels=False))
    mins_and_maxs.append(difference_plotter(axes[2][1], ['L4_EXC', 'L5_EXC'], *params, plot_yticklabels=False, plot_xticklabels=True))

    mins_and_maxs.append(difference_plotter(axes[0][2], ['L23_EXC', 'L23_INH'], *params, plot_yticklabels=False, plot_xticklabels=False))
    mins_and_maxs.append(difference_plotter(axes[1][2], ['L4_EXC', 'L4_INH'], *params, plot_yticklabels=False, plot_xticklabels=False))
    mins_and_maxs.append(difference_plotter(axes[2][2], ['L5_EXC', 'L5_INH'], *params, plot_yticklabels=False, plot_xticklabels=True))

    overall_min = np.min(np.asarray(mins_and_maxs)[:, 0]) - 0.1
    overall_max = np.max(np.asarray(mins_and_maxs)[:, 1]) + 0.1

    for ax_list in axes:
        for ax in ax_list:

            if (custom_lims == []):
                ax.set_xlim([0.0, overall_max])
                ax.set_ylim([0.0, overall_max])
            else:
                ax.set_xlim(custom_lims)
                ax.set_ylim(custom_lims)

            x = np.linspace(0,int(overall_max), 10)
            y = x
            ax.plot(x, y, '--k', linewidth=2, zorder=1)

            ax.xaxis.set_major_locator(FixedLocator(ax.get_yticks()))

            ax.set_aspect('equal', 'box')

    plt.suptitle(stat_key)
    
    for file_path in file_paths:
        plt.savefig(file_path)
    plt.close()
    


def difference_plotter(ax, ng_keys, stat_key, features_with_sim_info, vivo_fr_dict, window_str, plot_yticklabels=False, plot_xticklabels=False, draw_legend=False):

    values_for_min_max = []

    features_ng_0 = features_with_sim_info.etl.q(neuron_class=ng_keys[0])
    features_ng_1 = features_with_sim_info.etl.q(neuron_class=ng_keys[1])

    option = "1"
    if (option == "1"):
        silico_xs = features_ng_0[stat_key].to_numpy()
        silico_ys = features_ng_1[stat_key].to_numpy()

        ca_markers = []
        ca_colours = []
        for feature_ind, feature_row in features_ng_0.iterrows():
            ca = feature_row['simulation'].config['Run_Default']['ExtracellularCalcium']

            if (ca == '1.0'):
                ca_markers.append('.')
                ca_colours.append('b')
            else:
                ca_markers.append('+')
                ca_colours.append('g')

        for i in range(len(silico_xs)):
            ax.scatter(silico_xs[i], silico_ys[i], marker=ca_markers[i], c=ca_colours[i], zorder=2)
    

    # NEEDS TO BE FIXED
    # elif (option == "2"):

    #     silico_xs = []
    #     silico_ys = []
    #     labels = []
    #     c = 0
    #     for x, y, ca, fr_scale, stdev_mean_ratio  in zip(features_ng_0[stat_key].to_list(), features_ng_1[stat_key].to_list(), features_ng_1["ca"].to_list(), features_ng_1["fr_scale"].to_list(), features_ng_1["depol_stdev_mean_ratio"]):
    #         c+=1
    #         # print(x, y)
    #         if ((x < vivo_fr_dict[ng_keys[0]] + 0.1 * vivo_fr_dict[ng_keys[0]]) & (y < vivo_fr_dict[ng_keys[1]] + 0.1 * vivo_fr_dict[ng_keys[1]])):
                
    #             print(ca, fr_scale, stdev_mean_ratio)
    #             silico_xs.append(x)
    #             silico_ys.append(y)
    #             labels.append(str(c))
                


    #     colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(silico_xs)))
    
    #     for i in range(len(silico_xs)):
    #         ax.scatter(silico_xs[i], silico_ys[i], c=colors[i], s=20, label=labels[i], zorder=2)


    values_for_min_max.extend(silico_xs); values_for_min_max.extend(silico_ys)


    if (draw_legend):
        ax.legend()

    # VIVO
    stat_label = ""
    if (stat_key == "mean_of_mean_firing_rates_per_second"):
        stat_label = "Mean FR (spikes/s)"


        xs = [vivo_fr_dict[ng_keys[0]]]
        ys = [vivo_fr_dict[ng_keys[1]]]
        ax.scatter(xs, ys, c='r', zorder=2, marker='x')

        values_for_min_max.extend(xs); values_for_min_max.extend(ys)


    if (not plot_yticklabels):
        ax.set_yticklabels([])
        ax.set_ylabel(ng_keys[1])
    else:
        ax.set_ylabel(stat_label + "\n" + ng_keys[1])

    if (not plot_xticklabels):
        ax.set_xticklabels([])
        ax.set_xlabel(ng_keys[0])
    else:
        ax.set_xlabel(ng_keys[0] + "\n" + stat_label)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if (len(values_for_min_max)):
        this_min = np.min(values_for_min_max)
        this_max = np.max(values_for_min_max)
    else:
        this_min = 0.0
        this_max = 0.0001

    return [this_min, this_max]
