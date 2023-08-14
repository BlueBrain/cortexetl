import cortex_etl as c_etl

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress
import json
import sys
from matplotlib import cm
import math
import seaborn as sns
import pandas as pd


def heatmaps_and_lineplots(a):

    print("\n----- Plot meta heatmaps -----")

    # EVOKED
    file_dir = str(a.figpaths.heatmaps) + '/'
    hm_dims = (a.analysis_config.custom['heatmap_dims']['hor_key'], 
               a.analysis_config.custom['heatmap_dims']['ver_key'], 
               a.analysis_config.custom['heatmap_dims']['x_key'], 
               a.analysis_config.custom['heatmap_dims']['y_key'])

    if ('evoked_window_for_custom_post_analysis' in list(a.analysis_config.custom.keys())):

        for stat_key in ["overly_sustained_response", "higher_secondary_peak", "too_much_trial_to_trial_variance", "evoked_mask"]:
            c_etl.heatmap(a.custom['custom_simulations_post_analysis'], stat_key, file_dir + stat_key, *hm_dims)

        for (mask_key, highlight_false_key) in [("", ""), ("evoked_mask", "")]: #
            c_etl.heatmap(a.custom['custom_simulations_post_analysis'], "30ms_decay_point", file_dir + "30ms_decay_point", *hm_dims, mask_key=mask_key, highlight_false_key=highlight_false_key)
            c_etl.heatmap(a.custom['custom_simulations_post_analysis'], "60ms_decay_point", file_dir + "60ms_decay_point", *hm_dims, mask_key=mask_key, highlight_false_key=highlight_false_key)

        if ("mean_pairwise_first_spike_r_value" in a.custom['custom_simulations_post_analysis'].columns):
            c_etl.heatmap(a.custom['custom_simulations_post_analysis'], "mean_pairwise_first_spike_r_value", file_dir + "mean_pairwise_first_spike_r_value", *hm_dims, mask_key="evoked_mask")


    # SPONTANEOUS
    hm_dims = (a.analysis_config.custom['heatmap_dims']['hor_key'], a.analysis_config.custom['heatmap_dims']['ver_key'], a.analysis_config.custom['heatmap_dims']['x_key'], a.analysis_config.custom['heatmap_dims']['y_key'])
    for stat_key in ['atleast_one_neuron_class_fr_greater_than_invivo_thresh', 'simulation_id', 'bursting', 'bursting_ratio', 'bursting_or_fr_above_threshold']:
        c_etl.heatmap(a.custom['custom_simulations_post_analysis'], stat_key, file_dir + stat_key, *hm_dims)

    for (mask_key, highlight_false_key) in [("", ""), ("bursting", ""), ("bursting_or_fr_above_threshold", "bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range"), ("bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range", "")]:
        c_etl.heatmap(a.custom['custom_simulations_post_analysis'], "ei_corr_rval", file_dir + "ei_corr_rval", *hm_dims, mask_key=mask_key, highlight_false_key=highlight_false_key, figsize=(10.35, 6))

    features_by_neuron_class_with_sim_info = pd.merge(a.features.by_neuron_class.df.reset_index(), a.repo.simulations.df.reset_index())
    features_by_neuron_class_with_sim_info['log_mean_of_mean_firing_rates_per_second'] = np.log(features_by_neuron_class_with_sim_info['mean_of_mean_firing_rates_per_second'])

    c_etl.heatmap(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_EXC", window='conn_spont'), "mean_of_mean_firing_rates_per_second", file_dir + "ALL_EXC_FRs", *hm_dims)
    c_etl.heatmap(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_INH", window='conn_spont'), "mean_of_mean_firing_rates_per_second", file_dir + "ALL_INH_FRs", *hm_dims)

    c_etl.heatmap(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_EXC", window='conn_spont'), "log_mean_of_mean_firing_rates_per_second", file_dir + "ALL_EXC_LOG_FRs", *hm_dims)
    c_etl.heatmap(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_INH", window='conn_spont'), "log_mean_of_mean_firing_rates_per_second", file_dir + "ALL_INH_LOG_FRs", *hm_dims)


    hor_key="ca"; ver_key="depol_stdev_mean_ratio"; x_key="freq"; y_key="desired_connected_proportion_of_invivo_frs"
    hm_dims = (hor_key, ver_key, x_key, y_key)
    all_fft_dfs = pd.merge(a.custom['fft'].reset_index(), a.custom['custom_simulations_post_analysis'].reset_index()[['simulation_id', 'bursting', 'bursting_or_fr_above_threshold', 'bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range']])
    for (mask_key, highlight_false_key) in [("", ""), ("bursting", ""), ("bursting_or_fr_above_threshold", "bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range"), ("bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range", "")]:
        c_etl.heatmap(all_fft_dfs, "power", file_dir + "FFT", *hm_dims, mask_key=mask_key, highlight_false_key=highlight_false_key, figsize=(7, 2))
        # c_etl.heatmap(all_fft_dfs, "log_power", file_dir + "LogFFT", *hm_dims, mask_key=mask_key, highlight_false_key=highlight_false_key)


    print("\n----- Plot meta lineplots -----")

    hor_key="ca"; ver_key="none"; x_key="desired_connected_proportion_of_invivo_frs"; colour_var_key="depol_stdev_mean_ratio"
    hm_dims = (hor_key, ver_key, x_key, colour_var_key)
    file_dir = str(a.figpaths.lineplots) + '/'
    c_etl.lineplot(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_EXC", window='conn_spont'), "mean_of_mean_firing_rates_per_second", file_dir + "LP_ALL_EXC_FRs", *hm_dims)
    c_etl.lineplot(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_INH", window='conn_spont'), "mean_of_mean_firing_rates_per_second", file_dir + "LP_ALL_INH_FRs", *hm_dims)
    c_etl.lineplot(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_EXC", window='conn_spont'), "log_mean_of_mean_firing_rates_per_second", file_dir + "LP_ALL_EXC_LOG_FRs", *hm_dims)
    c_etl.lineplot(features_by_neuron_class_with_sim_info.etl.q(neuron_class="ALL_INH", window='conn_spont'), "log_mean_of_mean_firing_rates_per_second", file_dir + "LP_ALL_INH_LOG_FRs", *hm_dims)
    c_etl.lineplot(a.custom['custom_simulations_post_analysis'], "ei_corr_rval", file_dir + "LP_ei_corr_rval", *hm_dims)

    comparison_lineplots(a.custom['custom_simulations_post_analysis'], file_dir, *hm_dims)

    print("\n----- Plot parameter effects -----")
    for indep_var in a.analysis_config.custom['independent_variables']:
        for dep_var in ['ei_corr_rval', 'cv_of_neuron_class_proportions_of_vivo', 'std_of_neuron_class_proportions_of_vivo']:
            plt.figure()
            a.custom['custom_simulations_post_analysis'].plot.scatter(indep_var, dep_var, s=1, c='c')
            plt.savefig(str(a.figpaths.parameter_effects) + "/" + indep_var + '__' + dep_var)
            plt.close()
    
def comparison_lineplots(custom_simulations_post_analysis, file_dir, hor_key="none", ver_key="none", x_key="none", colour_var_key="none"):
    hm_dims = (hor_key, ver_key, x_key, colour_var_key)
    for filter_name in ["", "bursting", "bursting_or_fr_above_threshold"]:
        figwidth = 2*len(custom_simulations_post_analysis[hor_key].unique())
        c_etl.lineplot(custom_simulations_post_analysis, "ei_corr_rval", file_dir + "LP_ei_corr_rval", *hm_dims, mask_key=filter_name, marker='', figsize=(figwidth, 2))
        c_etl.lineplot(custom_simulations_post_analysis, "std_of_neuron_class_proportions_of_vivo", file_dir + "LP_std_of_neuron_class_proportions_of_vivo", *hm_dims, mask_key=filter_name, marker='', figsize=(figwidth, 2))
        c_etl.lineplot(custom_simulations_post_analysis, "cv_of_neuron_class_proportions_of_vivo", file_dir + "LP_cv_of_neuron_class_proportions_of_vivo", *hm_dims, mask_key=filter_name, marker='', figsize=(figwidth, 2))
        c_etl.lineplot(custom_simulations_post_analysis, "euc_dist_to_desired_proportion_of_in_vivo_FRs", file_dir + "LP_euc_dist_to_desired_proportion_of_in_vivo_FRs", *hm_dims, mask_key=filter_name, marker='', figsize=(figwidth, 2))
        c_etl.lineplot(custom_simulations_post_analysis, "euc_dist_to_scaled_in_vivo_FRs", file_dir + "LP_euc_dist_to_scaled_in_vivo_FRs", *hm_dims, mask_key=filter_name, marker='', figsize=(figwidth, 2), custom_ylim=[0.0, 3.0])
        c_etl.lineplot(custom_simulations_post_analysis, "difference_between_mean_proportion_and_target_proportion", file_dir + "LP_difference_between_mean_proportion_and_target_proportion", *hm_dims, mask_key=filter_name, marker='', figsize=(figwidth, 2))



def plot_multi_sim_analysis(a):

    print("\n----- Plot firing rates, proportions and predictions -----")

    # sim_stat_line_width = 1.0
    # sim_stat_line_width = 1.5
    sim_stat_line_width = 1.25

    for filter_label, filter_dict in [("", {}), ("NonBursting", {"bursting": False}), ("NonBurstingBelowVivoThreshold", {"bursting_or_fr_above_threshold": False})]:
        # for filter_label, filter_dict in [("NonBurstingBelowVivoThreshold", {"bursting_or_fr_above_threshold": False})]:
            
        c_etl.plot_nc_proportion_of_invivo_for_single_param_set(a, a.custom['custom_features_by_neuron_class'], filter_dict, "SingProportionOfInVivo_" + filter_label)
        c_etl.plot_nc_proportion_of_invivo_for_multiple_sims(a, a.custom['custom_features_by_neuron_class'], filter_dict, "ProportionOfInVivo_" + filter_label)
        c_etl.compare_firing_rates_for_condition_pairs_by_neuron_class(a, a.custom['custom_features_by_neuron_class'], filter_dict)

        neuron_class_groupings_cmaps = ['Blues', 'Reds']
        c_etl.plot_sim_stat_lines_all_sims(a, a.custom['custom_features_by_neuron_class'], "true_mean_conductance", filter_dict, str(a.figpaths.root) + "/" + "MeanConductanceInjection_" + filter_label, c_etl.E_AND_I_SEPERATE_GROUPINGS, neuron_class_groupings_cmaps, show_in_vivo_FRs=False, label_map=c_etl.bluepy_neuron_class_map, figsize=(sim_stat_line_width,3.5))
        c_etl.plot_sim_stat_lines_all_sims(a, a.custom['custom_features_by_neuron_class'], "mean_of_mean_firing_rates_per_second", filter_dict, str(a.figpaths.root) + "/" + "FRs_" + filter_label, c_etl.E_AND_I_SEPERATE_GROUPINGS, neuron_class_groupings_cmaps,  show_in_vivo_FRs=True, custom_x_lims=[0.0, 2.5], label_map=c_etl.bluepy_neuron_class_map, figsize=(sim_stat_line_width,3.5), major_loc=plt.MultipleLocator(2.5), minor_loc=plt.MultipleLocator(0.5))
        c_etl.plot_sim_stat_lines_all_sims(a, a.custom['custom_features_by_neuron_class'], "depol_mean", filter_dict, str(a.figpaths.root) + "/" + "DepolM_" + filter_label, c_etl.E_AND_I_SEPERATE_GROUPINGS, neuron_class_groupings_cmaps,  show_in_vivo_FRs=False, custom_x_lims=[0.0, 20.0], label_map=c_etl.bluepy_neuron_class_map, figsize=(sim_stat_line_width,3.5))
        c_etl.plot_sim_stat_lines_all_sims(a, a.custom['custom_features_by_neuron_class'], "connection_vs_unconn_proportion", filter_dict, str(a.figpaths.root) + "/" + "ConnUnconnProp_" + filter_label, c_etl.E_AND_I_SEPERATE_GROUPINGS, neuron_class_groupings_cmaps,  show_in_vivo_FRs=False, custom_x_lims=[0.0, 5.0], label_map=c_etl.bluepy_neuron_class_map, figsize=(sim_stat_line_width,3.5), major_loc=plt.MultipleLocator(5.0), minor_loc=plt.MultipleLocator(1.0))
        c_etl.plot_sim_stat_lines_all_sims(a, a.custom['layer_wise_features'], "ei_corr_rval", filter_dict, str(a.figpaths.root) + "/" + "LayerWiseEI_" + filter_label, [c_etl.silico_layer_strings[:0:-1]], ['Blues'], show_in_vivo_FRs=False, custom_x_lims=[-0.2, 1.0], show_colorbar=True)
            
    c_etl.plot_sim_stat_lines_all_sims(a, a.custom['custom_features_by_neuron_class'], "desired_unconnected_fr", {}, str(a.figpaths.root) + "/" + "DesiredUnconnectedFRs", c_etl.E_AND_I_SEPERATE_GROUPINGS, neuron_class_groupings_cmaps, show_in_vivo_FRs=True, label_map=c_etl.bluepy_neuron_class_map, figsize=(sim_stat_line_width,3.5))
    c_etl.plot_bursting_ratios(a, a.custom['custom_simulations_post_analysis']['bursting_ratio'])
    c_etl.video_from_image_files(a.custom['custom_simulations_post_analysis']['fft_plot_path'].tolist(), str(a.figpaths.fft_videos / ("fft_video.mp4")))

    heatmaps_and_lineplots(a) 


