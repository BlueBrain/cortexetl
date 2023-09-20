from scipy.fft import rfft, rfftfreq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import linregress
import os
import sys
from functools import partial
from itertools import chain

import cortex_etl as c_etl

def fft_analysis(simulation_windows, fft_plot_path, spiking_hist, hist_bin_size, kernel_sd, window, simulation_id, smoothing_type):

    if (not np.all(spiking_hist == spiking_hist[0])):

        # Number of samples in normalized_tone
        SAMPLE_RATE = 1.0 / hist_bin_size
        DURATION = simulation_windows.etl.q(window=window).iloc[0]["duration"]
        N = int(SAMPLE_RATE * DURATION)

        yf = rfft(spiking_hist)
        xf = rfftfreq(N, hist_bin_size) * 1000.0

        where_freq_less_than_threshold = np.where((xf > 0.0) & (xf < 20.0))[0]

        plt.plot(xf[1:], np.abs(yf)[1:])
        plt.savefig(fft_plot_path)
        plt.close()

        # sim_fft_df = pd.DataFrame({"power": np.abs(yf)[1:10], "freq": np.around(xf, 3)[1:10]})
        sim_fft_df = pd.DataFrame({"power": np.abs(yf)[where_freq_less_than_threshold], "freq": np.around(xf, 3)[where_freq_less_than_threshold]})
        sim_fft_df['log_power'] = np.log(sim_fft_df['power'])
        sim_fft_df['bin_size'] = hist_bin_size
        sim_fft_df['smoothing_type'] = smoothing_type
        sim_fft_df['kernel_sd'] = kernel_sd
        sim_fft_df['simulation_id'] = simulation_id
        return fft_plot_path, sim_fft_df

    else:
        return None, None


def bursting_test(spiking_hist, features_for_sim, neuron_classes):

    min_val = np.min(spiking_hist)
    max_val = np.max(spiking_hist)

    bursting = False
    nc_frs = [features_for_sim.etl.q(neuron_class=nc)['mean_of_mean_firing_rates_per_second'] for nc in neuron_classes]
    mean_of_nc_frs = np.mean(nc_frs)

    bursting_ratio = -1.0

    if ((max_val != 0) & (mean_of_nc_frs > 0.1)):
        bursting_ratio = min_val / max_val
        if ((bursting_ratio >= 0) & (bursting_ratio < 0.125)):
            bursting = True

    return bursting, bursting_ratio


def atleast_one_neuron_group_above_threshold_invivo_proportion_test(spont_features_for_sim, 
                                                                    vivo_frs, 
                                                                    threshold_proportion,
                                                                    neuron_classes):

    atleast_one_neuron_class_fr_greater_than_invivo_thresh = False
    for neuron_class in neuron_classes:
        if (spont_features_for_sim.etl.q(neuron_class=neuron_class)['mean_of_mean_firing_rates_per_second'].iloc[0] > (vivo_frs[neuron_class] * threshold_proportion)):
            atleast_one_neuron_class_fr_greater_than_invivo_thresh = True

    return atleast_one_neuron_class_fr_greater_than_invivo_thresh

def proportion_of_in_vivo_stats(spont_features_for_sim, 
                                vivo_frs,
                                neuron_classes):

    proportions_of_vivo = []
    for neuron_class in neuron_classes:
        proportions_of_vivo.append(spont_features_for_sim.etl.q(neuron_class=neuron_class)['mean_of_mean_firing_rates_per_second'] / vivo_frs[neuron_class])

    return np.std(proportions_of_vivo), abs(np.std(proportions_of_vivo) / np.mean(proportions_of_vivo)), proportions_of_vivo

def euclidean_distance_to_scaled_in_vivo_FRs(spont_features_for_sim, vivo_frs, target_proportion, neuron_classes):
    distances_to_in_vivo = []
    for neuron_class in neuron_classes:
        distances_to_in_vivo.append(spont_features_for_sim.etl.q(neuron_class=neuron_class)['mean_of_mean_firing_rates_per_second'] - target_proportion*vivo_frs[neuron_class])

    return np.linalg.norm(distances_to_in_vivo)




def custom_post_analysis_single_simulation(simulation_row, 
                                            filtered_dataframes, 
                                            analysis_config):

    print("Custom post analysis: ", str(simulation_row['simulation_id']))

    features_with_sim_info = pd.merge(filtered_dataframes['features_by_neuron_class'].reset_index(), simulation_row.to_frame().T)

    spont_window = "conn_spont"; spont_hist_bin_size = 3.0; spont_smoothing_type = 'Gaussian'; spont_kernel_sd = 1.0
    evoked_hist_bin_size = 3.0; evoked_smoothing_type = 'Gaussian'; evoked_kernel_sd = 1.0

    ##### SPONTANEOUS ANALYSIS ##### 
    spont_features_for_sim = features_with_sim_info.etl.q(window=spont_window)
    spont_hists = filtered_dataframes['histograms'].etl.q(window="conn_spont", 
                                                bin_size=spont_hist_bin_size, 
                                                smoothing_type=spont_smoothing_type, 
                                                kernel_sd=spont_kernel_sd)

    _, spont_hist_EXC = c_etl.hist_elements(spont_hists.etl.q(neuron_class="ALL_EXC"))
    _, spont_hist_INH = c_etl.hist_elements(spont_hists.etl.q(neuron_class="ALL_INH"))
    _, spont_hist_ALL = c_etl.hist_elements(spont_hists.etl.q(neuron_class="ALL"))

    r_dict = {'simulation_id': simulation_row['simulation_id'],
            'ei_corr_rval': 0.0,
            'fft_plot_path': ''}

    if (not np.all(spont_hist_EXC == spont_hist_EXC[0])):
        r_dict['ei_corr_rval'] = linregress(spont_hist_EXC, spont_hist_INH).rvalue


    r_dict['ei_corr_r_out_of_range'] = (r_dict['ei_corr_rval'] < analysis_config['ei_corr_r_val_limits'][0]) | (r_dict['ei_corr_rval'] > analysis_config['ei_corr_r_val_limits'][1])

    r_dict['std_of_neuron_class_proportions_of_vivo'], r_dict['cv_of_neuron_class_proportions_of_vivo'], proportions_of_vivo = proportion_of_in_vivo_stats(spont_features_for_sim, analysis_config['vivo_frs'], c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])

    if ('desired_connected_proportion_of_invivo_frs' not in features_with_sim_info.columns):
        features_with_sim_info['desired_connected_proportion_of_invivo_frs'] = 0.5

    target_proportion = features_with_sim_info['desired_connected_proportion_of_invivo_frs'].unique()[0]
    r_dict['euc_dist_to_desired_proportion_of_in_vivo_FRs'] = np.linalg.norm([target_proportion - measured_prop for measured_prop in proportions_of_vivo])
    r_dict['euc_dist_to_scaled_in_vivo_FRs'] = euclidean_distance_to_scaled_in_vivo_FRs(spont_features_for_sim, analysis_config['vivo_frs'], target_proportion, c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])

    r_dict['difference_between_mean_proportion_and_target_proportion'] = target_proportion - np.mean(proportions_of_vivo)

    r_dict['atleast_one_neuron_class_fr_greater_than_invivo_thresh'] = atleast_one_neuron_group_above_threshold_invivo_proportion_test(spont_features_for_sim, analysis_config['vivo_frs'], 1.05, c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])
    r_dict['bursting'], r_dict['bursting_ratio'] = bursting_test(spont_hist_ALL, spont_features_for_sim, c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])
    r_dict['bursting_or_fr_above_threshold'] = r_dict['bursting'] + r_dict['atleast_one_neuron_class_fr_greater_than_invivo_thresh']
    r_dict['bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range'] = r_dict['bursting_or_fr_above_threshold'] + r_dict['ei_corr_r_out_of_range']

    _, FFT_spont_hists = c_etl.hist_elements(filtered_dataframes['FFT_histograms'].etl.q(neuron_class="ALL", window="conn_spont"))

    if (not np.all(FFT_spont_hists == FFT_spont_hists[0])):
        r_dict['fft_plot_path'], r_dict['sim_fft_df'] = fft_analysis(filtered_dataframes['simulation_windows'], 
                                                                    str(simulation_row['fft_dir']) + '/FFT_plot.png', 
                                                                    FFT_spont_hists, 
                                                                    1.0, 
                                                                    -1, 
                                                                    spont_window, 
                                                                    simulation_row['simulation_id'], 
                                                                    spont_smoothing_type)

    ##### EVOKED ANALYSIS #####     
    r_dict['higher_secondary_peak'] = False; r_dict['overly_sustained_response'] = False; r_dict['too_much_trial_to_trial_variance'] = False
    if (('evoked_window_for_custom_post_analysis' in list(analysis_config.keys())) and (analysis_config['evoked_window_for_custom_post_analysis'] in filtered_dataframes['simulation_windows']['window'].unique())):

        _, evoked_100ms_hist_ALL = c_etl.hist_elements(filtered_dataframes['histograms'].etl.q(neuron_class="ALL", 
                                                                                        window=analysis_config['evoked_window_for_custom_post_analysis'], 
                                                                                        bin_size=evoked_hist_bin_size, 
                                                                                        smoothing_type=evoked_smoothing_type, 
                                                                                        kernel_sd=evoked_kernel_sd))

        r_dict['higher_secondary_peak'] = (np.sum(evoked_100ms_hist_ALL[5] < evoked_100ms_hist_ALL[15:]) > 0) | (np.sum(evoked_100ms_hist_ALL[6] < evoked_100ms_hist_ALL[15:]) > 0)

        min_evok = np.min(evoked_100ms_hist_ALL[:6])
        max_evok = np.max(evoked_100ms_hist_ALL[:6])
        threshold_evok = min_evok + 0.5 * (max_evok - min_evok)
        r_dict['30ms_decay_point'] = (evoked_100ms_hist_ALL[10] - min_evok) / (max_evok - min_evok)
        r_dict['60ms_decay_point'] = (evoked_100ms_hist_ALL[20] - min_evok) / (max_evok - min_evok)

        r_dict['overly_sustained_response'] = np.sum(evoked_100ms_hist_ALL[17:] > threshold_evok) > 0

        window_featuress_by_trial = filtered_dataframes['features_by_neuron_class_and_trial'].etl.q(neuron_class="ALL", window=analysis_config['evoked_window_for_custom_post_analysis']).reset_index()
        min_max_mean_count_ratio = window_featuress_by_trial['mean_of_spike_counts_for_each_trial'].min() / window_featuress_by_trial['mean_of_spike_counts_for_each_trial'].max()
        r_dict['too_much_trial_to_trial_variance'] = min_max_mean_count_ratio < 0.5

        r_dict['bursting_or_fr_above_threshold_or_overly_sustained_response'] = r_dict['bursting_or_fr_above_threshold'] + r_dict['overly_sustained_response']
        r_dict['evoked_mask'] = r_dict['higher_secondary_peak'] + r_dict['overly_sustained_response'] + r_dict['too_much_trial_to_trial_variance']

    return r_dict


def layer_wise_single_sim_analysis(simulation_row, 
                                    filtered_dataframes, 
                                    analysis_config):

    r_dicts = []
    for layer_str in c_etl.silico_layer_strings:
        if (layer_str != "L1"):
            r_dict = {'simulation_id': simulation_row['simulation_id'],
                        'neuron_class': layer_str,
                        'ei_corr_rval': 0.0, 
                        "window": "conn_spont"}
            e_layer_str = layer_str + "_EXC"
            i_layer_str = layer_str + "_INH"
            layer_hists = filtered_dataframes['histograms'].etl.q(neuron_class=[e_layer_str, i_layer_str])
            _, spont_layer_hist_EXC = c_etl.hist_elements(layer_hists.etl.q(neuron_class=e_layer_str))
            _, spont_layer_hist_INH = c_etl.hist_elements(layer_hists.etl.q(neuron_class=i_layer_str))

            if (not np.all(spont_layer_hist_EXC == spont_layer_hist_EXC[0])):
                r_dict['ei_corr_rval'] = linregress(spont_layer_hist_EXC, spont_layer_hist_INH).rvalue

            r_dicts.append(r_dict)
    return r_dicts



def load_custom_dataframes(a):
    a.custom = {}
    all_loaded = True
    for key in ['custom_simulations_post_analysis', 'custom_features_by_neuron_class', 'fft', 'layer_wise_features']:
        df_path = str(a.analysis_config['output']) + "/" + key + ".parquet"
        if (os.path.exists(df_path)):
            a.custom[key] = pd.read_parquet(df_path)
        else:
            all_loaded = False


def persist_custom_dataframes(a):

    dump_options = {
            "engine": "pyarrow",
            "index": True,
        }
    for key, df_to_save in a.custom.items():
        save_path = str(a.analysis_config['output']) + "/" + key + ".parquet"
        # df_to_save.select_dtypes(exclude=['object']).to_parquet(path=save_path, **dump_options)
        df_to_save.to_parquet(path=save_path, **dump_options)


from blueetl.parallel import call_by_simulation
from functools import partial
def calculate_layerwise_features(a, spont_hist_bin_size, spont_smoothing_type, spont_kernel_sd):

    dfs = {"simulation_windows": a.repo.windows.df, 
            "histograms": a.features.histograms.df.etl.q(neuron_class=c_etl.LAYER_EI_NEURON_CLASSES, window="conn_spont", bin_size=spont_hist_bin_size, smoothing_type=spont_smoothing_type, kernel_sd=spont_kernel_sd)}
    
    results = call_by_simulation(a.repo.simulations.df, 
                                    dfs, 
                                    func=partial(layer_wise_single_sim_analysis, analysis_config=a.analysis_config),
                                    how='series')
        
    a.custom['layer_wise_features'] = pd.DataFrame.from_records(list(chain.from_iterable(results)))



def add_sim_and_filters_info_to_df(a, custom_df_key):
    
    filter_keys = ['simulation_id', 
                    'ei_corr_r_out_of_range', 
                    'atleast_one_neuron_class_fr_greater_than_invivo_thresh', 
                    'bursting',
                    'bursting_or_fr_above_threshold',
                    'bursting_or_fr_above_threshold_or_ei_corr_r_out_of_range']
    
    a.custom[custom_df_key] = pd.merge(a.custom[custom_df_key], a.repo.simulations.df.loc[:, a.analysis_config.custom['independent_variables'] + ['simulation_id']])
    if (custom_df_key != 'custom_simulations_post_analysis'):
        a.custom[custom_df_key] = pd.merge(a.custom[custom_df_key], a.custom['custom_simulations_post_analysis'].loc[:, filter_keys])
    

def get_v(d, key, second_key=''):
    if (key in list(d.keys())):
        v = d[key]
        if (second_key != ''):
            v = v[second_key]
        return v
    else:
        return np.nan




import bluepy
import bluepysnap
def get_value_from_instance(row, value_key, a, map_to_use=None):
    if isinstance(a.repo.simulations.df.iloc[0].simulation.instance, bluepy.simulation.Simulation):
        if value_key == "desired_connected_fr_key":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config.Run, a.analysis_config.custom['desired_connected_fr_key'] + '_' + c_etl.bluepy_neuron_class_map[row.neuron_class])
        elif value_key == "desired_unconnected_fr_key":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config.Run, a.analysis_config.custom['desired_unconnected_fr_key'] + '_' + c_etl.bluepy_neuron_class_map[row.neuron_class])
        elif value_key == "MeanPercent":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config, "Stimulus_" + a.analysis_config.custom['depol_bc_key'] + '_' + map_to_use[row.neuron_class], second_key='MeanPercent')
        elif value_key == "SDPercent":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config, "Stimulus_" + a.analysis_config.custom['depol_bc_key'] + '_' + map_to_use[row.neuron_class], second_key='SDPercent')

    if isinstance(a.repo.simulations.df.iloc[0].simulation.instance, bluepysnap.simulation.Simulation):
        if value_key == "desired_connected_fr_key":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config, a.analysis_config.custom['desired_connected_fr_key'] + '_' + c_etl.bluepy_neuron_class_map[row.neuron_class])
        elif value_key == "desired_unconnected_fr_key":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config, a.analysis_config.custom['desired_unconnected_fr_key'] + '_' + c_etl.bluepy_neuron_class_map[row.neuron_class])
        elif value_key == "MeanPercent":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config['inputs'], "Stimulus " + a.analysis_config.custom['depol_bc_key'] + '_' + map_to_use[row.neuron_class], second_key='mean_percent')
        elif value_key == "SDPercent":
            return get_v(a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance.config['inputs'], "Stimulus " + a.analysis_config.custom['depol_bc_key'] + '_' + map_to_use[row.neuron_class], second_key='sd_percent')

    

def calculate_true_neuron_class_mean_input_conductance(row, input_conductance_by_neuron_class_df):

    resting_cond = input_conductance_by_neuron_class_df.etl.q(neuron_class=row.neuron_class)['resting_conductance']
    if len(resting_cond):
        return resting_cond.iloc[0] * row['depol_mean'] / 100.0
    else:
        return np.nan

from blueetl.parallel import call_by_simulation
from functools import partial
def post_analysis(a):

    print("\n----- Custom post analysis -----")
    tic = time.perf_counter()

    # load_custom_dataframes()

    a.custom = {}
    
    # DF 1 & 2: custom_simulations_post_analysis & fft
    ######################################################################
    # should be in config
    spont_window = "conn_spont"
    spont_hist_bin_size = 3.0
    spont_smoothing_type = 'Gaussian'
    spont_kernel_sd = 1.0
    evoked_hist_bin_size = 3.0
    evoked_smoothing_type = 'Gaussian'
    evoked_kernel_sd = 1.0
    
    hist_windows = [spont_window]
    if ("evoked_window_for_custom_post_analysis" in list(a.analysis_config.custom.keys())):
        hist_windows.append(a.analysis_config.custom['evoked_window_for_custom_post_analysis'])
    dataframes={"simulation_windows": a.repo.windows.df, 
                "histograms": a.features.histograms.df.etl.q(neuron_class=["ALL", "ALL_EXC", "ALL_INH"], window=hist_windows, bin_size=evoked_hist_bin_size, smoothing_type=evoked_smoothing_type, kernel_sd=evoked_kernel_sd), 
                "FFT_histograms": a.features.histograms.df.etl.q(neuron_class=["ALL"], window=hist_windows, bin_size=1., smoothing_type='None', kernel_sd=-1.),
                "features_by_neuron_class": a.features.by_neuron_class.df.etl.q(window=hist_windows),
                "features_by_neuron_class_and_trial": a.features.by_neuron_class_and_trial.df.etl.q(neuron_class="ALL", window=hist_windows)}

    results = call_by_simulation(a.repo.simulations.df, 
                                    dataframes, 
                                    func=partial(custom_post_analysis_single_simulation, 
                                                    analysis_config=a.analysis_config.custom),
                                    how='series')
    
    a.custom['fft'] = pd.concat([r.pop('sim_fft_df') for r in results])
    a.custom['custom_simulations_post_analysis'] = pd.DataFrame.from_records(results)
    add_sim_and_filters_info_to_df(a, 'fft')
    add_sim_and_filters_info_to_df(a, 'custom_simulations_post_analysis')
    
    # Replace spontaneous activity e/i metric
    if ("spont_replacement_custom_simulation_data_df" in list(a.analysis_config.custom.keys())):
        spont_replacement_df = pd.read_parquet(a.analysis_config.custom['spont_replacement_custom_simulation_data_df'])
        spont_replacement_df = spont_replacement_df.etl.q(ca=a.custom['custom_simulations_post_analysis']['ca'].unique(), depol_stdev_mean_ratio=a.custom['custom_simulations_post_analysis']['depol_stdev_mean_ratio'].unique(), desired_connected_proportion_of_invivo_frs=a.custom['custom_simulations_post_analysis']['desired_connected_proportion_of_invivo_frs'].unique())
        a.custom['custom_simulations_post_analysis'] = pd.merge(a.custom['custom_simulations_post_analysis'].drop(columns=['ei_corr_rval']), spont_replacement_df.loc[:, ['ca', 'depol_stdev_mean_ratio', 'desired_connected_proportion_of_invivo_frs', 'ei_corr_rval']])
    ######################################################################

    
    
    # DF 3: custom_features_by_neuron_class
    ######################################################################
    a.custom['custom_features_by_neuron_class'] = a.features.by_neuron_class.df.reset_index().etl.q(neuron_class=c_etl.__dict__[a.analysis_config.custom['fr_analysis_neuron_classes_constant']]).copy()
    add_sim_and_filters_info_to_df(a, 'custom_features_by_neuron_class')
    a.custom['custom_features_by_neuron_class']["desired_connected_fr"] = a.custom['custom_features_by_neuron_class'].apply(lambda row: get_value_from_instance(row, "desired_connected_fr_key", a), axis = 1).astype(float)
    if ('desired_unconnected_fr' in [x for xs in a.analysis_config.custom['fr_comparison_pairs'] for x in xs]):
        a.custom['custom_features_by_neuron_class']['desired_unconnected_fr'] = a.custom['custom_features_by_neuron_class'].apply(lambda row: get_value_from_instance(row, "desired_unconnected_fr_key", a), axis = 1).astype(float)
        a.custom['custom_features_by_neuron_class']['connection_fr_increase'] = a.custom['custom_features_by_neuron_class']['mean_of_mean_firing_rates_per_second'] - a.custom['custom_features_by_neuron_class']['desired_unconnected_fr']
        a.custom['custom_features_by_neuron_class']['connection_fr_error'] = a.custom['custom_features_by_neuron_class']['mean_of_mean_firing_rates_per_second'] - a.custom['custom_features_by_neuron_class']['desired_connected_fr']
        a.custom['custom_features_by_neuron_class']['connection_vs_unconn_proportion'] = a.custom['custom_features_by_neuron_class']['mean_of_mean_firing_rates_per_second'] / a.custom['custom_features_by_neuron_class']['desired_unconnected_fr']
    map_to_use = c_etl.bluepy_neuron_class_map_2
    if (a.analysis_config.custom['depol_bc_key'] == "RelativeShotNoise"):
        map_to_use = c_etl.bluepy_neuron_class_map
    a.custom['custom_features_by_neuron_class']['recorded_proportion_of_in_vivo_FR'] = a.custom['custom_features_by_neuron_class'].apply(lambda row: a.features.by_neuron_class.df.etl.q(simulation_id=row.simulation_id, neuron_class=row.neuron_class, window=row.window).iloc[0]['mean_of_mean_firing_rates_per_second'] / a.analysis_config.custom['vivo_frs'][row.neuron_class], axis = 1).astype(float)
    a.custom['custom_features_by_neuron_class']["depol_mean"] = a.custom['custom_features_by_neuron_class'].apply(lambda row: get_value_from_instance(row, "MeanPercent", a, map_to_use=map_to_use), axis = 1).astype(float)
    a.custom['custom_features_by_neuron_class']["depol_sd"] = a.custom['custom_features_by_neuron_class'].apply(lambda row: get_value_from_instance(row, "SDPercent", a, map_to_use=map_to_use), axis = 1).astype(float)
    input_conductance_by_neuron_class_df = pd.read_parquet(a.analysis_config.custom['input_conductance_by_neuron_class_parquet'])
    # OLD "input_conductance_by_neuron_class_parquet": '/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/blueetl_ji_analyses/data/input_conductance_by_neuron_class.parquet'
    a.custom['custom_features_by_neuron_class']["true_mean_conductance"] = a.custom['custom_features_by_neuron_class'].apply(lambda row: calculate_true_neuron_class_mean_input_conductance(row, input_conductance_by_neuron_class_df), axis = 1).astype(float)
    ######################################################################

    
    # DF 4: layer_wise_features
    ######################################################################
    calculate_layerwise_features(a, spont_hist_bin_size, spont_smoothing_type, spont_kernel_sd)
    add_sim_and_filters_info_to_df(a, 'layer_wise_features')
    ######################################################################

    # persist_custom_dataframes(a)
    
    
    print(f"----- Custom post analysis complete: {time.perf_counter() - tic:0.2f}s -----")

    