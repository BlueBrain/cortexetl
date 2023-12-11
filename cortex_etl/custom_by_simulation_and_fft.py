from scipy.fft import rfft, rfftfreq
from functools import partial
from blueetl.parallel import call_by_simulation
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cortex_etl as c_etl


def dist_target_FRs(spont_features_for_sim, 
					vivo_frs, 
					target_proportion, 
					ncs):
    dists = []
    for nc in ncs:
        dists.append(spont_features_for_sim.etl.q(neuron_class=nc)['mean_of_mean_firing_rates_per_second'] - target_proportion*vivo_frs[nc])
    return np.linalg.norm(dists)


def proportion_of_in_vivo_stats(spont_features_for_sim, 
                                vivo_frs,
                                ncs):

    proportions_of_vivo = []
    for nc in ncs:
        proportions_of_vivo.append(spont_features_for_sim.etl.q(neuron_class=nc)['mean_of_mean_firing_rates_per_second'] / vivo_frs[nc])

    return np.std(proportions_of_vivo), abs(np.std(proportions_of_vivo) / np.mean(proportions_of_vivo)), proportions_of_vivo



def bursting_test(spiking_hist, features_for_sim, ncs):

    min_val = np.min(spiking_hist)
    max_val = np.max(spiking_hist)

    bursting = False
    nc_frs = [features_for_sim.etl.q(neuron_class=nc)['mean_of_mean_firing_rates_per_second'] for nc in ncs]
    mean_of_nc_frs = np.mean(nc_frs)

    bursting_ratio = -1.0

    if ((max_val != 0) & (mean_of_nc_frs > 0.1)):
        bursting_ratio = min_val / max_val
        if ((bursting_ratio >= 0) & (bursting_ratio < 0.125)):
            bursting = True

    return bursting, bursting_ratio


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

def neuron_group_gt_threshold_fr(spont_features_for_sim, 
                                    vivo_frs, 
                                    thresh_prop,
                                    ncs):

    above = False
    for nc in ncs:
        if spont_features_for_sim.etl.q(neuron_class=nc)['mean_of_mean_firing_rates_per_second'].iloc[0] > (vivo_frs[nc] * thresh_prop):
            above = True

    return above



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
    r_dict['euc_dist_to_scaled_in_vivo_FRs'] = dist_target_FRs(spont_features_for_sim, analysis_config['vivo_frs'], target_proportion, c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])

    r_dict['difference_between_mean_proportion_and_target_proportion'] = target_proportion - np.mean(proportions_of_vivo)

    r_dict['neuron_group_gt_threshold_fr'] = neuron_group_gt_threshold_fr(spont_features_for_sim, analysis_config['vivo_frs'], 1.05, c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])
    r_dict['bursting'], r_dict['bursting_ratio'] = bursting_test(spont_hist_ALL, spont_features_for_sim, c_etl.__dict__[analysis_config['fr_analysis_neuron_classes_constant']])
    r_dict['bursting_or_fr_gt_threshold'] = r_dict['bursting'] + r_dict['neuron_group_gt_threshold_fr']
    r_dict['bursting_or_fr_gt_threshold_or_ei_corr_r_out_of_range'] = r_dict['bursting_or_fr_gt_threshold'] + r_dict['ei_corr_r_out_of_range']

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

        r_dict['bursting_or_fr_gt_threshold_or_overly_sustained_response'] = r_dict['bursting_or_fr_gt_threshold'] + r_dict['overly_sustained_response']
        r_dict['evoked_mask'] = r_dict['higher_secondary_peak'] + r_dict['overly_sustained_response'] + r_dict['too_much_trial_to_trial_variance']

    return r_dict



def custom_by_simulation_features(a):

	evoked_hist_bin_size = 3.0
	evoked_smoothing_type = 'Gaussian'
	evoked_kernel_sd = 1.0
	hist_windows = ['conn_spont']
	if ('evoked_window_for_custom_post_analysis' in list(a.analysis_config.custom.keys())):
		hist_windows += [a.analysis_config.custom['evoked_window_for_custom_post_analysis']]

	dataframes={"simulation_windows": a.repo.windows.df, 

	            "histograms": a.features.histograms.df.etl.q(neuron_class=["ALL", "ALL_EXC", "ALL_INH"], 
	                                                        window=hist_windows, 
	                                                        bin_size=evoked_hist_bin_size, 
	                                                        smoothing_type=evoked_smoothing_type, 
	                                                        kernel_sd=evoked_kernel_sd), 

	            "FFT_histograms": a.features.histograms.df.etl.q(neuron_class=["ALL"], 
	                                                            window=hist_windows, 
	                                                            bin_size=1., 
	                                                            smoothing_type='None', 
	                                                            kernel_sd=-1.),

	            "features_by_neuron_class": a.features.by_neuron_class.df.etl.q(window=hist_windows),
	            "features_by_neuron_class_and_trial": a.features.by_neuron_class_and_trial.df.etl.q(neuron_class="ALL", 
	                                                                                                window=hist_windows)}

	results = call_by_simulation(a.repo.simulations.df, 
	                                dataframes, 
	                                func=partial(custom_post_analysis_single_simulation, 
	                                                analysis_config=a.analysis_config.custom),
	                                how='series')


	a.custom['fft'] = pd.concat([r.pop('sim_fft_df') for r in results])
	a.custom['by_simulation'] = pd.DataFrame.from_records(results)
	c_etl.add_sim_and_filters_info_to_df(a, 'fft')
	c_etl.add_sim_and_filters_info_to_df(a, 'by_simulation')

	# Replace spontaneous activity e/i metric
	if ("spont_replacement_custom_simulation_data_df" in list(a.analysis_config.custom.keys())):
	    spont_replacement_df = pd.read_parquet(a.analysis_config.custom['spont_replacement_custom_simulation_data_df'])
	    spont_replacement_df = spont_replacement_df.etl.q(ca=a.custom['by_simulation']['ca'].unique(), 
	                                                    depol_stdev_mean_ratio=a.custom['by_simulation']['depol_stdev_mean_ratio'].unique(), 
	                                                    desired_connected_proportion_of_invivo_frs=a.custom['by_simulation']['desired_connected_proportion_of_invivo_frs'].unique())

	    a.custom['by_simulation'] = pd.merge(a.custom['by_simulation'].drop(columns=['ei_corr_rval']), spont_replacement_df.loc[:, ['ca', 'depol_stdev_mean_ratio', 'desired_connected_proportion_of_invivo_frs', 'ei_corr_rval']])



