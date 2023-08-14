import random
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from blueetl.parallel import call_by_simulation
from functools import partial
import cortex_etl as c_etl


def plot_histogram(values, nbins, hist_range, filename):
	plt.figure()
	plt.hist(values, bins=nbins, range=hist_range)
	plt.savefig(filename)
	plt.close()



def create_pairwise_combinations_of_gids(gids, max_number_of_pairs=-1):

	gids = random.shuffle(gids)

	pairwise_gid_combinations = list(itertools.combinations(gids, 2))
	random.shuffle(pairwise_gid_combinations)

	if (max_number_of_pairs == -1):
		max_number_of_pairs = len(pairwise_gid_combinations)
	else:
		max_number_of_pairs = min([max_number_of_pairs, len(pairwise_gid_combinations)])

	pairwise_gid_combinations = pairwise_gid_combinations[:max_number_of_pairs - 1, :]

	return pairwise_gid_combinations



def single_simulation_spike_pair_analysis(simulation_row, filtered_dataframes, analysis_config):

	# simulation_row_df = pd.Series(simulation_row._asdict()).to_frame().T
	simulation_id = simulation_row['simulation_id']

	r_dict = {}
	r_dict['mean_pairwise_first_spike_r_value'] = np.nan

	print("simulation_id: " + str(simulation_id))

	all_ps = []
	all_rs = []

	# corrected_target = analysis_config['extraction']['target'].replace(".", "_" )
	# target_sim_str = corrected_target + "_" + str(simulation_id) + '_'
	target_sim_str = str(simulation_id) + '_'

	for neuron_class in c_etl.LAYER_EI_NEURON_CLASSES:
		# print(neuron_class)

		window_and_neuron_class_df = filtered_dataframes['features_by_gid_and_trial'].etl.q(neuron_class=neuron_class).reset_index()
		
		neuron_class_gids = window_and_neuron_class_df['gid'].unique()

		num_spiking_trials_by_gid = window_and_neuron_class_df.groupby('gid')['first'].count()
		num_spiking_trials_by_gid_thresholded = num_spiking_trials_by_gid[num_spiking_trials_by_gid > analysis_config['theshold_conjunctive_trials_for_spike_pair_analysis']].reset_index()

		# print(num_spiking_trials_by_gid)

		combinations_of_gids_with_theshold_spiking_trials = list(itertools.combinations(num_spiking_trials_by_gid_thresholded['gid'], 2))
		random.shuffle(combinations_of_gids_with_theshold_spiking_trials)

		max_number_of_pairs_to_process = len(combinations_of_gids_with_theshold_spiking_trials)
		# max_number_of_pairs_to_process=1000

		print(max_number_of_pairs_to_process)

		correlation_p_values = []
		correlation_r_values = []

		for comb_ind, (gid_0, gid_1) in enumerate(combinations_of_gids_with_theshold_spiking_trials):
			# print(comb_ind)
			if (comb_ind < max_number_of_pairs_to_process):
				fss_0 = np.asarray(window_and_neuron_class_df.etl.q(gid=gid_0)['first'])
				fss_1 = np.asarray(window_and_neuron_class_df.etl.q(gid=gid_1)['first'])
				true_where_both_spike = np.logical_and(np.logical_not(np.isnan(fss_0)), np.logical_not(np.isnan(fss_1)))


				if (np.sum(true_where_both_spike) > analysis_config['theshold_conjunctive_trials_for_spike_pair_analysis']):

					lr = process_pair_of_neurons(fss_0, fss_1, gid_0, gid_1, make_pairwise_plot=False)
					correlation_p_values.append(lr.pvalue)
					all_ps.append(lr.pvalue)

					if (lr.pvalue<0.05):
						correlation_r_values.append(lr.rvalue)
						all_rs.append(lr.rvalue)

		if (correlation_p_values != []):
			plot_histogram(correlation_p_values, 20, [0.0, 1.0], target_sim_str + 'correlation_p_values_' + neuron_class)
			plot_histogram(correlation_r_values, 40, [-1.0, 1.0], target_sim_str + 'correlation_r_values_' + neuron_class)

	if (all_ps != []):
		plot_histogram(all_ps, 20, [0.0, 1.0], target_sim_str + 'all_ps')
		plot_histogram(all_rs, 20, [-1.0, 1.0], target_sim_str + 'all_rs')

		r_dict['mean_pairwise_first_spike_r_value'] = np.mean(all_rs)
		
		
	return r_dict



def spike_pair_analysis(a):

	print("spike_pair_analysis")

	dataframes={"features_by_gid_and_trial": a.features.by_gid_and_trial.df.etl.q(neuron_class=c_etl.LAYER_EI_NEURON_CLASSES, window='evoked_SOZ_25ms').reset_index()}
	results = call_by_simulation(a.repo.simulations.df, 
									dataframes, 
									func=partial(single_simulation_spike_pair_analysis, analysis_config=a.analysis_config.custom),
                                    how='series')

	mean_pairwise_first_spike_r_values = []
	for r_dict in results:
		mean_pairwise_first_spike_r_values.append(r_dict['mean_pairwise_first_spike_r_value'])

	a.custom['custom_simulations_post_analysis']['mean_pairwise_first_spike_r_value'] = mean_pairwise_first_spike_r_values





def process_pair_of_neurons(fss_0, fss_1, gid_0, gid_1, make_pairwise_plot=False):

	true_where_both_spike = np.logical_and(np.logical_not(np.isnan(fss_0)), np.logical_not(np.isnan(fss_1)))
	lr = linregress(fss_0[true_where_both_spike], fss_1[true_where_both_spike])

	if (make_pairwise_plot):
		if (lr.pvalue < 0.05):

			plt.figure()
			plt.scatter(fss_0, fss_1)
			plt.gca().set_aspect('equal', adjustable='box')
			
			lims = [np.min([np.nanmin(fss_0), np.nanmin(fss_1)]) - 0.5, np.max([np.nanmax(fss_0), np.nanmax(fss_1)]) + 0.5]
			plt.gca().plot(lims, lims, 'k-', alpha=0.75, zorder=0)
			plt.gca().set_xlabel(lims)
			plt.gca().set_ylabel(lims)

			plt.savefig(str(gid_0) + "_" + str(gid_1) + '.png')
			plt.close()

	return lr