import cortex_etl as c_etl

def print_coupled_coords_from_mask(a, mask_key):

	all_cas = []
	all_fr_scales = []
	all_depol_stdev_mean_ratios = []
	all_vpm_pcts = []
	all_pom_pcts = []
	all_vpm_l5e_cond_scaling_factors = []

	thalam_pct_pairs = [[2.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]]
	vpm_l5e_cond_scaling_factors = [4.0] # 0.0, 

	q = {mask_key: False, 'ca':[1.05, 1.1], 'depol_stdev_mean_ratio': [0.2, 0.3], 'desired_connected_proportion_of_invivo_frs': [0.1, 0.3, 0.5, 0.7, 0.9]}  #, "ei_corr_rval": {"ge": 3, "lt": 8}
	sims_df = a.custom['custom_simulations_post_analysis'].etl.q(q)


	cas = sims_df['ca'].tolist()
	fr_scales = sims_df['desired_connected_proportion_of_invivo_frs'].tolist()
	depol_stdev_mean_ratios = sims_df['depol_stdev_mean_ratio'].tolist()
	for vpm_l5e_cond_scaling_factor in vpm_l5e_cond_scaling_factors:
		for thalam_pct_pair in thalam_pct_pairs:
			all_cas.extend(cas)
			all_fr_scales.extend(fr_scales)
			all_depol_stdev_mean_ratios.extend(depol_stdev_mean_ratios)
			all_vpm_pcts.extend([thalam_pct_pair[0] for i in cas])
			all_pom_pcts.extend([thalam_pct_pair[1] for i in cas])
			all_vpm_l5e_cond_scaling_factors.extend([vpm_l5e_cond_scaling_factor for i in cas])

	print(len(all_cas))

	coords_str = 'coords: {\n'
	coords_str += '\"ca\": ' + str(all_cas) + ',\n'
	coords_str += '\"desired_connected_proportion_of_invivo_frs\": ' + str(all_fr_scales) + ',\n'
	coords_str += '\"depol_stdev_mean_ratio\": ' + str(all_depol_stdev_mean_ratios) + ',\n'
	coords_str += '\"vpm_pct\": ' + str(all_vpm_pcts) + ',\n'
	coords_str += '\"pom_pct\": ' + str(all_pom_pcts) + ',\n'
	coords_str += '\"vpm_l5e_cond_scaling_factor\": ' + str(all_vpm_l5e_cond_scaling_factors)
	coords_str += '}'

	print(coords_str)


import pandas as pd
def extract_fr_df(a):

	custom_by_neuron_class_df = pd.merge(a.custom['custom_features_by_neuron_class'].reset_index().drop(["index"], axis=1), a.custom['custom_simulations_post_analysis'].reset_index().drop(["index"], axis=1), on=['simulation_id', 'circuit_id'])

	sim_vars = ['simulation_id', 'window', 'neuron_class', 'bursting', 'bursting_or_fr_above_threshold', 'desired_connected_fr', 'desired_unconnected_fr', 'mean_of_mean_firing_rates_per_second']
	sim_vars.extend(a.analysis_config.custom['independent_variables']) 

	fr_df = custom_by_neuron_class_df.loc[:, sim_vars].etl.q(neuron_class=c_etl.LAYER_EI_NEURON_CLASSES, window=a.analysis_config.custom['fr_df_windows']).drop(['simulation_id'], axis=1)

	fr_df.to_parquet(a.analysis_config.custom['output'] / a.analysis_config.custom['fr_df_name'])