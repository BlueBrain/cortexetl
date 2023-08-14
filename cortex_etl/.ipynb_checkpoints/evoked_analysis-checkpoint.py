import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import blueetl as etl
from scipy.interpolate import interp1d
import seaborn as sns
from matplotlib import cm
import sys
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import os
import cortex_etl as c_etl
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3,
            "axes.titlesize": 7, "axes.spines.right": False, "axes.spines.top": False})

def load_and_preprocess_vivo_dataframes():

	rp_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/reyes-multi-sigma-features.parquet")
	rp_df = rp_df.pivot_table('decay', ['neuron_class', 'bin_size', 'sigma', 'latency'], 'ratio')
	rp_df = rp_df.reset_index(level=['latency'])
	# rp_df = rp_df + rp_offset
	rp_df = rp_df.reset_index()
	rp_df = rp_df.rename(columns={"latency": '1.0', 0.75: '0.75', 0.5: '0.5', 0.25: '0.25'})
	rp_df['experiment'] = 'ReyesPuerta'
	
	svo_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/svoboda-multi-sigma-features.parquet")
	svo_df = svo_df.pivot_table('decay', ['neuron_class', 'bin_size', 'sigma', 'latency'], 'ratio')
	svo_df = svo_df.reset_index(level=['latency'])
	# svo_df = svo_df + svo_offset
	svo_df = svo_df.reset_index()
	svo_df = svo_df.rename(columns={"latency": '1.0', 0.75: '0.75', 0.5: '0.5', 0.25: '0.25'})
	svo_df['experiment'] = 'YuSvoboda'
	vivo_df = pd.concat([rp_df, svo_df])

	rp_psth_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/reyes-multi-sigma-PSTH.parquet")
	svo_psth_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/svoboda-multi-sigma-PSTH.parquet")

	return vivo_df, rp_psth_df, svo_psth_df


def preprocess_silico_dataframes(a):

	latency_df = a.features.latency.set_index(['simulation_id', 'circuit_id', 'window', 'neuron_class', 'bin_size', 'sigma'])
	decay_df = a.features.decay.pivot_table('decay', ['simulation_id', 'circuit_id', 'window', 'neuron_class', 'bin_size', 'sigma'], 'ratio')
	decay_df = latency_df.join(decay_df)
	decay_df = decay_df.rename(columns={"latency": '1.0', 0.75: '0.75', 0.5: '0.5', 0.25: '0.25'}).reset_index()

	# sim_df = decay_df.etl.q(neuron_class='ALL', bin_size=1, sigma=1).reset_index().drop(columns=['bin_size', 'sigma', 'circuit_id'])
	# simulations = a.repo.simulations.df.reset_index().drop(columns=['index', 'simulation_path', 'circuit_id'])
	# sim_df = pd.merge(simulations, sim_df, on=['simulation_id']).reset_index()
	sim_df = None

	return decay_df, latency_df, a.features.baseline, sim_df

def compare_to_in_vivo(rp_vivo_df, svoboda_vivo_df, row, stat):

	vivo_query = []
	if (row['neuron_class'] in rp_vivo_df.neuron_class.unique()):
		vivo_query = rp_vivo_df.etl.q(neuron_class=row['neuron_class'], bin_size=1.0, sigma=1)
	elif (row['neuron_class'] in svoboda_vivo_df.neuron_class.unique()):
		vivo_query = svoboda_vivo_df.etl.q(neuron_class=row['neuron_class'], bin_size=1.0, sigma=1)
	# if (row['neuron_class'] in rp_vivo_df.neuron_class.unique()):
	# 	vivo_query = rp_vivo_df.etl.q(neuron_class=row['neuron_class'], bin_size=float(row['bin_size']), sigma=row['sigma'])
	# elif (row['neuron_class'] in svoboda_vivo_df.neuron_class.unique()):
	# 	vivo_query = svoboda_vivo_df.etl.q(neuron_class=row['neuron_class'], bin_size=float(row['bin_size']), sigma=row['sigma'])
	if (len(vivo_query)):
		return row[stat] - vivo_query.iloc[0][stat]
	return -100000.0

def overly_sustained_test(a, row, time_from, threshold_prop):

	return np.sum(a.features.s3_baseline_PSTH.df.reset_index().etl.q({"simulation_id": row['simulation_id'], "window":a.analysis_config["evoked_window_for_custom_post_analysis"], "neuron_class":row['neuron_class'], "time": {"ge": time_from}})['psth'] > threshold_prop)

def overly_sustained_mask(a, row):

	return (overly_sustained_test(a, row, 75, 0.35) | overly_sustained_test(a, row, 100, 0.4))



def compare_neuron_classes_with_in_vivo(decay_df, vivo_df, a):

	# nc_comparison_decay_df = decay_df.etl.q(bin_size=1.0, sigma=1).reset_index()
	nc_comparison_decay_df = decay_df.etl.q(bin_size=0.5, sigma=2).reset_index()

	# import pdb; pdb.set_trace()

	rp_vivo_df = vivo_df.etl.q(experiment='ReyesPuerta')
	svoboda_vivo_df = vivo_df.etl.q(experiment='YuSvoboda')

	nc_comparison_decay_df.loc[:, '100pDiffToRP'] = nc_comparison_decay_df.apply(lambda row: compare_to_in_vivo(rp_vivo_df, svoboda_vivo_df, row, "1.0"), axis = 1)
	nc_comparison_decay_df.loc[:, '75pDiffToRP'] = nc_comparison_decay_df.apply(lambda row: compare_to_in_vivo(rp_vivo_df, svoboda_vivo_df, row, "0.75"), axis = 1)
	nc_comparison_decay_df.loc[:, '50pDiffToRP'] = nc_comparison_decay_df.apply(lambda row: compare_to_in_vivo(rp_vivo_df, svoboda_vivo_df, row, "0.5"), axis = 1)
	nc_comparison_decay_df.loc[:, '25pDiffToRP'] = nc_comparison_decay_df.apply(lambda row: compare_to_in_vivo(rp_vivo_df, svoboda_vivo_df, row, "0.25"), axis = 1)

	nc_comparison_decay_df.loc[:, 'OverlySustained'] = nc_comparison_decay_df.apply(lambda row: overly_sustained_mask(a, row), axis = 1).astype('bool')

	nc_comparison_decay_df['Bad100pDecay'] = ((nc_comparison_decay_df["1.0"] == -1.0) | (nc_comparison_decay_df["100pDiffToRP"] > 10.0)).astype('bool')
	nc_comparison_decay_df['Bad50pDecay'] = ((nc_comparison_decay_df["0.5"] == -1.0) | (nc_comparison_decay_df["50pDiffToRP"] > 10.0)).astype('bool')
	nc_comparison_decay_df['Bad25pDecay'] = ((nc_comparison_decay_df["0.25"] == -1.0) | (nc_comparison_decay_df["25pDiffToRP"] > 40.0)).astype('bool')

	nc_comparison_decay_df['Bad50p25pDecay'] = (nc_comparison_decay_df['Bad50pDecay'] | nc_comparison_decay_df['Bad25pDecay'])
	nc_comparison_decay_df['Bad100p50p25pDecay'] = (nc_comparison_decay_df['Bad100pDecay'] | nc_comparison_decay_df['Bad50pDecay'] | nc_comparison_decay_df['Bad25pDecay'])
	nc_comparison_decay_df['Bad100p50p25pDecayOverlySustained'] = (nc_comparison_decay_df['Bad100p50p25pDecay'] | nc_comparison_decay_df['OverlySustained'])

	return nc_comparison_decay_df


def create_sim_mask(nc_comparison_decay_df, sim_df, neuron_classes):

	filtered_nc_comparison_decay_df = nc_comparison_decay_df.etl.q(neuron_class=neuron_classes)

	sim_mask_df = filtered_nc_comparison_decay_df.groupby(by=['simulation_id', 'window'])['Bad25pDecay', 'Bad50pDecay', 'Bad100pDecay', 'Bad50p25pDecay', 'Bad100p50p25pDecay', 'Bad100p50p25pDecayOverlySustained'].sum().astype('bool')
	sim_mask_df = sim_mask_df.rename(columns={"Bad25pDecay": "SimBad25pDecay", "Bad50pDecay": "SimBad50pDecay", "Bad100pDecay": "SimBad100pDecay", "Bad50p25pDecay": "SimBad50p25pDecay", "Bad100p50p25pDecay": "SimBad100p50p25pDecay", "Bad100p50p25pDecayOverlySustained":"SimBad100p50p25pDecayOverlySustained"}).reset_index()

	sim_mask_df = pd.merge(sim_df, sim_mask_df, on=['simulation_id']).drop(columns=[])
	return sim_mask_df


def merge_comparison_and_mask(nc_comparison_decay_df, sim_mask_df):
	nc_stat_and_mask_df = pd.merge(nc_comparison_decay_df, sim_mask_df, on=['simulation_id', 'window'])
	return nc_stat_and_mask_df


def create_heatmaps(a, window_sim_mask_df, all_window_stat_and_mask_df, masks_dir, heatmaps_dir):
	hm_dims = (a.analysis_config['heatmap_dims']['hor_key'], a.analysis_config['heatmap_dims']['ver_key'], a.analysis_config['heatmap_dims']['x_key'], a.analysis_config['heatmap_dims']['y_key'])

	c_etl.heatmap(window_sim_mask_df, "SimBad25pDecay", masks_dir + "SimBad25pDecay", *hm_dims)
	c_etl.heatmap(window_sim_mask_df, "SimBad50pDecay", masks_dir + "SimBad50pDecay", *hm_dims)
	c_etl.heatmap(window_sim_mask_df, "SimBad100pDecay", masks_dir + "SimBad100pDecay", *hm_dims)
	c_etl.heatmap(window_sim_mask_df, "SimBad50p25pDecay", masks_dir + "SimBad50p25pDecay", *hm_dims)
	c_etl.heatmap(window_sim_mask_df, "SimBad100p50p25pDecay", masks_dir + "SimBad100p50p25pDecay", *hm_dims)
	c_etl.heatmap(window_sim_mask_df, "SimBad100p50p25pDecayOverlySustained", masks_dir + "SimBad100p50p25pDecayOverlySustained", *hm_dims)
	c_etl.heatmap(window_sim_mask_df, "SimBad25pDecay", masks_dir + "SimBad25pDecay", *hm_dims, mask_key='SimBad100pDecay')
	c_etl.heatmap(window_sim_mask_df, "SimBad50p25pDecay", masks_dir + "SimBad50p25pDecay", *hm_dims, mask_key='SimBad100pDecay')
	c_etl.heatmap(window_sim_mask_df, "SimBad100p50p25pDecay", masks_dir + "SimBad100p50p25pDecay", *hm_dims, mask_key='SimBad100p50p25pDecay')

	c_etl.heatmap(all_window_stat_and_mask_df, "mean_ratio_difference", heatmaps_dir + "mean_ratio_difference", *hm_dims, mask_key='SimBad100p50p25pDecayOverlySustained')

	# c_etl.heatmap(all_window_stat_and_mask_df, "0.5", heatmaps_dir + "0.5", *hm_dims)
	# c_etl.heatmap(all_window_stat_and_mask_df, "0.5", heatmaps_dir + "0.5", *hm_dims, mask_key='SimBad50p25pDecay', highlight_false_key='SimBad50pDecay')

	# c_etl.heatmap(all_window_stat_and_mask_df, "100pDiffToRP", heatmaps_dir + "100pDiffToRP", *hm_dims)
	# c_etl.heatmap(all_window_stat_and_mask_df, "100pDiffToRP", heatmaps_dir + "100pDiffToRP", *hm_dims, mask_key='SimBad50pDecay', highlight_false_key='SimBad50p25pDecay')

	# c_etl.heatmap(all_window_stat_and_mask_df, "50pDiffToRP", heatmaps_dir + "50pDiffToRP", *hm_dims)
	# c_etl.heatmap(all_window_stat_and_mask_df, "50pDiffToRP", heatmaps_dir + "50pDiffToRP", *hm_dims, mask_key='SimBad50pDecay', highlight_false_key='SimBad50p25pDecay')

	# c_etl.heatmap(all_window_stat_and_mask_df, "25pDiffToRP", heatmaps_dir + "25pDiffToRP", *hm_dims)
	# c_etl.heatmap(all_window_stat_and_mask_df, "25pDiffToRP", heatmaps_dir + "25pDiffToRP", *hm_dims, mask_key='SimBad50pDecay', highlight_false_key='SimBad50p25pDecay')


def layer_and_pairwise_comparison(nc_stat_and_mask_df, vivo_df, vivo_exp, decay, nc_pairs, path, xlim, xticks):

	flattened_ncs = [item for sublist in nc_pairs for item in sublist]

	plt.figure(plt.figure(figsize=(8.2*0.2, 8.2*0.2)))

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



def compare_time_courses(nc_stat_and_mask_df, vivo_df, figs_dir):

	nc_stat_and_mask_df['layer'] = nc_stat_and_mask_df.apply(lambda row: row['neuron_class'].split('_')[0], axis = 1)

	for decay in ['1.0']:

		layer_and_pairwise_comparison(nc_stat_and_mask_df, vivo_df, "ReyesPuerta", decay, [["L6_INH", "L6_EXC"], ["L5_INH", "L5_EXC"], ["L4_INH", "L4_EXC"], ["L23_INH", "L23_EXC"]], figs_dir + "ReyesPuertaEI_latencies.pdf", [0, 11], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
		layer_and_pairwise_comparison(nc_stat_and_mask_df, vivo_df, "YuSvoboda", decay, [["L6_PV", "L6_SST"], ["L5_PV", "L5_SST"], ["L4_PV", "L4_SST"], ["L23_PV", "L23_SST"]], figs_dir + "ReyesPuertaPVSST_latencies.pdf", [0, 19], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])

		compare_metrics_in_vivo_vs_in_silico(nc_stat_and_mask_df, vivo_df, 'ReyesPuerta', ['L23', 'L4', 'L5'], ["EXC", "INH"], decay, "T", figs_dir, upper_lim=15.0, mask_key='SimBad100p50p25pDecayOverlySustained')
		compare_metrics_in_vivo_vs_in_silico(nc_stat_and_mask_df, vivo_df, 'YuSvoboda', ['L23', 'L4', 'L5', 'L6'], ["EXC", "INH"], decay, "T", figs_dir, upper_lim=15.0, mask_key='SimBad100p50p25pDecayOverlySustained')
		compare_metrics_in_vivo_vs_in_silico(nc_stat_and_mask_df, vivo_df, 'YuSvoboda', ['L23', 'L4', 'L5', 'L6'], ["PV", "SST"], decay, "T", figs_dir, upper_lim=20.0, mask_key='SimBad100p50p25pDecayOverlySustained')

		compare_metrics_of_two_layerwise_groups(nc_stat_and_mask_df, vivo_df, ['ReyesPuerta'], ['L23', 'L4', 'L5', 'L6'], "EXC", "INH", decay, "ReyesPuerta", figs_dir, mask_key='SimBad100p50p25pDecayOverlySustained')
		compare_metrics_of_two_layerwise_groups(nc_stat_and_mask_df, vivo_df, ['YuSvoboda'], ['L23', 'L4', 'L5', 'L6'], "EXC", "INH", decay, "YuSvoboda", figs_dir, mask_key='SimBad100p50p25pDecayOverlySustained')
		compare_metrics_of_two_layerwise_groups(nc_stat_and_mask_df, vivo_df, ['YuSvoboda'], ['L23', 'L4', 'L5', 'L6'], "PV", "SST", decay, "YuSvoboda", figs_dir, mask_key='SimBad100p50p25pDecayOverlySustained')



def compare_metrics_in_vivo_vs_in_silico(df, vivo_df, vivo_exp, layers, groups, stat, prefix, figs_dir, upper_lim=20.0, mask_key=''):

	if (mask_key != ''):
		mask_dict = {mask_key:False}
		df = df.etl.q(mask_dict)

	cmap = cm.get_cmap('rocket', 5)
	colors = cmap(np.linspace(0, 1, 5))

	layer_colours = {
				"L23": colors[0],
				"L4": colors[1],
				"L5": colors[2],
				"L6": colors[3]}

	group_colours = {
				"EXC": 'r',
				"INH": 'b',
				"PV": 'g',
				"SST": 'k'
	}

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



def compare_metrics_of_two_layerwise_groups(df, vivo_df, vivo_experiments, layers, group_1, group_2, stat, prefix, figs_dir, mask_key=''):

	if (mask_key != ''):
		mask_dict = {mask_key:False}
		df = df.etl.q(mask_dict)

	cmap = cm.get_cmap('rocket', 5)
	colors = cmap(np.linspace(0, 1, 5))

	layer_colours = {
				"L23": colors[0],
				"L4": colors[1],
				"L5": colors[2],
				"L6": colors[3]}

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



def plot_psths_neuron_class_pairs(neuron_class_pairs, baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, figs_dir, c_for_masked_psths, xlim, prefix, silico_sigma, mask_key, major_locator, minor_locator, override_c_for_unmasked='', override_c_for_vivo=''):

	time_key = "time"

	silico_bin_size = 1.0
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


def plot_psths_single_neuron_class(neuron_class, baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, figs_dir, c_for_masked_psths, override_c_for_unmasked='', override_c_for_vivo=''):

	time_key = "time"
	silico_sigma = 1
	silico_bin_size = 1.0
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



def create_directories(a):
	masks_dir = str(a.analysis_config['evoked']) + "/masks/"
	psths_dir = str(a.analysis_config['evoked']) + "/psths/"
	heatmaps_dir = str(a.analysis_config['evoked']) + "/heatmaps/"
	time_course_comp_dir = str(a.analysis_config['evoked']) + "/time_course_comp/"
	os.makedirs(masks_dir, exist_ok=True)
	os.makedirs(psths_dir, exist_ok=True)
	os.makedirs(heatmaps_dir, exist_ok=True)
	os.makedirs(time_course_comp_dir, exist_ok=True)

	return masks_dir, psths_dir, heatmaps_dir, time_course_comp_dir

def min_max_normalise(data):
	normalised_data = data - np.min(data)
	normalised_data = normalised_data / np.max(normalised_data)
	return normalised_data

def mean_normalise(data):
	normalised_data = data / np.mean(data)
	return normalised_data


def evoked_ratios_line_plot(a, silico_evoked_ratios, silico_mean_ratios_across_sims, vivo_nc_ratios, neuron_classes, all_window_stat_and_mask_df, normalisation_type='none'):

	# mean_ratio_differences = []
	# for simulation_id in silico_evoked_ratios.simulation_id.unique():
	# 	simulation_nc_evoked_ratios = silico_evoked_ratios.etl.q(simulation_id=simulation_id, neuron_class=neuron_classes)
	# 	mean_ratio_differences.append(np.mean(np.asarray(simulation_nc_evoked_ratios['ratio'])) - vivo_nc_mean_ratio)
	# all_window_stat_and_mask_df.loc[:, 'mean_ratio_difference'] = mean_ratio_differences

	plt.figure(figsize=(8.2*0.3, 8.2*0.3*0.5))
	plt.gca().set_facecolor('grey')
	cmap = sns.color_palette("seismic", as_cmap=True)

	vivo_nc_mean_ratio = np.mean(vivo_nc_ratios)

	silico_mean_ratio_differences = []
	for simulation_id in silico_evoked_ratios.simulation_id.unique():
		simulation_nc_evoked_ratios = silico_evoked_ratios.etl.q(simulation_id=simulation_id, neuron_class=neuron_classes)

		if (normalisation_type == "none"):
			silico_mean_ratio_differences.append(np.mean(np.asarray(simulation_nc_evoked_ratios['ratio'])) - vivo_nc_mean_ratio)
		elif (normalisation_type == "mean_normalise"):
			silico_mean_ratio_differences.append(np.mean(mean_normalise(np.asarray(simulation_nc_evoked_ratios['ratio']))) - mean_normalise(vivo_nc_mean_ratio))


	abs_max_silico_mean_ratio_differences = np.max(np.abs(silico_mean_ratio_differences))
	range_width = abs_max_silico_mean_ratio_differences * 2

	

	for i, simulation_id in enumerate(silico_evoked_ratios.simulation_id.unique()):
		
		simulation_nc_evoked_ratios = silico_evoked_ratios.etl.q(simulation_id=simulation_id, neuron_class=neuron_classes)
		c_val = (silico_mean_ratio_differences[i] / range_width) + 0.5

		if (normalisation_type == "none"):
			plt.plot(range(len(neuron_classes)), simulation_nc_evoked_ratios['ratio'], c=cmap(c_val), lw=.2)
		elif (normalisation_type == "mean_normalise"):
			plt.plot(range(len(neuron_classes)), mean_normalise(simulation_nc_evoked_ratios['ratio']), c=cmap(c_val), lw=.2)
		
	if (normalisation_type == "none"):
		plt.plot(range(len(neuron_classes)), vivo_nc_ratios, c='w', lw=1, marker='o', ms=3)
		plt.plot(range(len(neuron_classes)), silico_mean_ratios_across_sims.etl.q(neuron_class=neuron_classes), c='k', lw=1, marker='o', ms=3)
		plt.gca().set_ylim([0.0, 200.0])

	elif (normalisation_type == "mean_normalise"):
		plt.plot(range(len(neuron_classes)), mean_normalise(vivo_nc_ratios), c='w', lw=1, marker='o', ms=3)
		plt.plot(range(len(neuron_classes)), mean_normalise(silico_mean_ratios_across_sims.etl.q(neuron_class=neuron_classes)), c='k', lw=1, marker='o', ms=3)
		# plt.gca().set_ylim([0.0, 1.0])

	

	plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(200))
	plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(50))
	plt.gca().set_xticks(range(len(neuron_classes)), labels=[c_etl.neuron_class_label_map[nc] for nc in neuron_classes])
	plt.gca().set_ylabel('Evok/spont\nratio', labelpad=-10.0)
	plt.tight_layout()
	plt.savefig(str(a.analysis_config['evoked']) + "/" + normalisation_type + "_EvokedRatios.pdf")
	
	plt.close()


def evoked_ratios_analaysis(a, all_window_stat_and_mask_df, sim_mask_df):

	if ('temp_ratios_df_path' in list(a.analysis_config.keys())):
		evoked_ratios = pd.read_parquet(a.analysis_config['temp_ratios_df_path'])
		silico_evoked_ratios = evoked_ratios.etl.q(type='silico', window='evoked_SOZ_200ms')
		vivo_evoked_ratios = evoked_ratios.etl.q(type='vivo')
		silico_evoked_ratios['simulation_id'] = silico_evoked_ratios['simulation_id'].astype('int')
		silico_evoked_ratios['circuit_id'] = silico_evoked_ratios['circuit_id'].astype('int')

		neuron_classes = ["L23_EXC", "L23_INH", "L4_EXC", "L4_INH", "L5_EXC", "L5_INH"]
		vivo_nc_ratios = vivo_evoked_ratios.etl.q(neuron_class=neuron_classes)['ratio']
		silico_evoked_ratios = pd.merge(silico_evoked_ratios, sim_mask_df, on=['simulation_id'])
		silico_evoked_ratios = silico_evoked_ratios.etl.q(SimBad100p50p25pDecayOverlySustained=False)

		silico_mean_ratios_across_sims = silico_evoked_ratios.groupby(['neuron_class']).median()['ratio']
		silico_ratio_std_across_sims = silico_evoked_ratios.groupby(['neuron_class']).std()['ratio']

		evoked_ratios_line_plot(a, silico_evoked_ratios, silico_mean_ratios_across_sims, vivo_nc_ratios, neuron_classes, all_window_stat_and_mask_df, normalisation_type='none')
		evoked_ratios_line_plot(a, silico_evoked_ratios, silico_mean_ratios_across_sims, vivo_nc_ratios, neuron_classes, all_window_stat_and_mask_df, normalisation_type='mean_normalise')

# 		import pdb; pdb.set_trace()

		plt.figure(figsize=(8.2*0.25, 8.2*0.25))
		for nc in neuron_classes:
			plt.errorbar(vivo_evoked_ratios.etl.q(neuron_class=nc)['ratio'], silico_mean_ratios_across_sims.etl.q(neuron_class=nc), yerr=silico_ratio_std_across_sims.etl.q(neuron_class=nc), c=c_etl.LAYER_EI_NEURON_CLASS_COLOURS[nc], marker=c_etl.LAYER_EI_NEURON_CLASS_MARKERS[nc])
		lims = [0.0, 200.0]
		plt.gca().plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		plt.gca().set_xlim([0, 200.0])
		plt.gca().set_ylim([0, 200.0])
		plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
		plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(50))
		plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(200))
		plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(50))
		plt.gca().set_title('Evok/spont ratio')
		plt.gca().set_xlabel('In-vivo', labelpad=-7.0)
		plt.gca().set_ylabel('In-silico', labelpad=-7.0)
		plt.tight_layout()
		plt.savefig(str(a.analysis_config['evoked']) + "/EvokedRatios2.pdf")
		plt.close()


		scaling_1_L5E_ratios = silico_evoked_ratios.etl.q(neuron_class="L5_EXC", vpm_l5e_cond_scaling_factor=1.0)
		scaling_1_36_L5E_ratios = silico_evoked_ratios.etl.q(neuron_class="L5_EXC", vpm_l5e_cond_scaling_factor=1.36)
		plt.figure(figsize=(8.2*0.25*0.75, 8.2*0.25))
		plt.scatter([1.0 for i in range(len(scaling_1_L5E_ratios))], [scaling_1_L5E_ratios['ratio']], s=1, c='k')
		plt.scatter([1.36 for i in range(len(scaling_1_36_L5E_ratios))], [scaling_1_36_L5E_ratios['ratio']], s=1, c='k')
		for simulation_id in scaling_1_L5E_ratios['simulation_id'].unique():
			scaling_1_L5E_row = scaling_1_L5E_ratios.etl.q(simulation_id=simulation_id).iloc[0]
			scaling_1_36_L5E_same_vars = scaling_1_36_L5E_ratios.etl.q(ca=scaling_1_L5E_row['ca'], desired_connected_proportion_of_invivo_frs=scaling_1_L5E_row['desired_connected_proportion_of_invivo_frs'], depol_stdev_mean_ratio=scaling_1_L5E_row['depol_stdev_mean_ratio'], vpm_pct=scaling_1_L5E_row['vpm_pct'])
			if (len(scaling_1_36_L5E_same_vars)):
				plt.plot([1, 1.36], [scaling_1_L5E_row['ratio'], scaling_1_36_L5E_same_vars.iloc[0]['ratio']], c='k', lw='0.1')			

		plt.gca().set_xlim([0.9, 1.46])
		plt.gca().set_ylim([0.0, silico_evoked_ratios.etl.q(neuron_class="L5_EXC")['ratio'].max() * 1.05])
		plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))
		plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(5))
		plt.gca().set_ylabel('Evok/spont ratio', labelpad=-7.0)
		plt.gca().set_xticks(ticks=[1.0, 1.36],labels=['Mean', 'Max'])
		plt.gca().set_xlabel('TC -> L5E conductance')
		plt.tight_layout()
		plt.savefig(str(a.analysis_config['evoked']) + '/L5E_scaling_ratio_comparison.pdf')
		plt.close()
	

def evoked_analysis(a):

	analysis_window = a.analysis_config['evoked_window_for_custom_post_analysis']
	vivo_df, rp_psth_df, svo_psth_df = load_and_preprocess_vivo_dataframes()
	decay_df, latency_df, baseline_PSTH_df, sim_df = preprocess_silico_dataframes(a)
	masks_dir, psths_dir, heatmaps_dir, time_course_comp_dir = create_directories(a)

	nc_comparison_decay_df = compare_neuron_classes_with_in_vivo(decay_df, vivo_df, a)
	sim_mask_df = create_sim_mask(nc_comparison_decay_df, a.repo.simulations.df, ["ALL", "L23_EXC", "L23_INH", "L4_EXC", "L4_INH", "L5_EXC", "L5_INH"])
	nc_stat_and_mask_df = merge_comparison_and_mask(nc_comparison_decay_df, sim_mask_df)
	
	nc_window_comparison_decay_df = nc_comparison_decay_df.etl.q(window=analysis_window)
	nc_window_stat_and_mask_df = nc_stat_and_mask_df.etl.q(window=analysis_window)
	all_window_stat_and_mask_df = nc_window_stat_and_mask_df.etl.q(neuron_class='ALL')
	window_sim_mask_df = sim_mask_df.etl.q(window=analysis_window)
	window_baseline_PSTH_df = baseline_PSTH_df.etl.q(window=analysis_window)

	evoked_ratios_analaysis(a, all_window_stat_and_mask_df, sim_mask_df)
	create_heatmaps(a, window_sim_mask_df, all_window_stat_and_mask_df, masks_dir, heatmaps_dir)
	# compare_time_courses(nc_window_stat_and_mask_df.etl.q(bin_size=1.0, sigma=1), vivo_df, time_course_comp_dir)
	# import pdb; pdb.set_trace()
	compare_time_courses(nc_window_stat_and_mask_df.etl.q(bin_size=0.5, sigma=2), vivo_df, time_course_comp_dir)
	# window_baseline_PSTH_df = window_baseline_PSTH_df.etl.q(simulation_id=sim_mask_df.etl.q(vpm_pct=[5.0, 10.0], vpm_l5e_cond_scaling_factor=1.36)['simulation_id'])
	# window_baseline_PSTH_df = window_baseline_PSTH_df.etl.q(simulation_id=[221,381,421,861])
	for neuron_class in ["ALL", "L23_EXC", "L23_INH", "L4_EXC", "L4_INH", "L5_EXC", "L5_INH"]:
		plot_psths_single_neuron_class(neuron_class, window_baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_comparison_decay_df, psths_dir, 'slategrey', override_c_for_unmasked='lightgreen', override_c_for_vivo='k')
	plot_psths_neuron_class_pairs([["ALL", "L1_INH", "", ""], ["L23_EXC", "L23_INH", "L23_PV", "L23_SST"], ["L4_EXC", "L4_INH", "L4_PV", "L4_SST"], ["L5_EXC", "L5_INH", "L5_PV", "L5_SST"], ["L6_EXC", "L6_INH", "L6_PV", "L6_SST"]], window_baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_stat_and_mask_df, psths_dir, 'w', [0, 50], 'short_without_bad', 1, 'Bad100p50p25pDecayOverlySustained', ticker.MultipleLocator(50), ticker.MultipleLocator(10), override_c_for_vivo='k')
	plot_psths_neuron_class_pairs([["ALL", "L1_INH", "", ""], ["L23_EXC", "L23_INH", "L23_PV", "L23_SST"], ["L4_EXC", "L4_INH", "L4_PV", "L4_SST"], ["L5_EXC", "L5_INH", "L5_PV", "L5_SST"], ["L6_EXC", "L6_INH", "L6_PV", "L6_SST"]], window_baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_stat_and_mask_df, psths_dir, 'r', [0, 200], 'long_with_bad', 3, 'Bad100p50p25pDecayOverlySustained', ticker.MultipleLocator(200), ticker.MultipleLocator(25), override_c_for_unmasked='g', override_c_for_vivo='k')
	plot_psths_neuron_class_pairs([["ALL", "L1_INH", "", ""], ["L23_EXC", "L23_INH", "L23_PV", "L23_SST"], ["L4_EXC", "L4_INH", "L4_PV", "L4_SST"], ["L5_EXC", "L5_INH", "L5_PV", "L5_SST"], ["L6_EXC", "L6_INH", "L6_PV", "L6_SST"]], window_baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_stat_and_mask_df, psths_dir, 'w', [0, 50], 'short_without_sim_bad', 1, 'SimBad100p50p25pDecayOverlySustained', ticker.MultipleLocator(50), ticker.MultipleLocator(10), override_c_for_vivo='k')
	plot_psths_neuron_class_pairs([["ALL", "L1_INH", "", ""], ["L23_EXC", "L23_INH", "L23_PV", "L23_SST"], ["L4_EXC", "L4_INH", "L4_PV", "L4_SST"], ["L5_EXC", "L5_INH", "L5_PV", "L5_SST"], ["L6_EXC", "L6_INH", "L6_PV", "L6_SST"]], window_baseline_PSTH_df, rp_psth_df, svo_psth_df, nc_window_stat_and_mask_df, psths_dir, 'slategrey', [0, 200], 'long_with_sim_bad', 3, 'SimBad100p50p25pDecayOverlySustained', ticker.MultipleLocator(200), ticker.MultipleLocator(25), override_c_for_unmasked='mediumseagreen', override_c_for_vivo='k')
	

	# import pdb; pdb.set_trace()










# def compare_metrics_in_vivo_vs_in_silico_boxplot(df, vivo_df, vivo_exp, layers, groups, stat, prefix, figs_dir, upper_lim=20.0, mask_key=''):

# 	if (mask_key != ''):
# 		mask_dict = {mask_key:False}
# 		df = df.etl.q(mask_dict)

# 	cmap = cm.get_cmap('rocket', 5)
# 	colors = cmap(np.linspace(0, 1, 5))

# 	layer_colours = {
# 				"L23": colors[0],
# 				"L4": colors[1],
# 				"L5": colors[2],
# 				"L6": colors[3]}

# 	group_colours = {
# 				"EXC": 'r',
# 				"INH": 'b',
# 				"PV": 'g',
# 				"SST": 'k'
# 	}

# 	for layer in layers:
# 		for group in groups:
# 			neuron_class = layer + "_" + group

# 			nc_df = df.etl.q(neuron_class=neuron_class)
# 			vivo_stat_df_1 = vivo_df.etl.q(experiment=vivo_exp, neuron_class=neuron_class, bin_size=1, sigma=1)

# 			plt.boxplot([nc_df[stat]], positions=[vivo_stat_df_1.iloc[0][stat]])

# 			num_points = len(nc_df)
# 			plt.scatter([vivo_stat_df_1.iloc[0][stat] for i in range(num_points)], nc_df[stat] + np.random.normal(0.0, 0.05, num_points), s=0.2, marker=c_etl.LAYER_MARKERS[layer], c=[group_colours[group] for i in range(num_points)])

# 	# plt.gca().legend()

# 	plt.gca().set_xlim([0.0, upper_lim])
# 	plt.gca().set_ylim([0.0, upper_lim])
# 	plt.gca().set_xlabel("In vivo")
# 	plt.gca().set_ylabel("In silico")
# 	plt.gca().plot([0.0, upper_lim], [0.0, upper_lim], 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
# 	plt.savefig(figs_dir + vivo_exp + "COMP_" + prefix + '_' + groups[0] + '_+_' + groups[1] + '_' + stat + '_' + mask_key + '_BOXPLOT.pdf')
# 	plt.close()




