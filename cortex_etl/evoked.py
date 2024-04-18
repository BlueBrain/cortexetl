import pandas as pd
import numpy as np
import os
import cortex_etl as c_etl


# ENTRY FUNCTION
######################################################################
def evoked_analysis(a):

    neuron_classes, vivo_df, rp_psth_df, svo_psth_df = evoked_preprocessing(a)

    evoked_ratios_analysis(a, neuron_classes)
    create_heatmaps(a)
    compare_time_courses(a, vivo_df, silico_bin_size=0.5, silico_sigma=2)
    psth_plots(a, neuron_classes, rp_psth_df, svo_psth_df)


# HELPERS
######################################################################
def add_custom_df(a, key, df):
	if ('custom' not in a.__dict__.keys()):
		a.custom = {}
	a.custom[key] = df


# TESTS
######################################################################
def silico_vivo_decay_difference(rp_vivo_df, svoboda_vivo_df, row, stat, evoked_mask_comparison_dataset):
	nc = row['neuron_class']

	vivo_query = []
	if ((evoked_mask_comparison_dataset == 'ReyesPuerta') & (nc in rp_vivo_df.neuron_class.unique())):
		vivo_query = rp_vivo_df.etl.q(neuron_class=row['neuron_class'])
	elif (nc in svoboda_vivo_df.neuron_class.unique()):
		vivo_query = svoboda_vivo_df.etl.q(neuron_class=nc)

	if (len(vivo_query)):
		return row[stat] - vivo_query.iloc[0][stat]
	return -100000.0


def overly_sustained_test(psth_df, row, threshold_prop=0.35):
    return np.sum(psth_df.etl.q({"simulation_id": row['simulation_id'], "neuron_class":row['neuron_class']})['psth'] > threshold_prop) #, "time": {"ge": time_from}}


# PROCESSING
######################################################################
def evoked_preprocessing(a):

    create_directories(a)

    vivo_df, rp_psth_df, svo_psth_df = preprocess_vivo_dfs()
    neuron_classes = select_comparison_neuron_classes(a)

    compare_vivo_and_silico(a, vivo_df)
    create_sim_mask(a, ["ALL"] + neuron_classes)
    merge_comparison_and_mask(a)
    filter_dfs_by_window(a)

    return neuron_classes, vivo_df, rp_psth_df, svo_psth_df


def create_directories(a):
    a.analysis_config.custom['masks_dir'] = str(a.figpaths.evoked) + "/masks/"
    a.analysis_config.custom['psths_dir'] = str(a.figpaths.evoked) + "/psths/"
    a.analysis_config.custom['heatmaps_dir'] = str(a.figpaths.evoked) + "/heatmaps/"
    a.analysis_config.custom['time_course_comp_dir'] = str(a.figpaths.evoked) + "/time_course_comp/"
    os.makedirs(a.analysis_config.custom['masks_dir'], exist_ok=True); os.makedirs(a.analysis_config.custom['psths_dir'], exist_ok=True); os.makedirs(a.analysis_config.custom['heatmaps_dir'], exist_ok=True); os.makedirs(a.analysis_config.custom['time_course_comp_dir'], exist_ok=True)


def preprocess_vivo_dfs():
    vivo_dfs = []
    for vivo_path, vivo_name in zip(["/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/reyes-multi-sigma-features.parquet",
                                    "/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/svoboda-multi-sigma-features.parquet"], 
                         ['ReyesPuerta', 'YuSvoboda']):
        
        df = pd.read_parquet(vivo_path)
        df = df.pivot_table('decay', ['neuron_class', 'bin_size', 'sigma', 'latency'], 'ratio')
        df = df.reset_index(level=['latency'])
        df = df.reset_index()
        df = df.rename(columns={"latency": '1.0', 0.75: '0.75', 0.5: '0.5', 0.25: '0.25'})
        df['experiment'] = vivo_name
        vivo_dfs.append(df)

    vivo_df = pd.concat(vivo_dfs)

    rp_psth_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/reyes-multi-sigma-PSTH.parquet")
    svo_psth_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/bc-simulation-analysis/invivo/svoboda-multi-sigma-PSTH.parquet")

    return vivo_df, rp_psth_df, svo_psth_df



def select_comparison_neuron_classes(a):
    
    if (a.analysis_config.custom['evoked_mask_comparison_dataset'] == "ReyesPuerta"): return c_etl.LAYER_EI_RP_NEURON_CLASSES
    elif (a.analysis_config.custom['evoked_mask_comparison_dataset'] == "YuSvoboda"): return c_etl.LAYER_EI_SVO_NEURON_CLASSES
    

# Check neuron class latencies and decays aren't too different from vivo
# Also check for overly sustained response (i.e. bursting, or double bump response)
def compare_vivo_and_silico(a, vivo_df):

	# Decay parameters
	silico_decay_bin_size=0.5
	silico_decay_sigma=2
	vivo_decay_bin_size=1.0
	vivo_decay_sigma=1
	p100_diff_threshold = 10.0
	p50_diff_threshold = 10.0
	p25_diff_threshold = 40.0

	# Overly sustained parameters (i.e. bursting, or double bump response)
	overly_sustained_bin_size=1.0
	overly_sustained_sigma = 3.0
	overly_sustained_gt_time = 75
	overly_sustained_threshold_prop = 0.35


	# Prepare silico and vivo dataframes
	silico_decay_df = a.features.decay.df.pivot_table('decay', ['simulation_id', 'circuit_id', 'window', 'neuron_class', 'bin_size', 'sigma'], 'ratio') \
										.rename(columns={1.0: '1.0', 0.75: '0.75', 0.5: '0.5', 0.25: '0.25'}) \
										.reset_index() \
										.etl.q(bin_size=silico_decay_bin_size, sigma=silico_decay_sigma)
	# print(silico_decay_df)


	rp_vivo_df = vivo_df.etl.q(experiment='ReyesPuerta', bin_size=vivo_decay_bin_size, sigma=vivo_decay_sigma)
	svoboda_vivo_df = vivo_df.etl.q(experiment='YuSvoboda', bin_size=vivo_decay_bin_size, sigma=vivo_decay_sigma)

	
	# Make comparisons
	nc_comparison_decay_df = silico_decay_df
	mask_comparison_dataset_key = a.analysis_config.custom['evoked_mask_comparison_dataset']

	# -- Decay
	nc_comparison_decay_df.loc[:, '100pDiffToVivo'] = nc_comparison_decay_df.apply(lambda row: silico_vivo_decay_difference(rp_vivo_df, svoboda_vivo_df, row, "1.0", mask_comparison_dataset_key), axis = 1)
	nc_comparison_decay_df.loc[:, '75pDiffToVivo'] = nc_comparison_decay_df.apply(lambda row: silico_vivo_decay_difference(rp_vivo_df, svoboda_vivo_df, row, "0.75", mask_comparison_dataset_key), axis = 1)
	nc_comparison_decay_df.loc[:, '50pDiffToVivo'] = nc_comparison_decay_df.apply(lambda row: silico_vivo_decay_difference(rp_vivo_df, svoboda_vivo_df, row, "0.5", mask_comparison_dataset_key), axis = 1)
	nc_comparison_decay_df.loc[:, '25pDiffToVivo'] = nc_comparison_decay_df.apply(lambda row: silico_vivo_decay_difference(rp_vivo_df, svoboda_vivo_df, row, "0.25", mask_comparison_dataset_key), axis = 1)
	nc_comparison_decay_df['Bad100pDecay'] = ((nc_comparison_decay_df["1.0"] == -1.0) | (nc_comparison_decay_df["100pDiffToVivo"] > p100_diff_threshold)).astype('bool')
	nc_comparison_decay_df['Bad50pDecay'] = ((nc_comparison_decay_df["0.5"] == -1.0) | (nc_comparison_decay_df["50pDiffToVivo"] > p50_diff_threshold)).astype('bool')
	nc_comparison_decay_df['Bad25pDecay'] = ((nc_comparison_decay_df["0.25"] == -1.0) | (nc_comparison_decay_df["25pDiffToVivo"] > p25_diff_threshold)).astype('bool')
	nc_comparison_decay_df['Bad50p25pDecay'] = (nc_comparison_decay_df['Bad50pDecay'] | nc_comparison_decay_df['Bad25pDecay'])
	nc_comparison_decay_df['Bad100p50p25pDecay'] = (nc_comparison_decay_df['Bad100pDecay'] | nc_comparison_decay_df['Bad50pDecay'] | nc_comparison_decay_df['Bad25pDecay'])

	# -- Overly sustained mask (i.e. checking for bursting, or double bump response)
	psth_df_for_sustained_test = a.features.baseline_PSTH.df.reset_index().etl.q({"window": a.analysis_config.custom["evoked_window_for_custom_post_analysis"], 
																				"bin_size": overly_sustained_bin_size, 
																				"sigma": overly_sustained_sigma, 
																				"time": {"ge": overly_sustained_gt_time}})
	nc_comparison_decay_df.loc[:, 'OverlySustained'] = nc_comparison_decay_df.apply(lambda row: overly_sustained_test(psth_df_for_sustained_test, row, threshold_prop=overly_sustained_threshold_prop), axis=1).astype('bool')

	# -- Mask for bad decay and overly sustained response
	nc_comparison_decay_df['Bad100p50p25pDecayOverlySustained'] = (nc_comparison_decay_df['Bad100p50p25pDecay'] | nc_comparison_decay_df['OverlySustained'])


	add_custom_df(a, "nc_comparison_decay_df", nc_comparison_decay_df)


def create_sim_mask(a, ncs):

	filt_nc_comparison_decay_df = a.custom['nc_comparison_decay_df'].etl.q(neuron_class=ncs)

	# Check if any of these tests failed for any of the neurona classes (including all population if in ncs)
	test_keys = ['Bad25pDecay', 'Bad50pDecay', 'Bad100pDecay', 'Bad50p25pDecay', 'Bad100p50p25pDecay', 'Bad100p50p25pDecayOverlySustained']
	sim_mask_df = filt_nc_comparison_decay_df.groupby(by=['simulation_id', 'window'])[test_keys].sum().astype('bool')

	# Rename columns. Now have one row per simulation.
	sim_mask_df = sim_mask_df.rename(columns={"Bad25pDecay": "SimBad25pDecay", 
												"Bad50pDecay": "SimBad50pDecay", 
												"Bad100pDecay": "SimBad100pDecay", 
												"Bad50p25pDecay": "SimBad50p25pDecay", 
												"Bad100p50p25pDecay": "SimBad100p50p25pDecay", 
												"Bad100p50p25pDecayOverlySustained":"SimBad100p50p25pDecayOverlySustained"}).reset_index()

	sim_mask_df = pd.merge(a.repo.simulations.df, sim_mask_df, on=['simulation_id']) #.drop(columns=[])

	add_custom_df(a, "sim_mask_df", sim_mask_df)


def merge_comparison_and_mask(a):

	# print("a.custom['nc_comparison_decay_df']: " + str(a.custom['nc_comparison_decay_df'].columns))
	# print("a.custom['sim_mask_df']: " + str(a.custom['sim_mask_df']))

	nc_stat_and_mask_df = pd.merge(a.custom['nc_comparison_decay_df'], a.custom['sim_mask_df'], on=['simulation_id', 'window'])
	# print("nc_stat_and_mask_df: " + str(nc_stat_and_mask_df))

	add_custom_df(a, "nc_stat_and_mask_df", nc_stat_and_mask_df)


def filter_dfs_by_window(a):

	window = a.analysis_config.custom['evoked_window_for_custom_post_analysis']

	add_custom_df(a, "nc_window_comparison_decay_df", 	a.custom['nc_comparison_decay_df'].etl.q(window=window))
	add_custom_df(a, "nc_window_stat_and_mask_df", 		a.custom['nc_stat_and_mask_df'].etl.q(window=window))
	add_custom_df(a, "all_window_stat_and_mask_df", 	a.custom['nc_window_stat_and_mask_df'].etl.q(neuron_class='ALL'))
	add_custom_df(a, "window_sim_mask_df", 				a.custom['sim_mask_df'].etl.q(window=window))


# PLOTTING
######################################################################
def create_heatmaps(a):
    
    hm_dims = (a.analysis_config.custom['heatmap_dims']['hor_key'], a.analysis_config.custom['heatmap_dims']['ver_key'], a.analysis_config.custom['heatmap_dims']['x_key'], a.analysis_config.custom['heatmap_dims']['y_key'])
    
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad25pDecay", a.analysis_config.custom['masks_dir'] + "SimBad25pDecay", *hm_dims)
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad50pDecay", a.analysis_config.custom['masks_dir'] + "SimBad50pDecay", *hm_dims)
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad100pDecay", a.analysis_config.custom['masks_dir'] + "SimBad100pDecay", *hm_dims)
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad50p25pDecay", a.analysis_config.custom['masks_dir'] + "SimBad50p25pDecay", *hm_dims)
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad100p50p25pDecay", a.analysis_config.custom['masks_dir'] + "SimBad100p50p25pDecay", *hm_dims)
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad100p50p25pDecayOverlySustained", a.analysis_config.custom['masks_dir'] + "SimBad100p50p25pDecayOverlySustained", *hm_dims)
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad25pDecay", a.analysis_config.custom['masks_dir'] + "SimBad25pDecay", *hm_dims, mask_key='SimBad100pDecay')
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad50p25pDecay", a.analysis_config.custom['masks_dir'] + "SimBad50p25pDecay", *hm_dims, mask_key='SimBad100pDecay')
    c_etl.heatmap(a.custom['window_sim_mask_df'], "SimBad100p50p25pDecay", a.analysis_config.custom['masks_dir'] + "SimBad100p50p25pDecay", *hm_dims, mask_key='SimBad100p50p25pDecay')


    if ("mean_ratio_difference" in list(a.custom['all_window_stat_and_mask_df'].columns)):
        c_etl.heatmap(a.custom['all_window_stat_and_mask_df'], "mean_ratio_difference", a.analysis_config.custom['heatmaps_dir'] + "mean_ratio_difference", *hm_dims, mask_key='SimBad100p50p25pDecayOverlySustained')



def compare_time_courses(a, vivo_df, silico_bin_size=0.5, silico_sigma=2):
    
    nc_stat_and_mask_df = a.custom['nc_window_stat_and_mask_df'].etl.q(bin_size=silico_bin_size, sigma=silico_sigma)
    figs_dir = a.analysis_config.custom['time_course_comp_dir']
    
    nc_stat_and_mask_df['layer'] = nc_stat_and_mask_df.apply(lambda row: row['neuron_class'].split('_')[0], axis = 1)
    for decay in ['1.0']:
        c_etl.layer_and_pairwise_comparison(nc_stat_and_mask_df, vivo_df, "ReyesPuerta", decay, [["L6_INH", "L6_EXC"], ["L5_INH", "L5_EXC"], ["L4_INH", "L4_EXC"], ["L23_INH", "L23_EXC"]], figs_dir + "ReyesPuertaEI_latencies.pdf", [0, 11], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        c_etl.layer_and_pairwise_comparison(nc_stat_and_mask_df, vivo_df, "YuSvoboda", decay, [["L6_PV", "L6_SST"], ["L5_PV", "L5_SST"], ["L4_PV", "L4_SST"], ["L23_PV", "L23_SST"]], figs_dir + "ReyesPuertaPVSST_latencies.pdf", [0, 19], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])

        c_etl.compare_metrics_in_vivo_vs_in_silico(nc_stat_and_mask_df, vivo_df, 'ReyesPuerta', ['L23', 'L4', 'L5'], ["EXC", "INH"], decay, "T", figs_dir, upper_lim=15.0, mask_key='SimBad100p50p25pDecayOverlySustained')
        c_etl.compare_metrics_in_vivo_vs_in_silico(nc_stat_and_mask_df, vivo_df, 'YuSvoboda', ['L23', 'L4', 'L5', 'L6'], ["EXC", "INH"], decay, "T", figs_dir, upper_lim=15.0, mask_key='SimBad100p50p25pDecayOverlySustained')
        c_etl.compare_metrics_in_vivo_vs_in_silico(nc_stat_and_mask_df, vivo_df, 'YuSvoboda', ['L23', 'L4', 'L5', 'L6'], ["PV", "SST"], decay, "T", figs_dir, upper_lim=20.0, mask_key='SimBad100p50p25pDecayOverlySustained')
        
        c_etl.compare_metrics_of_two_layerwise_groups(nc_stat_and_mask_df, vivo_df, ['ReyesPuerta'], ['L23', 'L4', 'L5', 'L6'], "EXC", "INH", decay, "ReyesPuerta", figs_dir)
        c_etl.compare_metrics_of_two_layerwise_groups(nc_stat_and_mask_df, vivo_df, ['YuSvoboda'], ['L23', 'L4', 'L5', 'L6'], "EXC", "INH", decay, "YuSvoboda", figs_dir, mask_key='SimBad100p50p25pDecayOverlySustained')
        c_etl.compare_metrics_of_two_layerwise_groups(nc_stat_and_mask_df, vivo_df, ['YuSvoboda'], ['L23', 'L4', 'L5', 'L6'], "PV", "SST", decay, "YuSvoboda", figs_dir, mask_key='SimBad100p50p25pDecayOverlySustained')


import pandas as pd
import numpy as np

def evoked_ratios_analysis(a, neuron_classes):
    
    if ('vivo_ratios_df' in list(a.analysis_config.custom.keys())):
        
        vivo_evoked_ratios = pd.read_feather(a.analysis_config.custom['vivo_ratios_df']).etl.q(type='vivo', vivo=a.analysis_config.custom['vivo_ratios_dataset']) # Change to temp_ratios_df_path
        vivo_nc_ratios = vivo_evoked_ratios.etl.q(neuron_class=neuron_classes)['ratio']

        silico_evoked_ratios = pd.merge(a.features.mean_psth.df.reset_index(), a.features.max_psth.df.reset_index())
        silico_evoked_ratios = silico_evoked_ratios.etl.q(window='evoked_SOZ_250ms')        
        silico_evoked_ratios['ratio'] = silico_evoked_ratios['max_psth'] / silico_evoked_ratios['mean_psth']

        silico_evoked_ratios = pd.merge(silico_evoked_ratios, a.custom['sim_mask_df'], on=['simulation_id'])
        silico_evoked_ratios = silico_evoked_ratios.etl.q(SimBad100p50p25pDecayOverlySustained=False)
        
        silico_mean_ratios_across_sims = silico_evoked_ratios.groupby(['neuron_class'])['ratio'].median()
        silico_ratio_std_across_sims = silico_evoked_ratios.groupby(['neuron_class'])['ratio'].std()

        c_etl.evoked_ratios_line_plot(a, silico_evoked_ratios, silico_mean_ratios_across_sims, vivo_nc_ratios, neuron_classes, normalisation_type='none')
        c_etl.evoked_ratios_line_plot(a, silico_evoked_ratios, silico_mean_ratios_across_sims, vivo_nc_ratios, neuron_classes, normalisation_type='mean_normalise')



import matplotlib.ticker as ticker
def psth_plots(a, neuron_classes, rp_psth_df, svo_psth_df):

    ##### PLOT PSTHs ####
    window_baseline_PSTH_df = a.features.baseline_PSTH.df.reset_index().etl.q(window=a.analysis_config.custom['evoked_window_for_custom_post_analysis'])
    for neuron_class in ["ALL"] + neuron_classes:
        c_etl.plot_psths_single_neuron_class(neuron_class, window_baseline_PSTH_df, rp_psth_df, svo_psth_df, a.custom['nc_window_comparison_decay_df'], a.analysis_config.custom['psths_dir'], 'slategrey', 1.0, 1, override_c_for_unmasked='mediumseagreen', override_c_for_vivo='k')  
    
    c_etl.plot_psths_neuron_class_pairs(c_etl.PYR_AND_IN_PSTH_NC_GROUPINGS_BY_LAYER, window_baseline_PSTH_df, rp_psth_df, svo_psth_df, a.custom['nc_window_stat_and_mask_df'], a.analysis_config.custom['psths_dir'], 'w', [0, 50], 'short_without_bad', 1.0, 1, 'Bad100p50p25pDecayOverlySustained', ticker.MultipleLocator(50), ticker.MultipleLocator(10), override_c_for_vivo='k')
    c_etl.plot_psths_neuron_class_pairs(c_etl.PYR_AND_IN_PSTH_NC_GROUPINGS_BY_LAYER, window_baseline_PSTH_df, rp_psth_df, svo_psth_df, a.custom['nc_window_stat_and_mask_df'], a.analysis_config.custom['psths_dir'], 'w', [0, 50], 'short_without_sim_bad', 1.0, 1, 'SimBad100p50p25pDecayOverlySustained', ticker.MultipleLocator(50), ticker.MultipleLocator(10), override_c_for_vivo='k')

    c_etl.plot_psths_neuron_class_pairs(c_etl.PYR_AND_IN_PSTH_NC_GROUPINGS_BY_LAYER, window_baseline_PSTH_df, rp_psth_df, svo_psth_df, a.custom['nc_window_stat_and_mask_df'], a.analysis_config.custom['psths_dir'], 'slategrey', [0, 200], 'long_with_bad', 1.0, 3, 'Bad100p50p25pDecayOverlySustained', ticker.MultipleLocator(200), ticker.MultipleLocator(25), override_c_for_unmasked='mediumseagreen', override_c_for_vivo='k')
    c_etl.plot_psths_neuron_class_pairs(c_etl.PYR_AND_IN_PSTH_NC_GROUPINGS_BY_LAYER, window_baseline_PSTH_df, rp_psth_df, svo_psth_df, a.custom['nc_window_stat_and_mask_df'], a.analysis_config.custom['psths_dir'], 'slategrey', [0, 200], 'long_with_sim_bad', 1.0, 3, 'SimBad100p50p25pDecayOverlySustained', ticker.MultipleLocator(200), ticker.MultipleLocator(25), override_c_for_unmasked='mediumseagreen', override_c_for_vivo='k')


