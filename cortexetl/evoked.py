import pandas as pd
import numpy as np
import os
import cortexetl as c_etl

"""
Entry function
"""
def evoked_analysis(a):

    comp_ncs, \
    vivo_df, \
    rp_psth_df, \
    svo_psth_df = evoked_processing(a)

    c_etl.evoked_plots(a, comp_ncs, vivo_df, rp_psth_df, svo_psth_df)


"""
Processing
"""
def evoked_processing(a):

    c_etl.create_dirs(a)

    vivo_df, rp_psth_df, svo_psth_df = c_etl.preprocess_vivo_dfs()
    comp_ncs = c_etl.select_comparison_neuron_classes(a)

    c_etl.compare_latencies_and_decay(a, vivo_df)
    c_etl.create_sim_mask(a, ["ALL"] + comp_ncs)
    c_etl.compare_evoked_ratios(a, comp_ncs)
    c_etl.merge_comparison_and_mask(a)
    c_etl.filter_dfs_by_window(a)

    return comp_ncs, vivo_df, rp_psth_df, svo_psth_df


def preprocess_vivo_dfs():
    vivo_dfs = []
    for vivo_path, vivo_name in zip(["/gpfs/bbp.cscs.ch/project/proj83/home/isbister/data/reference_data_do_not_delete/bc-simulation-analysis/invivo/reyes-multi-sigma-features.parquet",
                                    "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/data/reference_data_do_not_delete/bc-simulation-analysis/invivo/svoboda-multi-sigma-features.parquet"], 
                         ['ReyesPuerta', 'YuSvoboda']):
        
        df = pd.read_parquet(vivo_path)\
        			.pivot_table('decay', ['neuron_class', 'bin_size', 'sigma', 'latency'], 'ratio')\
        			.reset_index(level=['latency'])\
        			.reset_index()\
        			.rename(columns={"latency": '1.0', 0.75: '0.75', 0.5: '0.5', 0.25: '0.25'})
        df['experiment'] = vivo_name
        vivo_dfs.append(df)

    vivo_df = pd.concat(vivo_dfs)

    rp_psth_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/data/reference_data_do_not_delete/bc-simulation-analysis/invivo/reyes-multi-sigma-PSTH.parquet")
    svo_psth_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/data/reference_data_do_not_delete/bc-simulation-analysis/invivo/svoboda-multi-sigma-PSTH.parquet")

    vivo_df['neuron_class'] = vivo_df.apply(lambda row: row.neuron_class.replace("Htr3a", "5HT3aR"), axis=1)
    rp_psth_df['neuron_class'] = rp_psth_df.apply(lambda row: row.neuron_class.replace("Htr3a", "5HT3aR"), axis=1)
    svo_psth_df['neuron_class'] = svo_psth_df.apply(lambda row: row.neuron_class.replace("Htr3a", "5HT3aR"), axis=1)

    return vivo_df, rp_psth_df, svo_psth_df



def select_comparison_neuron_classes(a):
    
    if (a.analysis_config.custom['evoked_mask_comparison_dataset'] == "ReyesPuerta"): return c_etl.LAYER_EI_RP_NEURON_CLASSES
    elif (a.analysis_config.custom['evoked_mask_comparison_dataset'] == "YuSvoboda"): return c_etl.LAYER_EI_SVO_NEURON_CLASSES
    

# Check neuron class latencies and decays aren't too different from vivo
# Also check for overly sustained response (i.e. bursting, or double bump response)
def compare_latencies_and_decay(a, v_timepoints):

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

	rp_timepoints = v_timepoints.etl.q(experiment='ReyesPuerta', bin_size=vivo_decay_bin_size, sigma=vivo_decay_sigma)
	svoboda_timepoints = v_timepoints.etl.q(experiment='YuSvoboda', bin_size=vivo_decay_bin_size, sigma=vivo_decay_sigma)

	
	# Make comparisons
	timepoints_comp = silico_decay_df
	mask_comp_dataset_key = a.analysis_config.custom['evoked_mask_comparison_dataset']

	# -- Decay
	timepoints_comp['100pDiffToVivo'] = timepoints_comp.apply(lambda row: \
		silico_vivo_decay_difference(rp_timepoints, svoboda_timepoints, row, "1.0", mask_comp_dataset_key), axis=1)

	timepoints_comp.loc[:, '75pDiffToVivo'] = timepoints_comp.apply(lambda row: \
		silico_vivo_decay_difference(rp_timepoints, svoboda_timepoints, row, "0.75", mask_comp_dataset_key), axis=1)

	timepoints_comp.loc[:, '50pDiffToVivo'] = timepoints_comp.apply(lambda row: \
		silico_vivo_decay_difference(rp_timepoints, svoboda_timepoints, row, "0.5", mask_comp_dataset_key), axis=1)

	timepoints_comp.loc[:, '25pDiffToVivo'] = timepoints_comp.apply(lambda row: \
		silico_vivo_decay_difference(rp_timepoints, svoboda_timepoints, row, "0.25", mask_comp_dataset_key), axis=1)


	timepoints_comp['Bad100pDecay'] = ((timepoints_comp["1.0"] == -1.0) | (timepoints_comp["100pDiffToVivo"] > p100_diff_threshold)).astype('bool')
	timepoints_comp['Bad50pDecay'] = ((timepoints_comp["0.5"] == -1.0) | (timepoints_comp["50pDiffToVivo"] > p50_diff_threshold)).astype('bool')
	timepoints_comp['Bad25pDecay'] = ((timepoints_comp["0.25"] == -1.0) | (timepoints_comp["25pDiffToVivo"] > p25_diff_threshold)).astype('bool')

	timepoints_comp['Bad50p25pDecay'] = (timepoints_comp['Bad50pDecay'] | timepoints_comp['Bad25pDecay'])
	timepoints_comp['Bad100p50p25pDecay'] = (timepoints_comp['Bad100pDecay'] | timepoints_comp['Bad50pDecay'] | timepoints_comp['Bad25pDecay'])

	# -- Overly sustained mask (i.e. checking for bursting, or double bump response)
	psth_df_for_sustained_test = a.features.baseline_PSTH.df.reset_index().etl.q({"window": a.analysis_config.custom["evoked_window_for_custom_post_analysis"], 
																				"bin_size": overly_sustained_bin_size, 
																				"sigma": overly_sustained_sigma, 
																				"time": {"ge": overly_sustained_gt_time}})
	timepoints_comp.loc[:, 'OverlySustained'] = timepoints_comp.apply(lambda row: overly_sustained_test(psth_df_for_sustained_test, row, threshold_prop=overly_sustained_threshold_prop), axis=1).astype('bool')

	# -- Mask for bad decay and overly sustained response
	timepoints_comp['Bad100p50p25pDecayOverlySustained'] = (timepoints_comp['Bad100p50p25pDecay'] | timepoints_comp['OverlySustained'])

	c_etl.add_custom_df(a, "timepoints_comp", timepoints_comp)


def compare_evoked_ratios(a, comp_ncs):
    # Ratios
    if ('vivo_ratios_df' in list(a.analysis_config.custom.keys())):

        # Vivo
        v_ratios = pd.read_feather(a.analysis_config.custom['vivo_ratios_df'])\
                            .etl.q(neuron_class=comp_ncs, 
                                    type='vivo', 
                                    vivo=a.analysis_config.custom['vivo_ratios_dataset']\
                                    )\
                            .loc[:, ['neuron_class', 'ratio']]\
                            .rename(columns={"ratio": "vivo_ratio"})

        v_nc_mean_ratio = np.mean(v_ratios['vivo_ratio'])
        v_ratios['vivo_ratio_normalised'] = v_ratios['vivo_ratio'] / v_nc_mean_ratio

        
        
        # Silico
        s_ratios = pd.merge(a.features.mean_psth.df.reset_index(), 
                            a.features.max_psth.df.reset_index())\
                            .etl.q(window=a.analysis_config.custom['evoked_window_for_custom_post_analysis']
#                                    neuron_class=comp_ncs
                                   )

        s_ratios = pd.merge(s_ratios,\
                            a.custom['sim_mask'],\
                            on=['simulation_id', 'window', 'circuit_id'])\
                            .etl.q(SimBad100p50p25pDecayOverlySustained=False)

        s_ratios.loc[:, 'ratio'] = s_ratios['max_psth'] / s_ratios['mean_psth']

        mean_ratio_by_sim = s_ratios \
                            .etl.q(neuron_class=comp_ncs)\
                            .groupby(['simulation_id'])\
                            .mean(numeric_only=True)\
                            .reset_index() \
                            .rename(columns={"ratio":"mean_ratio"})

        mean_ratio_by_sim['mean_ratio_diff_to_v'] = mean_ratio_by_sim['mean_ratio'] - v_nc_mean_ratio
        



        s_ratios = pd.merge(s_ratios, mean_ratio_by_sim.loc[:, ['simulation_id', 'mean_ratio', 'mean_ratio_diff_to_v']], on=['simulation_id'])
        s_ratios['ratio_normalised'] = s_ratios['ratio'] / s_ratios['mean_ratio']

        c_etl.add_custom_df(a, "s_ratios", s_ratios)
        c_etl.add_custom_df(a, "mean_ratio_by_sim", mean_ratio_by_sim)

        silico_mean_ratios_across_sims = s_ratios.groupby(['neuron_class'], observed=False)['ratio'].median()
        silico_ratio_std_across_sims = s_ratios.groupby(['neuron_class'], observed=False)['ratio'].std()
        

       

def create_sim_mask(a, comp_ncs):

	filt_timepoints_comp = a.custom['timepoints_comp'].etl.q(neuron_class=comp_ncs)

	# Check if any of these tests failed for any of the neurona classes (including all population if in comp_ncs)
	test_keys = ['Bad25pDecay', 'Bad50pDecay', 'Bad100pDecay', 'Bad50p25pDecay', 'Bad100p50p25pDecay', 'Bad100p50p25pDecayOverlySustained']
	sim_mask = filt_timepoints_comp.groupby(by=['simulation_id', 'window'])[test_keys].sum().astype('bool')

	# Rename columns. Now have one row per simulation.
	sim_mask = sim_mask.rename(columns={"Bad25pDecay": "SimBad25pDecay", 
										"Bad50pDecay": "SimBad50pDecay", 
										"Bad100pDecay": "SimBad100pDecay", 
										"Bad100p50p25pDecay": "SimBad100p50p25pDecay", 
										"Bad100p50p25pDecayOverlySustained":"SimBad100p50p25pDecayOverlySustained"}).reset_index()

	sim_mask = pd.merge(a.repo.simulations.df, sim_mask, on=['simulation_id']) #.drop(columns=[])

	c_etl.add_custom_df(a, "sim_mask", sim_mask)


def merge_comparison_and_mask(a):

	timepoints_comp_w_mask = pd.merge(a.custom['timepoints_comp'], a.custom['sim_mask'], on=['simulation_id', 'window'])
	c_etl.add_custom_df(a, "timepoints_comp_w_mask", timepoints_comp_w_mask)


def filter_dfs_by_window(a):

	window = a.analysis_config.custom['evoked_window_for_custom_post_analysis']
	add_custom_df(a, "window_timepoints_comp_w_mask", 	a.custom['timepoints_comp_w_mask'].etl.q(window=window))
	add_custom_df(a, "window_sim_mask", 				a.custom['sim_mask'].etl.q(window=window))


"""
Helpers
"""
def add_custom_df(a, key, df):
	if ('custom' not in a.__dict__.keys()):
		a.custom = {}
	a.custom[key] = df

def create_dirs(a):
    a.analysis_config.custom['masks_dir'] = str(a.figpaths.evoked) + "/masks/"
    a.analysis_config.custom['psths_dir'] = str(a.figpaths.evoked) + "/psths/"
    a.analysis_config.custom['heatmaps_dir'] = str(a.figpaths.evoked) + "/heatmaps/"
    a.analysis_config.custom['time_course_comp_dir'] = str(a.figpaths.evoked) + "/time_course_comp/"
    os.makedirs(a.analysis_config.custom['masks_dir'], exist_ok=True); os.makedirs(a.analysis_config.custom['psths_dir'], exist_ok=True); os.makedirs(a.analysis_config.custom['heatmaps_dir'], exist_ok=True); os.makedirs(a.analysis_config.custom['time_course_comp_dir'], exist_ok=True)


"""
Metric calculations
"""
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
