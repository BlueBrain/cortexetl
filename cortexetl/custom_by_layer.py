import numpy as np
import pandas as pd
from functools import partial
from blueetl.parallel import call_by_simulation
from itertools import chain
from scipy.stats import linregress
import cortexetl as c_etl


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




def calculate_layerwise_features(a, spont_hist_bin_size, spont_smoothing_type, spont_kernel_sd):

    dfs = {"simulation_windows": a.repo.windows.df, 
            "histograms": a.features.histograms.df.etl.q(neuron_class=c_etl.LAYER_EI_NEURON_CLASSES, window="conn_spont", bin_size=spont_hist_bin_size, smoothing_type=spont_smoothing_type, kernel_sd=spont_kernel_sd)}
    
    results = call_by_simulation(a.repo.simulations.df, 
                                    dfs, 
                                    func=partial(layer_wise_single_sim_analysis, analysis_config=a.analysis_config),
                                    how='series')
        
    a.custom['by_layer_and_simulation'] = pd.DataFrame.from_records(list(chain.from_iterable(results)))


def custom_by_layer_features(a):
	spont_window = "conn_spont"
	spont_hist_bin_size = 3.0
	spont_smoothing_type = 'Gaussian'
	spont_kernel_sd = 1.0

	calculate_layerwise_features(a, spont_hist_bin_size, spont_smoothing_type, spont_kernel_sd)
	c_etl.add_sim_and_filters_info_to_df(a, 'by_layer_and_simulation')
