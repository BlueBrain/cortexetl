import numpy as np
import pandas as pd
import bluepy
import bluepysnap
import cortex_etl as c_etl


def get_value_with_backup_nc(d, key_prefix, nc_map, nc, second_key='', backup_nc=''):

    if (key_prefix + nc_map[nc] in list(d.keys())):
        v = d[key_prefix + nc_map[nc]]
        if (second_key != ''):
            v = v[second_key]
        return v
    elif (key_prefix + nc_map[backup_nc] in list(d.keys())):
        v = d[key_prefix + nc_map[backup_nc]]
        if (second_key != ''):
            v = v[second_key]
        return v
    else:
        return np.nan

def get_value_from_instance(row, value_key, a, map_to_use=None):

    sim_inst = a.repo.simulations.df.etl.q(simulation_id=row.simulation_id).iloc[0].simulation.instance
    sim_conf = sim_inst.config
    
    # print(a.repo.simulations.df.iloc[0].simulation.instance)
    if not isinstance(a.repo.simulations.df.iloc[0].simulation.instance, bluepysnap.simulation.Simulation):
        fr_conf = sim_conf.Run
        inputs_conf = sim_conf
        second_mean_key = 'MeanPercent'
        second_std_key = 'SDPercent'
        stim_spacer = '_'
    else:
        fr_conf = sim_conf
        inputs_conf = sim_conf['inputs']
        second_mean_key = 'mean_percent'
        second_std_key = 'sd_percent'
        stim_spacer = ' '


    backup_nc = c_etl.backup_ncs[row.neuron_class]

    if value_key == "desired_connected_fr_key":
        des_conn_fr_key = a.analysis_config.custom['desired_connected_fr_key'] + '_'
        return get_value_with_backup_nc(fr_conf, des_conn_fr_key, map_to_use, row.neuron_class, backup_nc=backup_nc)
    elif value_key == "desired_unconnected_fr_key":
        des_unconn_fr_key = a.analysis_config.custom['desired_unconnected_fr_key'] + '_'
        return get_value_with_backup_nc(fr_conf, des_unconn_fr_key, map_to_use, row.neuron_class, backup_nc=backup_nc)    
    elif value_key == "MeanPercent":
        mean_perc_key = "Stimulus" + stim_spacer  + a.analysis_config.custom['depol_bc_key'] + '_'
        return get_value_with_backup_nc(inputs_conf, mean_perc_key, map_to_use, row.neuron_class, second_key=second_mean_key, backup_nc=backup_nc)
    elif value_key == "SDPercent":
        std_perc_key = "Stimulus" + stim_spacer + a.analysis_config.custom['depol_bc_key'] + '_'
        return get_value_with_backup_nc(inputs_conf, std_perc_key, map_to_use, row.neuron_class, second_key=second_std_key, backup_nc=backup_nc)


def neuron_class_mean_input_conductances(row, input_conductance_by_neuron_class_df):

    resting_cond = input_conductance_by_neuron_class_df.etl.q(neuron_class=row.neuron_class)['resting_conductance']
    if len(resting_cond):
        return resting_cond.iloc[0] * row['depol_mean'] / 100.0
    else:
        return np.nan



def custom_by_neuron_class_features(a):

    a.custom['by_neuron_class'] = a.features.by_neuron_class.df.reset_index().etl.q(neuron_class=c_etl.__dict__[a.analysis_config.custom['fr_analysis_neuron_classes_constant']]).copy()
    c_etl.add_sim_and_filters_info_to_df(a, 'by_neuron_class')

    map_to_use = c_etl.bluepy_neuron_class_map_2

    a.custom['by_neuron_class']["desired_connected_fr"] = a.custom['by_neuron_class'].apply(lambda row: get_value_from_instance(row, "desired_connected_fr_key", a, map_to_use=c_etl.bluepy_neuron_class_map), axis = 1).astype(float)
    a.custom['by_neuron_class']['desired_unconnected_fr'] = a.custom['by_neuron_class'].apply(lambda row: get_value_from_instance(row, "desired_unconnected_fr_key", a, map_to_use=c_etl.bluepy_neuron_class_map), axis = 1).astype(float)
    a.custom['by_neuron_class']["depol_mean"] = a.custom['by_neuron_class'].apply(lambda row: get_value_from_instance(row, "MeanPercent", a, map_to_use=c_etl.bluepy_neuron_class_map_2), axis = 1).astype(float)
    a.custom['by_neuron_class']["depol_sd"] = a.custom['by_neuron_class'].apply(lambda row: get_value_from_instance(row, "SDPercent", a, map_to_use=c_etl.bluepy_neuron_class_map_2), axis = 1).astype(float)

    if ('desired_unconnected_fr' in c_etl.flatten(a.analysis_config.custom['fr_comparison_pairs'])):

        a.custom['by_neuron_class']['connection_fr_increase'] = a.custom['by_neuron_class']['mean_of_mean_firing_rates_per_second'] - a.custom['by_neuron_class']['desired_unconnected_fr']
        a.custom['by_neuron_class']['connection_fr_error'] = a.custom['by_neuron_class']['mean_of_mean_firing_rates_per_second'] - a.custom['by_neuron_class']['desired_connected_fr']
        a.custom['by_neuron_class']['connection_vs_unconn_proportion'] = a.custom['by_neuron_class']['mean_of_mean_firing_rates_per_second'] / a.custom['by_neuron_class']['desired_unconnected_fr']

    a.custom['by_neuron_class']['recorded_proportion_of_in_vivo_FR'] = a.custom['by_neuron_class'].apply(lambda row: a.features.by_neuron_class.df.etl.q(simulation_id=row.simulation_id, neuron_class=row.neuron_class, window=row.window).iloc[0]['mean_of_mean_firing_rates_per_second'] / a.analysis_config.custom['vivo_frs'][row.neuron_class], axis = 1).astype(float)
    
    input_conductance_by_neuron_class_df = pd.read_parquet(a.analysis_config.custom['input_conductance_by_neuron_class_parquet'])
    a.custom['by_neuron_class']["true_mean_conductance"] = a.custom['by_neuron_class'].apply(lambda row: neuron_class_mean_input_conductances(row, input_conductance_by_neuron_class_df), axis = 1).astype(float)


