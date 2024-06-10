import pandas as pd
import time
import cortexetl as c_etl


def load_custom_dataframes(a, df_keys):
    a.custom = {}
    for key in df_keys:
        df_path = str(a.analysis_config.output) + "/" + key + ".parquet"
        if (os.path.exists(df_path)):
            a.custom[key] = pd.read_parquet(df_path)


def persist_custom_dataframes(a):

    opts = {"engine": "pyarrow", "index": True}
    for key, df in a.custom.items(): 
        df.to_parquet(path=str(a.analysis_config.output) + "/" + key + ".parquet", **opts)


def add_sim_and_filters_info_to_df(a, custom_df_key):
    
    filter_keys = ['simulation_id', 
                    'ei_corr_r_out_of_range', 
                    'neuron_group_gt_threshold_fr', 
                    'bursting',
                    'bursting_or_fr_gt_threshold',
                    'bursting_or_fr_gt_threshold_or_ei_corr_r_out_of_range']
    
    a.custom[custom_df_key] = pd.merge(a.custom[custom_df_key], a.repo.simulations.df.loc[:, a.analysis_config.custom['independent_variables'] + ['simulation_id']])
    if (custom_df_key != 'by_simulation'):
        a.custom[custom_df_key] = pd.merge(a.custom[custom_df_key], a.custom['by_simulation'].loc[:, filter_keys])
    

import time
def post_analysis(a):

    print("\n----- Custom post analysis -----")
    tic = time.perf_counter()

    do_load_custom_dataframes = False
    if do_load_custom_dataframes: load_custom_dataframes(['by_simulation', 'by_neuron_class', 'fft', 'by_layer_and_simulation'])

    a.custom = {}
    c_etl.custom_by_simulation_features(a)
    c_etl.custom_by_neuron_class_features(a)
    c_etl.custom_by_layer_features(a)

    do_persist_custom_dataframes = True
    if do_persist_custom_dataframes: persist_custom_dataframes(a)
    
    print(f"----- Custom post analysis complete: {time.perf_counter() - tic:0.2f}s -----")

    