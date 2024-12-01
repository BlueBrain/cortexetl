import pandas as pd
import blueetl



spikes_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/cortex_etl_output/2-PfrTransfer/2-PfrTransfer-6-3rdConnectionRemaining-etl3/hex0_spikes/repo/report.parquet").reset_index()
simulations_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/cortex_etl_output/2-PfrTransfer/2-PfrTransfer-6-3rdConnectionRemaining-etl3/hex0_spikes/repo/simulations.parquet").reset_index()
sim_info_spikes_df = pd.merge(simulations_df.loc[:, ['simulation_id', 'ca', 'desired_connected_proportion_of_invivo_frs', 'depol_stdev_mean_ratio']], spikes_df, on='simulation_id')


print(sim_info_spikes_df)




# features_by_gid_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/cortex_etl_output/2-PfrTransfer/2-PfrTransfer-6-3rdConnectionRemaining-etl3/hex0_spikes/features/by_gid.parquet").reset_index()
# simulations_df = pd.read_parquet("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/cortex_etl_output/2-PfrTransfer/2-PfrTransfer-6-3rdConnectionRemaining-etl3/hex0_spikes/repo/simulations.parquet").reset_index()

# # Add simulation info (i.e. meta parameters) to features dataframe
# sim_info_features_by_gid_df = pd.merge(simulations_df.loc[:, ['simulation_id', 'ca', 'desired_connected_proportion_of_invivo_frs', 'depol_stdev_mean_ratio']], features_by_gid_df, on='simulation_id')


# # Filter for a particular set of meta-parameters
# single_sim_features_by_gid_df = sim_info_features_by_gid_df.etl.q(ca=1.1, desired_connected_proportion_of_invivo_frs=0.5, depol_stdev_mean_ratio=0.4)

# # Merge connected + unconnected FRs
# conn_frs = single_sim_features_by_gid_df.reset_index().etl.q(window='conn_spont').loc[:, ['gid', 'neuron_class', 'mean_firing_rates_per_second']]
# unconn_frs = single_sim_features_by_gid_df.reset_index().etl.q(window='unconn_2nd_half').loc[:, ['gid', 'neuron_class', 'mean_firing_rates_per_second']]
# print(conn_frs)




# joint_frs = pd.merge(conn_frs, unconn_frs, on=['gid', 'neuron_class'], suffixes=['_CONN', '_UNCONN'])
# # print(joint_frs)

# # Calculate ratio
# joint_frs["CONN_UNCONN_RATIO"] = joint_frs['mean_firing_rates_per_second_CONN'] / joint_frs['mean_firing_rates_per_second_UNCONN']


# # Filter by a particular neuron class
# print(joint_frs.etl.q(neuron_class="L5_EXC"))

# # Filter for a the ALL neuron class
# print(joint_frs.etl.q(neuron_class="ALL"))

