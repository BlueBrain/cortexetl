"""
PREPROCESSING
"""


import math
def deg_to_rad(dr):
    return (dr*math.pi)/180

def add_rotation_column(df, a, log_windows=False):
    
    # return pd.merge(a.repo.windows.df.etl.q(window=rotation_windows).loc[:, ['simulation_id', 'window']], df, on=['simulation_id', 'window'])
    # STANDARD MERGE CAUSED MEMORY PROBLEMS SO DOING ROW-BY-ROW
    
    for simulation_id in df.simulation_id.unique():
        for window in df.window.unique():
            if log_windows:
                print(simulation_id, window)
            filtered_df = df.etl.q(simulation_id=simulation_id, window=window)
            window_row = a.repo.windows.df.etl.q(simulation_id=simulation_id, window=window).iloc[0]
    
            df.loc[filtered_df.index, 'rotation'] = window_row['rotation']
            df.loc[filtered_df.index, 'rotation_radians'] = window_row['rotation_radians']

    return df

def rotation_windows_and_info(df, rotation_windows, neuron_classes, a, log_windows=False):
    df = df.etl.q(window=rotation_windows, neuron_class=neuron_classes).reset_index().drop(['circuit_id'], axis=1)    
    df = add_rotation_column(df, a, log_windows=log_windows)
    return df


def process_features_and_spikes(a, rotation_windows, neuron_class):
    # Features
    features_by_gid_rotation_windows = rotation_windows_and_info(a.features.by_gid.df, rotation_windows, neuron_class, a)
    features_by_gid_and_trial_rotation_windows = rotation_windows_and_info(a.features.by_gid_and_trial.df, rotation_windows, neuron_class, a, log_windows=True)

    # Spiking neurons only
    features_by_spiking_gid_rotation_windows = features_by_gid_rotation_windows.dropna(axis='rows', subset=["first_spike_time_means_cort_zeroed"])
    _, features_by_spiking_gid_rotation_windows["spiking_neuron_class_index"] = np.unique(features_by_spiking_gid_rotation_windows["neuron_class_index"], return_inverse=True)

    # Spikes
    spikes_rotation_windows = rotation_windows_and_info(a.repo.report.df, rotation_windows, neuron_class, a)
    spikes_rotation_windows['spiking_neuron_class_index'] = np.nan

    uniq_rotations = spikes_rotation_windows.rotation.unique()
    uniq_trials = spikes_rotation_windows.trial.unique()
    num_trials = len(uniq_trials)
    bins = np.arange(7.0, 15.0, 0.5)
    num_bins = len(bins) - 1
    
    nc_spikes_rotation_windows = spikes_rotation_windows.etl.q(neuron_class=neuron_class)
    spiking_gids, spiking_neuron_class_indices = np.unique(nc_spikes_rotation_windows["gid"], return_inverse=True)
    spikes_rotation_windows.loc[nc_spikes_rotation_windows.index, "spiking_neuron_class_index"] = spiking_neuron_class_indices
    nc_spikes_rotation_windows.loc[nc_spikes_rotation_windows.index, "spiking_neuron_class_index"] = spiking_neuron_class_indices
    num_spiking_gids = len(spiking_gids)

    # histograms_by_rotation_and_trial
    histograms_by_rotation_and_trial = {}
    for rotation in uniq_rotations:
        histograms_by_rotation_and_trial[rotation] = np.zeros((num_trials, num_spiking_gids, num_bins), dtype=np.int8)
        rot_nc_spikes_rotation_windows = nc_spikes_rotation_windows.etl.q(rotation=rotation)

        for trial in uniq_trials:
            trial_rot_nc_spikes_rotation_windows = rot_nc_spikes_rotation_windows.etl.q(trial=trial)
            binned_spike_counts = np.histogram2d(trial_rot_nc_spikes_rotation_windows['spiking_neuron_class_index'], trial_rot_nc_spikes_rotation_windows['time'],  bins=[np.arange(0.5, num_spiking_gids + 1, 1), bins])[0]
            histograms_by_rotation_and_trial[rotation][trial, :, :] = binned_spike_counts
    
            
    # trial_averaged_histograms
    trial_averaged_histograms = {}
    for neuron_class in spikes_rotation_windows.neuron_class.unique():
        for rotation in uniq_rotations:
            trial_averaged_histograms[rotation] = np.mean(histograms_by_rotation_and_trial[rotation], axis=0)

    data_with_rotation_info = {'uniq_rotations': uniq_rotations,
                            'features_by_gid_rotation_windows': features_by_gid_rotation_windows,
                              'features_by_gid_and_trial_rotation_windows': features_by_gid_and_trial_rotation_windows,
                              'features_by_spiking_gid_rotation_windows': features_by_spiking_gid_rotation_windows,
                                'histogram_gids': spiking_gids,
                              'histograms_by_rotation_and_trial': histograms_by_rotation_and_trial,
                              'trial_averaged_histograms': trial_averaged_histograms}
    
    return data_with_rotation_info





"""
FIGURES
"""


import matplotlib.pyplot as plt
import numpy as np

def polar_tuning_curve_plots(features_by_spiking_gid_rotation_windows, neuron_classes, stat_key, mean_spike_count_threshold=-1):
    for neuron_class in neuron_classes:
        for i in np.unique(features_by_spiking_gid_rotation_windows.etl.q(neuron_class=neuron_class)['spiking_neuron_class_index']):
            n_df = features_by_spiking_gid_rotation_windows.etl.q(spiking_neuron_class_index=i, neuron_class=neuron_class)

#             n_df = n_df[n_df['mean_spike_counts'] > mean_spike_count_threshold] 

            n_df.sort_values(by=['rotation'], inplace=True)
    
            if len(n_df) == 40:

                r = n_df[stat_key]
                theta = n_df['rotation_radians']

                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                ax.plot(theta, r)
                ax.set_rmax(r.max())
                # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
                ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
                ax.grid(True)

                ax.set_title("A line plot on a polar axis", va='bottom')
#                 plt.savefig(str(n_df['spiking_neuron_class_index'].iloc[0]))
                plt.show()
                plt.close()