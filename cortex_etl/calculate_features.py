# adapted from BlueNetworkActivityComparison/bnac/data_processor.py
import logging

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from blueetl.constants import BIN, COUNT, GID, NEURON_CLASS_INDEX, TIME, TRIAL

L = logging.getLogger(__name__)
FIRST = "first"


def get_initial_spiking_stats(repo, key, df, params):
    duration = repo.windows.get_duration(key.window)

    # df with index (trial, gid) and columns (count, times)
    spikes_by_trial = df.groupby([TRIAL, GID, NEURON_CLASS_INDEX])[TIME].agg(
        **{
            COUNT: "count",
            FIRST: "min",
            # TIMES: lambda x: [i for i in x if not np.isnan(i)],  # slow
        }
    )
    # first spike for each trial and gid, averaged across all trials where the neuron was present
    first_spike_time_means_cort_zeroed = (
        spikes_by_trial[FIRST]
        .groupby([GID, NEURON_CLASS_INDEX])
        .mean()
        .rename("first_spike_time_means_cort_zeroed")
    )

    mean_spike_counts = spikes_by_trial[COUNT].fillna(0).groupby([GID, NEURON_CLASS_INDEX]).mean()
    mean_spike_counts = mean_spike_counts.rename("mean_spike_counts")
    mean_of_spike_counts_for_each_trial = (
        spikes_by_trial[COUNT]
        .fillna(0)
        .groupby(TRIAL)
        .mean()
        .rename("mean_of_spike_counts_for_each_trial")
    )

    mean_firing_rates_per_second = mean_spike_counts * 1000.0 / duration
    mean_firing_rates_per_second = mean_firing_rates_per_second.rename(
        "mean_firing_rates_per_second"
    )

    return {
        # by_gid_and_trial
        "spikes_by_trial": spikes_by_trial,
        # by_gid
        "first_spike_time_means_cort_zeroed": first_spike_time_means_cort_zeroed,
        "mean_spike_counts": mean_spike_counts,
        "mean_firing_rates_per_second": mean_firing_rates_per_second,
        # by_neuron_class_and_trial
        "mean_of_spike_counts_for_each_trial": mean_of_spike_counts_for_each_trial,
        # by_neuron_class (scalar values)
        "mean_of_mean_spike_counts": np.nanmean(mean_spike_counts),
        "mean_of_mean_firing_rates_per_second": np.mean(mean_firing_rates_per_second),
        "std_of_mean_firing_rates_per_second": np.std(mean_firing_rates_per_second),
    }


# def get_histogram_features(repo, key, df, hist):
#     number_of_trials = repo.windows.get_number_of_trials(key.window)
#     duration = repo.windows.get_duration(key.window)
#     t_start, t_stop = repo.windows.get_bounds(key.window)
#     # all the spike times are concatenated regardless of the trial
#     times = df[TIME].to_numpy()
#     hist, _ = np.histogram(times, range=[t_start, t_stop], bins=int(duration))
#     num_target_cells = len(
#         repo.neurons.df.etl.q(circuit_id=key.circuit_id, neuron_class=key.neuron_class)
#     )
#     hist = hist / (num_target_cells * number_of_trials)
#     min_hist = np.min(hist)
#     max_hist = np.max(hist)
#     norm_hist = hist / (max_hist or 1)
#     smoothed_hist = gaussian_filter(hist, sigma=4.0)
#     max_smoothed_hist = np.max(smoothed_hist)
#     norm_smoothed_hist = smoothed_hist / (max_smoothed_hist or 1)
#     return {
#         "hist": hist,
#         "mean_of_spike_times_normalised_hist_1ms_bin": np.mean(hist),
#         "min_of_spike_times_normalised_hist_1ms_bin": min_hist,
#         "max_of_spike_times_normalised_hist_1ms_bin": max_hist,
#         "argmax_spike_times_hist_1ms_bin": np.argmax(hist),
#         "spike_times_max_normalised_hist_1ms_bin": norm_hist,
#         "smoothed_3ms_spike_times_max_normalised_hist_1ms_bin": norm_smoothed_hist,
#     }

def get_histogram_features(repo, key, df, hist_params):

    number_of_trials = repo.windows.get_number_of_trials(key.window)
    duration = repo.windows.get_duration(key.window)
    t_start, t_stop = repo.windows.get_bounds(key.window)
    # all the spike times are concatenated regardless of the trial
    times = df[TIME].to_numpy()

    hist, _ = np.histogram(times, range=[t_start, t_stop], bins=int(duration / hist_params['bin_size']))
    num_target_cells = len(
        repo.neurons.df.etl.q(circuit_id=key.circuit_id, neuron_class=key.neuron_class)
    )
    hist = hist / (num_target_cells * number_of_trials)
    
    d = {"hist": hist}

    return d



    


def calculate_features_multi(repo, key, df, params):

    # print('hey')

    export_all_neurons = params.get("export_all_neurons", False)
    spiking_stats = get_initial_spiking_stats(repo, key, df, params)

    # df with (gid) as index, and features as columns
    by_gid = pd.concat(
        [
            spiking_stats["first_spike_time_means_cort_zeroed"],
            spiking_stats["mean_spike_counts"],
            spiking_stats["mean_firing_rates_per_second"],
        ],
        axis=1,
    )
    if not export_all_neurons:
        # return only neurons with spikes
        # TODO: to be tested
        by_gid = by_gid.dropna("all")

    # df with (trial, gid) as index, and features as columns
    by_gid_and_trial = pd.concat(
        [
            spiking_stats["spikes_by_trial"],
        ],
        axis=1,
    )
    if not export_all_neurons:
        # return only neurons with spikes
        # TODO: to be tested
        by_gid_and_trial = by_gid_and_trial.dropna("all")

    # df with features as columns and a single row
    # the index will be dropped when concatenating because it's unnamed
    by_neuron_class = pd.DataFrame(
        {
            "mean_of_mean_spike_counts": spiking_stats["mean_of_mean_spike_counts"],
            "mean_of_mean_firing_rates_per_second": spiking_stats[
                "mean_of_mean_firing_rates_per_second"
            ],
            "std_of_mean_firing_rates_per_second": spiking_stats[
                "std_of_mean_firing_rates_per_second"
            ],
        },
        index=[0],
    )

    # df with (trial) as index, and features as columns
    by_neuron_class_and_trial = spiking_stats["mean_of_spike_counts_for_each_trial"].to_frame()

    histograms = pd.DataFrame({})
    if ("histograms" in params.keys()):
        for hist_key, hist_params in params['histograms'].items():
            # print(hist_key)
            histogram_features = get_histogram_features(repo, key, df, hist_params)

            # df with (bin) as index, and features as columns
            histogram = pd.DataFrame({"hist": histogram_features["hist"]}).rename_axis(BIN)
            histogram['bin_size'] = hist_params['bin_size']
            histogram['smoothing_type'] = pd.Categorical(['None'])[0]
            # histogram['kernel_sd'] = pd.Categorical(['None'])[0]
            histogram['kernel_sd'] = -1.0

            # histograms.set_index(['bin_size', 'smoothing_type', 'kernel_sd'], append=True, inplace=True)

            if ('smoothing' in hist_params.keys()):
                for smoothing_key, smoothing_params in hist_params['smoothing'].items():

                    smoothed_histogram = pd.DataFrame({"hist": gaussian_filter(histogram_features["hist"], sigma=smoothing_params['kernel_sd'])}).rename_axis(BIN)
                    smoothed_histogram['bin_size'] = hist_params['bin_size']
                    smoothed_histogram['smoothing_type'] = smoothing_params['smoothing_type']
                    smoothed_histogram['kernel_sd'] = smoothing_params['kernel_sd']
                    
                    histogram = pd.concat([histogram, smoothed_histogram])

            histograms = pd.concat([histograms, histogram])
        # print(histograms)


    # print('Finished')

    return {
        "by_gid": by_gid,
        "by_gid_and_trial": by_gid_and_trial,
        "by_neuron_class": by_neuron_class,
        "by_neuron_class_and_trial": by_neuron_class_and_trial,
        "histograms": histograms,
    }
