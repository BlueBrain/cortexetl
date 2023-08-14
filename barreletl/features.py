import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from scipy.interpolate import interp1d
from blueetl.constants import GID, TIME
from .histogram import normalize_histogram, smooth_histogram, onset_from_spikes


def extract_latency_index(histogram):
    """Finds the time index corresponding to the maximum value in a given histogram.

    Args:
        histogram (numpy.ndarray): A 1-dimensional array containing the histogram data.

    Returns:
        int: The index of the maximum value in the histogram, representing the time at which
             the event with the highest frequency occurred.

             Returns -1 if the histogram is empty.
    """
    latency = np.argwhere(histogram == np.max(histogram)).flatten()
    if np.size(latency) == 0:
        return -1
    return latency[0]


def extract_decay(histogram, time, ratio=0.5):
    """Extracts the time at which the histogram decays to a specified ratio of its maximum value.

    Args:
        histogram (numpy.ndarray): A 1-dimensional array containing the histogram data.
        time (numpy.ndarray): A 1-dimensional array containing the time values corresponding to the histogram data.
        ratio (float, optional): The ratio of the maximum histogram value at which to calculate the decay time. Defaults to 0.5.

    Returns:
        float: The time at which the histogram decays to the specified ratio of its maximum value.

               Returns -1 if the histogram is empty, if the peak occurs at the end of the histogram or if there are too few time points after the peak to calculate the decay time.
    """
    peak_index = extract_latency_index(histogram)
    if (
        (peak_index == -1)
        or (peak_index + 1 >= np.shape(histogram))
        or (np.size(time[peak_index + 1 :]) < 5)
    ):
        return -1

    interp1d_func = interp1d(
        time[peak_index + 1 :], histogram[peak_index + 1 :], kind="cubic"
    )
    t_stop = time.max()
    bin_size = np.diff(time)[0]
    interp_time = np.arange(time[peak_index + 1], t_stop - 1, bin_size / 10)
    interp_histogram = interp1d_func(interp_time)

    decay_index = np.argwhere(interp_histogram <= ratio).flatten()
    if np.size(decay_index) == 0:
        return -1
    return interp_time[decay_index[0]]


def extract_baseline(histogram, offset, duration=10, bin_size=1, mean=False):
    """Extracts the baseline activity level of a histogram.

    Args:
        histogram (numpy.ndarray): A 1-dimensional array containing the histogram data.
        offset (int): The time bin index around which to extract the baseline activity level.
        duration (int, optional): The duration of the baseline period to extract, in time bins. Defaults to 10.
        bin_size (int, optional): The size of the time bins in the histogram, in seconds. Defaults to 1.
        mean (bool, optional): Whether to return the mean activity level instead of the minimum. Defaults to False.

    Returns:
        float: The baseline activity level of the histogram, either as the minimum value or the mean value of the selected time bin range.

    Raises:
        ValueError: If the specified offset or duration results in an index out of range for the histogram array.

    Examples:
        >>> histogram = np.array([1, 2, 3, 2, 1])
        >>> extract_baseline(histogram, 2, duration=2)
        1
        >>> extract_baseline(histogram, 2, duration=2, mean=True)
        2
    """
    if offset < duration or offset >= len(histogram) or offset < 0:
        raise ValueError("Invalid offset value")

    baseline_range = histogram[(offset - duration) * bin_size : offset * bin_size]
    if mean:
        return np.mean(baseline_range)

    baseline = np.min(baseline_range)
    return baseline


def get_PSTH(spiketrains, t_start, t_stop, bin_size=2):
    """Calculate the peristimulus time histogram (PSTH) for a given population of neurons.

    The PSTH is a histogram that counts the number of spikes in a given time bin,
    aligned with a stimulus or event onset. This function calculates the PSTH for
    the selected population of neurons.

    Args:
        spiketrains (np.ndarray): A 1D or 2D numpy array of all spike times. If 2D,
            each row represents a different neuron's spike train.
        t_start (float): The start time of the PSTH, in milliseconds.
        t_stop (float): The end time of the PSTH, in milliseconds.
        bin_size (int, optional): The size of each time bin in the PSTH, in milliseconds.
            Defaults to 2.

    Returns:
        pd.DataFrame: A pandas DataFrame with two columns: 'time' and 'psth'. Each row
            represents a time bin and the corresponding spike count in that bin.

    """
    spiketrains = np.concatenate(spiketrains)
    inputbins = np.arange(t_start, t_stop, bin_size)
    counts, bins = np.histogram(
        spiketrains,
        bins=inputbins.size,
        range=(t_start, t_stop),
    )
    return pd.DataFrame([counts, bins[:-1]], index=["psth", "time"]).T


def get_normalized_PSTH(spiketrains, t_start, t_stop, bin_size=2):
    """Get a normalized peristimulus time histogram. The maximum value of the
    histogram is set to 1.

    Args:
        spiketrains (np.ndarray): An array of all spike times.
        t_start (float): The start time of the PSTH.
        t_stop (float): The end time of the PSTH.
        bin_size (int, optional): The size of the histogram bin in milliseconds. Defaults to 2.

    Returns:
        pd.DataFrame: A pandas DataFrame with two columns - time bin and PSTH bin count.
    """

    histogram = get_PSTH(spiketrains, t_start, t_stop, bin_size=bin_size)
    histogram.psth = normalize_histogram(histogram.psth)

    return histogram


def get_baseline(spiketrains, t_start, t_stop, duration=10, bin_size=1, mean=False):
    """
    Get the baseline activity of a population of neurons.

    Args:
        spiketrains (np.ndarray): An array of spike times.
        t_start (float): The start time of the time period to analyze.
        t_stop (float): The end time of the time period to analyze.
        duration (float, optional): The duration in seconds of the baseline period. Default is 10.
        bin_size (float, optional): The bin size in milliseconds to use for binning spike times. Default is 1.
        mean (bool, optional): Whether to return the mean of the baseline period. Default is False.

    Returns:
        pd.DataFrame: A dataframe with one column - the baseline activity.

    """
    histogram = get_normalized_PSTH(
        spiketrains, t_start, t_stop, bin_size=bin_size
    ).psth.values
    offset = np.argwhere(histogram.time.values == t_stim).flatten()[0]

    return pd.DataFrame(
        [
            extract_baseline(
                histogram,
                offset=offset,
                duration=duration,
                bin_size=bin_size,
                mean=mean,
            )
        ],
        columns=["baseline"],
    )


def get_onset(
    spiketrains,
    t_start,
    t_stop,
    bin_size=2,
    duration=10,
    sigma=1,
    t_stim=0,
    threshold_multiple=4,
):
    """Detect the onset of the response to a stimulus in the population of neurons represented by the spike trains.

    The onset is determined as the first time point after the stimulus time at which the peristimulus time histogram
    (PSTH) exceeds a threshold that is a multiple of the standard deviation of the baseline activity. The baseline activity
    is estimated by computing the PSTH in a time window preceding the stimulus onset.

    Args:
        spiketrains (np.ndarray): array of spike times of shape (N,), where N is the number of spikes.
        t_start (float): start time of the PSTH.
        t_stop (float): end time of the PSTH.
        bin_size (int, optional): size of histogram bin in milliseconds. Defaults to 2.
        duration (int, optional): duration of the baseline window in milliseconds. Defaults to 10.
        sigma (float, optional): standard deviation factor used to determine the onset threshold. Defaults to 1.
        t_stim (float, optional): time of the stimulus onset in milliseconds. Defaults to 0.
        threshold_multiple (float, optional): factor of the standard deviation used as threshold. Defaults to 4.

    Returns:
        pd.DataFrame: A dataframe with one column - the time of the onset.
    """
    histogram = get_baseline_PSTH(
        spiketrains, t_start, t_stop, bin_size, duration, sigma
    )
    return pd.DataFrame(
        [
            histogram.time.values[
                onset_from_spikes(
                    histogram.psth.values,
                    histogram.time_stimulus.values,
                    t_stim=t_stim,
                    threshold_multiple=threshold_multiple,
                )
            ]
        ],
        columns=["onset"],
    )


def get_smoothed_PSTH(
    spiketrains,
    t_start,
    t_stop,
    bin_size=2,
    sigma=1,
    offset=0,
):
    """
    Calculates a smoothed peristimulus time histogram.

    Args:
        spiketrains (np.ndarray): array of all spike times
        t_start (float): start time of the PSTH
        t_stop (float): end time of the PSTH
        bin_size (int, optional): size of histogram bin in milliseconds. Defaults to 2.
        sigma (int, optional): standard deviation of the Gaussian filter applied for smoothing. Defaults to 1.
        offset (float, optional): offset to add to the time axis. Defaults to 0.

    Returns:
        pd.DataFrame: A dataframe with three columns - time bin, PSTH bin count, and time stimulus.
            The time bin column is shifted by the given offset, and the PSTH bin count column is smoothed.
    """
    histogram = get_PSTH(spiketrains, t_start, t_stop, bin_size=bin_size)
    histogram.psth = smooth_histogram(histogram.psth, sigma=sigma)
    histogram.loc[:, "time_stimulus"] = histogram.time

    histogram.loc[:, "time"] = histogram.time + offset
    return histogram


def get_baseline_PSTH(
    spiketrains,
    t_start,
    t_stop,
    bin_size=2,
    duration=10,
    sigma=1,
    t_stim=0,
    threshold_multiple=4,
    offset=False,
):
    """
    Computes the baseline-subtracted and onset-aligned peristimulus time histogram (PSTH) of the selected population.

    Args:
        spiketrains (np.ndarray): Array of all spike times.
        t_start (float): Start time of the PSTH.
        t_stop (float): End time of the PSTH.
        bin_size (int, optional): Size of histogram bin in milliseconds. Defaults to 2.
        duration (int, optional): Duration in milliseconds of the baseline window before stimulus onset. Defaults to 10.
        sigma (int, optional): Standard deviation of the Gaussian filter applied to the PSTH before baseline subtraction. Defaults to 1.
        t_stim (float, optional): Time of stimulus onset. Defaults to 0.
        threshold_multiple (float, optional): Number of standard deviations above the mean of the baseline to set the onset threshold. Defaults to 4.
        offset (bool, optional): Whether to return the PSTH with time offset by the stimulus onset time. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe with three columns - time, time_stimulus, and PSTH bin count.
        The PSTH is baseline-subtracted and normalized to have maximum value of 1.
        If `offset` is True, the time column is offset by the stimulus onset time.
        Otherwise, the time column is onset-aligned and starts at zero.
    """
    histogram = get_normalized_PSTH(spiketrains, t_start, t_stop, bin_size=bin_size)
    try:
        offset_index = np.argwhere(histogram.time.values == t_stim).flatten()[0]
    except:
        offset_index = 0
    histogram.psth = smooth_histogram(histogram.psth, sigma=sigma)
    baseline = extract_baseline(
        histogram.psth, offset=np.abs(offset_index), duration=duration
    )

    histogram.psth = histogram.psth - baseline
    histogram.psth = normalize_histogram(histogram.psth)
    histogram.loc[:, "time_stimulus"] = histogram.time

    if offset:
        histogram.loc[:, "time"] = histogram.time + offset
        return histogram

    onset_time = histogram.time.values[
        onset_from_spikes(
            histogram.psth.values,
            histogram.time.values,
            t_stim=t_stim,
            threshold_multiple=threshold_multiple,
        )
    ]
    histogram.loc[:, "time"] = histogram.time - onset_time
    return histogram


def get_latency(histogram, onset=False):
    """
    Extract the latency of a normalized histogram by finding the time index of the population maximum = 1.

    Args:
        histogram (pd.DataFrame): a dataframe containing the histogram data.
        onset (bool, optional): if True, calculate the latency from the onset of the stimulus instead of the start of the recording. Defaults to False.

    Returns:
        pd.DataFrame: a dataframe containing the latency value in milliseconds.
    """
    if onset:
        return pd.DataFrame(
            [
                histogram.time_stimulus.values[
                    extract_latency_index(histogram.psth.values)
                ]
            ],
            columns=["latency"],
        )
    return pd.DataFrame(
        [histogram.time.values[extract_latency_index(histogram.psth.values)]],
        columns=["latency"],
    )


def get_decay(histogram, ratio=[0.5]):
    """Extract the decay time of a normalized histogram. The decay time is the time it takes for the population firing
    rate to decay from the maximum of 1 to a specified ratio (default is 0.5). Returns a pandas DataFrame containing the
    decay times in milliseconds for each specified ratio.

    Args:
        histogram (pd.DataFrame): The normalized histogram to extract decay from.
        ratio (list, optional): A list of ratios to calculate decay times for. Defaults to [0.5].

    Returns:
        pd.DataFrame: A DataFrame with columns for "decay" (the decay time in ms) and "ratio" (the specified decay ratio).
    """
    if not isinstance(ratio, list):
        ratio = [ratio]

    decays = []
    for r in ratio:
        decays.append(
            pd.DataFrame(
                [
                    extract_decay(histogram.psth.values, histogram.time.values, r),
                    r,
                ],
                index=["decay", "ratio"],
            ).T
        )

    return pd.concat(decays)


def get_max_psth(histogram, t_stim=0.0, prestimulus=False):
    """Computes the maximum value of the PSTH for a given histogram.

    Args:
        histogram (pd.DataFrame): The histogram of the neuron or neuronal population activity.
        t_stim (float, optional): The time of the stimulus onset in seconds. Defaults to 0.
        prestimulus (bool, optional): If True, the maximum spike rate is computed in the interval before the stimulus onset. Otherwise, the maximum spike rate is computed in the interval after the stimulus onset. Defaults to False.

    Returns:
        pd.DataFrame: with one row and one column:
    - "max_psth": float
        The maximum population spike rate in the selected interval of the PSTH
    """
    stimulus_index = np.argwhere(histogram.time.values >= t_stim).flatten()[0]
    if prestimulus:
        return pd.DataFrame(
            [np.max(histogram.psth.values[:stimulus_index])], columns=["max_psth"]
        )
    else:
        return pd.DataFrame(
            [np.max(histogram.psth.values[stimulus_index:])], columns=["max_psth"]
        )


def get_mean_psth(histogram, t_stim=0, prestimulus=True):
    """
    Computes the mean peri-stimulus time histogram (PSTH) from a given histogram.

    Args:
        histogram (pd.DataFrame): A histogram with time and PSTH values.
        t_stim (float, optional): Time of the stimulus onset. Defaults to 0.
        prestimulus (bool, optional): If True, computes mean PSTH for the period before stimulus onset. If False, computes mean PSTH for the period after stimulus onset.Defaults to True.

    Returns:
     pd.DataFrame: A DataFrame with a single row containing the mean PSTH value.

    Raises:
        IndexError: If `t_stim` is greater than the maximum time value in the histogram.
    """
    stimulus_index = np.argwhere(histogram.time.values >= t_stim).flatten()[0]
    if prestimulus:
        return pd.DataFrame(
            [np.mean(histogram.psth.values[:stimulus_index])], columns=["mean_psth"]
        )
    else:
        return pd.DataFrame(
            [np.mean(histogram.psth.values[stimulus_index:])], columns=["mean_psth"]
        )


def calculate_features_by_neuron_class(repo, key, df, params):
    """Calculates population-level features grouped by neuron class from a given data frame of spike trains, extracts population level features.

    Args:
        repo (object): A repository object containing the data.
        key (str): A key to access the data.
        df (pd.DataFrame): A data frame containing spike train information.
        params (dict): A dictionary containing configuration parameters for feature extraction.

    Returns:
        dict: A dictionary containing the calculated features, with feature names as keys and feature values as values.

    Raises:
        KeyError: If 'window' is not a key in the given `key`.
        ValueError: If the 'window' key in `key` is invalid.
    """
    params = deepcopy(params)
    prefix = ""
    if "prefix" in params.keys():
        prefix = params.pop("prefix")

    t_start, t_stop = repo.windows.get_bounds(key.window)

    # create an array containing multiple arrays of spikes, one for each gid
    spiketrains = df.groupby([GID])[TIME].apply(np.array).to_numpy()
    functions = {
        f"{prefix}PSTH": partial(get_PSTH, spiketrains, t_start=t_start, t_stop=t_stop),
        f"{prefix}normalized_PSTH": partial(
            get_normalized_PSTH, spiketrains, t_start=t_start, t_stop=t_stop
        ),
        f"{prefix}smoothed_PSTH": partial(
            get_smoothed_PSTH, spiketrains, t_start=t_start, t_stop=t_stop
        ),
        f"{prefix}baseline_PSTH": partial(
            get_baseline_PSTH, spiketrains, t_start=t_start, t_stop=t_stop
        ),
        f"{prefix}baseline": partial(
            get_baseline, spiketrains, t_start=t_start, t_stop=t_stop
        ),
        f"{prefix}onset": partial(
            get_onset, spiketrains, t_start=t_start, t_stop=t_stop
        ),
    }
    psth_name = f"{prefix}baseline_PSTH"
    if "psth_name" in params.keys():
        psth_name = f"{prefix}{params.pop('psth_name')}"

    psth_params = params[psth_name].get("params", {})
    _ = params.pop(psth_name)
    histogram = functions[psth_name](**psth_params)

    functions_psth = {
        f"{prefix}latency": partial(get_latency, histogram),
        f"{prefix}decay": partial(get_decay, histogram),
        f"{prefix}max_psth": partial(get_max_psth, histogram),
        f"{prefix}mean_psth": partial(get_mean_psth, histogram),
    }
    result = {}
    result[psth_name] = histogram
    for feature_name, feature_config in params.items():
        if type(feature_config) is dict:
            feature_params = feature_config.get("params", {})
            if feature_name in functions.keys():
                result[feature_name] = functions[feature_name](**feature_params)
            else:
                result[feature_name] = functions_psth[feature_name](**feature_params)
        else:
            result[feature_name] = functions_psth[feature_name]()
    return result
