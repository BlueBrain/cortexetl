import blueetl
import pandas as pd
from functools import partial
from typing import List, Union, Dict, Any, Optional
from blueetl.analysis import MultiAnalyzer
from blueetl.parallel import call_by_simulation
from .histogram import (
    apply_on_neuron_class,
    smooth_histogram,
    plot_2sim_PSTH_synapse_type,
    plot_2sim_PSTH_creline,
    normalize_histogram,
    onset_from_spikes,
)
from .features import (
    extract_baseline,
    extract_decay,
    extract_latency_index,
    get_max_psth,
    get_mean_psth,
)
from .mapping import (
    neuron_classes,
    neuron_layers,
    neuron_synapse_type,
    creline_classes,
    creline_layers,
    creline_mapping,
)


def _add_feature_summary_row(
    df: pd.DataFrame, simulation_id: str, window: str
) -> pd.DataFrame:
    """Add a summary row to the DataFrame with calculated features for the 'ALL' neuron_class category.

    The summary row contains a sum of all the features in the DataFrame, and is allocated to the specified simulation_id and window. The original DataFrame is not modified; a new DataFrame with the summary row added is returned.

    Args:
        df (pd.DataFrame): The DataFrame with calculated features to add the summary row to.
        simulation_id (str): The ID of the simulation to which the summary row should be allocated.
        window (str): The window to which the summary row should be allocated.

    Returns:
        pd.DataFrame: A new DataFrame with the summary row added, and the simulation_id and window columns set.
    """
    row = df.sum()
    row.neuron_class = "SUM"
    row = pd.DataFrame(row)

    df = pd.concat([df, row.T])
    df["simulation_id"] = simulation_id
    df["window"] = window

    return df


def zero_value(value: float, baseline: float) -> float:
    """Calculate the zero value of a given value with respect to a baseline.

    Args:
        value (float): The original value for which the zero value needs to be calculated.
        baseline (float): The baseline value to which the zero value needs to be calculated.

    Returns:
        float: The zero value of the original value with respect to the baseline.
    """
    if baseline < 0:
        return value + baseline
    elif baseline >= 0:
        return value - baseline


def subtract_baseline(
    invivo: pd.DataFrame,
    key: str,
    new_key: str = "psth_base",
    offset: int = 50,
    duration: int = 10,
) -> pd.DataFrame:
    """Subtract the baseline from a given key in the DataFrame for each neuron_class.

    Args:
        invivo (pd.DataFrame): The input DataFrame containing the key to subtract the baseline from.
        key (str): The name of the key in the DataFrame for which the baseline should be calculated and subtracted.
        new_key (str): The name of the new column to be added to the DataFrame containing the subtracted values.Defaults to 'psth_base'.
        offset (int): The offset in milliseconds from the start of the time series at which to begin the baseline extraction. Defaults to 50.
        duration (int): The duration in milliseconds over which to average the baseline values. Defaults to 10.

    Returns:
        pd.DataFrame: A new DataFrame containing the subtracted values for the specified key for each neuron_class, with a new column added named according to the 'new_key' argument. The resulting DataFrame has the same shape and index as the input DataFrame 'invivo', with the addition of the new column.
    """
    baseline = apply_on_neuron_class(
        invivo,
        extract_baseline,
        key=key,
        new_key="baseline",
        offset=offset,
        duration=duration,
    )

    psth_base = (
        pd.DataFrame(
            invivo.groupby("neuron_class")
            .apply(
                lambda df: zero_value(
                    df[key].values,
                    baseline.etl.q(
                        neuron_class=df.neuron_class.iloc[0]
                    ).baseline.values,
                )
            )
            .explode(),
        )
        .reset_index()
        .rename(columns={0: new_key})
    )
    psth_base["time"] = invivo["time"]
    psth_base.psth_base = psth_base.psth_base.astype(float)
    return pd.merge(invivo, psth_base, on=["neuron_class", "time"])


def find_onset_time_neuron_class(
    invivo: pd.DataFrame,
    t_stim: float = 0,
    threshold_multiple: int = 4,
    key: str = "psth_smooth",
    offset: bool = False,
) -> pd.DataFrame:
    """Find the onset time for each neuron_class in the DataFrame relative to the stimulus.

    This function calculates the onset time for each neuron_class in the DataFrame relative to the stimulus onset
    time 't_stim'. The onset time is defined as the first time point after 't_stim' at which the value for the
    specified key in the DataFrame exceeds a certain threshold, determined by the 'threshold_multiple' argument.
    The resulting DataFrame contains a new column with the onset times for each neuron_class.

    Args:
        invivo (pd.DataFrame): The input DataFrame containing the time series data for each neuron_class.
        t_stim (float): The time in milliseconds at which the stimulus was applied. Defaults to 0.
        threshold_multiple (int): The multiple of the standard deviation above the mean at which to set the
                                  threshold for detecting the onset. Defaults to 4.
        key (str): The name of the key in the DataFrame for which to find the onset times. Defaults to 'psth_smooth'.
        offset (bool): A flag indicating whether to add an offset value to the onset time for each neuron_class.
                       If True, an offset value is added to the onset time equal to the 'offset' argument.
                       If False, the onset time is calculated relative to the minimum time in the DataFrame.
                       Defaults to False.

    Returns:
        pd.DataFrame: A new DataFrame containing the same data as the input DataFrame 'invivo', with the addition
                      of a new column named 'time' that contains the onset times for each neuron_class. If the 'offset'
                      argument is True, the onset times are adjusted by the specified offset value.
    """

    invivo["time_stimulus"] = invivo["time"]
    invivo.drop("time", axis=1, inplace=True)

    if offset:
        time_onset = (
            pd.DataFrame(
                invivo.groupby("neuron_class")
                .apply(lambda df: df["time_stimulus"].values + offset)
                .explode(),
            )
            .reset_index()
            .rename(columns={0: "time"})
        )

    else:
        onsets = invivo.groupby("neuron_class").apply(
            lambda df: onset_from_spikes(
                df[key].values,
                df.time_stimulus.values,
                t_stim=t_stim,
                threshold_multiple=threshold_multiple,
            )
        )

        time_onset = (
            pd.DataFrame(
                invivo.groupby("neuron_class")
                .apply(
                    lambda df: df["time_stimulus"].values
                    - (onsets[df.neuron_class.iloc[0]] + df.time.min()),
                )
                .explode(),
            )
            .reset_index()
            .rename(columns={0: "time"})
        )
        print("test")

    time_onset.time = time_onset.time.astype(float)
    time_onset["time_stimulus"] = invivo["time_stimulus"]

    return pd.merge(invivo, time_onset, on=["neuron_class", "time_stimulus"])


def prepare_invivo(
    invivo: pd.DataFrame,
    key: str = "psth_mean",
    t_stim: float = 0,
    bounds: list = [-50, 250],
    threshold_multiple: int = 4,
    sigma: float = 1,
    offset: bool = False,
    normalize: bool = True,
) -> pd.DataFrame:
    """Prepares the invivo data for comparison with simulation data. It performs several preprocessing steps, including filtering the data to a specified time window, smoothing the spike count data with a Gaussian filter, subtracting the baseline from the smoothed data, and optionally normalizing the data. Finally, it computes the onset time of each neuron class relative to the stimulus onset.

    Parameters:
        invivo (pd.DataFrame): the input invivo data
        key (str): the column name for the spike count data (default: "psth_mean")
        t_stim (int or float): the stimulus onset time (default: 0)
        bounds (list of two int or float): the lower and upper bounds of the time window to keep (default: [-50, 250])
        threshold_multiple (int or float): the threshold multiple for detecting the onset time of each neuron class (default: 4)
        sigma (int or float): the standard deviation of the Gaussian filter for smoothing the spike count data (default: 1)
        offset (bool): whether to add a fixed offset to the onset time (default: False)
        normalize (bool): whether to normalize the data (default: True)

    Returns:
        pd.DataFrame: the preprocessed invivo data, with additional columns for the smoothed spike count data, the baseline-subtracted data, the normalized data (if normalize=True), and the onset time of each neuron class.
    """
    invivo = invivo.etl.q(time={"ge": bounds[0], "le": bounds[1]})

    invivo_smooth = apply_on_neuron_class(
        invivo, smooth_histogram, key=key, new_key="psth_smooth", sigma=sigma
    )
    invivo_smooth["time"] = invivo.time.values
    invivo = pd.merge(invivo, invivo_smooth, on=["neuron_class", "time"])

    invivo = subtract_baseline(invivo, key="psth_smooth", new_key="psth_base")

    if normalize:
        invivo_normalized = apply_on_neuron_class(
            invivo, normalize_histogram, key="psth_base", new_key="psth_norm"
        )
        invivo_normalized.psth_norm = invivo_normalized.psth_norm.astype(float)

        invivo_normalized["time"] = invivo.time.values
        invivo = pd.merge(invivo, invivo_normalized, on=["neuron_class", "time"])

        return find_onset_time_neuron_class(
            invivo, t_stim, threshold_multiple, key="psth_norm", offset=offset
        )
    return find_onset_time_neuron_class(
        invivo, t_stim, threshold_multiple, key="psth_base", offset=offset
    )


def latency_distance_2sim(
    baseline: pd.DataFrame, simulation_row: pd.Series, filtered_dataframes: dict
) -> pd.DataFrame:
    """
    Calculate the latency distance between a baseline and a simulation for a given set of filtered dataframes.

    Args:
        baseline (pd.DataFrame): A dataframe containing the baseline latency values for each neuron class.
        simulation_row (pd.Series): A series representing a single simulation and its metadata.
        filtered_dataframes (dict): A dictionary containing dataframes that have been filtered by neuron class and time window.

    Returns:
        pd.DataFrame: A summary of the latency distance between the baseline and simulation, including the mean and standard deviation for each neuron class.
    """
    latency = pd.merge(
        filtered_dataframes["latency"].reset_index()[
            ["neuron_class", "window", "latency"]
        ],
        baseline,
        on="neuron_class",
        suffixes=["_silico", "_vivo"],
    )
    latency["latency_diff"] = latency.apply(
        lambda row: row.latency_silico - row.latency_vivo, axis=1
    )

    return _add_feature_summary_row(
        latency, simulation_row.simulation_id, latency.iloc[0].window
    )


def silico_vivo_latency_distance(
    MultiAnalyzer: MultiAnalyzer,
    filtered_dataframes: Dict[str, pd.DataFrame],
    sigma: float = 1,
    vivo_key: str = "vivo_reyesp",
    offset: Union[bool, str] = False,
) -> pd.DataFrame:
    """
    Calculate the latency distance for each simulation in a blueetl.analysis.MultiAnalyzer object.

    Args:
        MultiAnalyzer (blueetl.analysis.MultiAnalyzer): A MultiAnalyzer object containing the simulation data.
        filtered_dataframes (dict): A dictionary containing dataframes that have been filtered by neuron class and time window.
        sigma (float, optional): The sigma value to use for smoothing. Defaults to 1.
        vivo_key (str, optional): The key for the in vivo data in the analysis config. Defaults to "vivo_reyesp".
        offset (bool, optional): The key for the offset in the analysis config, or False if no offset is used. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe containing the latency distance for each simulation, including the mean and standard deviation for each neuron class.
    """
    config = MultiAnalyzer.analyzers["spikes"].analysis_config.dict()

    invivo = pd.read_feather(config["custom"][vivo_key])
    if offset:
        offset = config["custom"][offset]

    post_analysis_window = config["custom"]["evoked_window_for_custom_post_analysis"]
    bounds = config["extraction"]["windows"][post_analysis_window]["bounds"]

    invivo = prepare_invivo(
        invivo,
        threshold_multiple=4,
        t_stim=0,
        sigma=sigma,
        offset=offset,
        bounds=bounds,
    )

    vivo_latency = pd.DataFrame(
        invivo.groupby("neuron_class").apply(
            lambda df: df.time.values[extract_latency_index(df.psth_smooth.values)]
        ),
        columns=["latency"],
    ).reset_index()

    silico_vivo_latency_distance = call_by_simulation(
        MultiAnalyzer.analyzers["spikes"].repo.simulations.df,
        filtered_dataframes,
        partial(
            latency_distance_2sim,
            vivo_latency,
        ),
    )
    return pd.concat(silico_vivo_latency_distance).reset_index(drop=True)


def decay_distance_2sim(
    baseline: pd.DataFrame, simulation_row: pd.Series, filtered_dataframes: dict
) -> pd.DataFrame:
    """
    Calculate the decay distance between a baseline and a simulation for a given set of filtered dataframes.

    Args:
        baseline (pd.DataFrame): A dataframe containing the baseline decay values for each neuron class.
        simulation_row (pd.Series): A series containing information about the simulation being analyzed.
        filtered_dataframes (dict): A dictionary containing dataframes that have been filtered by neuron class and time window.

    Returns:
        pd.DataFrame: A dataframe containing the decay distance for each simulation, including the mean and standard deviation for each neuron class.
    """
    decay = pd.merge(
        filtered_dataframes["decay"].reset_index()[
            ["neuron_class", "window", "decay", "ratio"]
        ],
        baseline,
        on=["neuron_class", "ratio"],
        suffixes=["_silico", "_vivo"],
    )
    decay["decay_diff"] = decay.apply(
        lambda row: row.decay_silico - row.decay_vivo, axis=1
    )

    return _add_feature_summary_row(
        decay, simulation_row.simulation_id, decay.iloc[0].window
    )


def silico_vivo_decay_distance(
    MultiAnalyzer: MultiAnalyzer,
    filtered_dataframes: Dict[str, pd.DataFrame],
    sigma: int = 1,
    ratio: Union[float, List[float]] = 0.75,
    vivo_key: str = "vivo_reyesp",
    offset: Union[bool, str] = False,
) -> pd.DataFrame:
    """
    Calculate the decay distance for each simulation in a blueetl.analysis.MultiAnalyzer object.

    Args:
    MultiAnalyzer (blueetl.analysis.MultiAnalyzer): The MultiAnalyzer object containing the simulations to be analyzed.
    filtered_dataframes(Dict[str, pd.DataFrame]): A dictionary containing the filtered dataframes for each feature.
    sigma (int, optional):The standard deviation for the Gaussian kernel used for smoothing, by default 1.
    ratio (Union[float, List[float]], optional): The ratio(s) at which to calculate the decay distance, by default 0.75.
    vivo_key (str, optional): The key for the invivo data file in the MultiAnalyzer analysis configuration, by default "vivo_reyesp".
    offset (Union[bool, str], optional): Whether or not to use an offset value for the data, by default False.

    Returns:
        pd.DataFrame: A dataframe containing the decay distance for each simulation.
    """
    if not isinstance(ratio, list):
        ratio = [ratio]

    config = MultiAnalyzer.analyzers["spikes"].analysis_config.dict()
    invivo = pd.read_feather(config["custom"][vivo_key])
    if offset:
        offset = config["custom"][offset]

    post_analysis_window = config["custom"]["evoked_window_for_custom_post_analysis"]
    bounds = config["extraction"]["windows"][post_analysis_window]["bounds"]
    invivo = prepare_invivo(
        invivo,
        threshold_multiple=4,
        t_stim=0,
        sigma=sigma,
        offset=offset,
        bounds=bounds,
    )

    vivo_decay = []
    for r in ratio:
        ratiodf = pd.DataFrame(
            invivo.groupby("neuron_class").apply(
                lambda df: extract_decay(
                    df.psth_smooth.values,
                    df.time.values,
                    ratio=r,
                )
            ),
            columns=["decay"],
        )
        ratiodf["ratio"] = r

        vivo_decay.append(ratiodf)
    vivo_decay = pd.concat(vivo_decay)
    vivo_decay = vivo_decay.reset_index()

    silico_vivo_decay_distance = call_by_simulation(
        MultiAnalyzer.analyzers["spikes"].repo.simulations.df,
        filtered_dataframes,
        partial(decay_distance_2sim, vivo_decay),
    )
    return pd.concat(silico_vivo_decay_distance).reset_index(drop=True)


def plot_psth_comparison_by_sim(
    baseline: pd.DataFrame,
    analysis_config: Dict[str, Any],
    subclass: str,
    vivo_key: str,
    simulation_row: pd.Series,
    filtered_dataframes: Dict[str, pd.DataFrame],
) -> None:
    """Plot peristimulus time histograms for each simulation instance in SimulationCampaign.

    Args:
        baseline (pd.DataFrame): DataFrame containing the baseline sim/vivo to compare with.
        analysis_config (dict): Analysis configuration information (plot setup).
        subclass (str): Name of the neuron subcategory, either "synapse_type" or "creline".
        vivo_key (str): The key for the in vivo data in the analysis configuration.
        simulation_row (pd.Series): Simulation-specific information.
        filtered_dataframes (dict): Dictionary of pd.DataFrames with pre-extracted histogram features.
    """
    filtered_dataframes["histogram_baseline"] = (
        filtered_dataframes["histogram_baseline"]
        .etl.q(neuron_class=neuron_classes)
        .reset_index()
    )

    filtered_dataframes["histogram_baseline"].loc[:, "layer"] = filtered_dataframes[
        "histogram_baseline"
    ].apply(lambda row: neuron_layers[row.neuron_class], axis=1)

    filtered_dataframes["histogram_baseline"].loc[
        :, "synapse_type"
    ] = filtered_dataframes["histogram_baseline"].apply(
        lambda row: neuron_synapse_type[row.neuron_class], axis=1
    )

    plot_2sim_PSTH_synapse_type(
        filtered_dataframes["histogram_baseline"],
        baseline,
        filtered_dataframes["decay"],
        analysis_config["custom"]["plot_output"].joinpath(
            f"{vivo_key}_{subclass}_psth_{simulation_row.simulation_id.values[0]}"
        ),
        simulation_row=simulation_row,
        vivo_key=vivo_key,
    )


def silico_vivo_psth_comparison(
    MultiAnalyzer: MultiAnalyzer,
    filtered_dataframes: Dict[str, pd.DataFrame],
    sigma: float = 1,
    vivo_key: str = "vivo_reyesp",
    offset: bool = False,
) -> None:
    """Compare SimulationCampaign analysis histograms to a baseline activity (in-vivo): visualization of peristimulus time histograms for each simulation.

    Args:
        MultiAnalyzer (blueetl.analysis.MultiAnalyzer): analyzer object
        filtered_dataframes (Dict[str, pd.DataFrame]): dictionary of pd.DataFrames with pre-extracted
            features to be used in the analysis
        sigma (float, optional): standard deviation of gaussian kernel to smooth firing rates. Defaults to 1.
        vivo_key (str, optional): name of the vivo data in analysis_config of Analyzer. Defaults to "vivo_reyesp".
        offset (bool, optional): boolean value to indicate if offset in simulation analysis should be considered.
            Defaults to False.
    """
    config = MultiAnalyzer.analyzers["spikes"].analysis_config.dict()

    invivo = pd.read_feather(config["custom"][vivo_key])
    if offset:
        offset = config["custom"][offset]

    post_analysis_window = config["custom"]["evoked_window_for_custom_post_analysis"]
    bounds = config["extraction"]["windows"][post_analysis_window]["bounds"]

    invivo = prepare_invivo(
        invivo,
        threshold_multiple=4,
        t_stim=0,
        sigma=sigma,
        offset=offset,
        bounds=bounds,
    ).etl.q(neuron_class=neuron_classes)

    call_by_simulation(
        MultiAnalyzer.analyzers["spikes"].repo.simulations.df,
        filtered_dataframes,
        partial(
            plot_psth_comparison_by_sim,
            baseline=invivo,
            analysis_config=config,
            subclass="synapse_type",
            vivo_key=vivo_key,
        ),
    )


def plot_psth_comparison_by_sim_creline(
    baseline: pd.DataFrame,
    analysis_config: Dict[str, Any],
    subclass: str,
    vivo_key: str,
    simulation_row: pd.Series,
    filtered_dataframes: Dict[str, pd.DataFrame],
) -> None:
    """Plot peristimulus time histograms for each simulation instance in SimulationCampaign. Split by creline

    Args:
        baseline (pd.DataFrame): DataFrame containing the baseline sim/vivo to compare with.
        analysis_config (dict): Analysis configuration information (plot setup).
        subclass (str): Name of the neuron subcategory, either "synapse_type" or "creline".
        vivo_key (str): The key for the in vivo data in the analysis configuration.
        simulation_row (pd.Series): Simulation-specific information.
        filtered_dataframes (dict): Dictionary of pd.DataFrames with pre-extracted histogram features
    """

    filtered_dataframes["histogram_baseline"] = (
        filtered_dataframes["histogram_baseline"]
        .etl.q(neuron_class=creline_classes)
        .reset_index()
    )

    filtered_dataframes["histogram_baseline"].loc[:, "layer"] = filtered_dataframes[
        "histogram_baseline"
    ].apply(lambda row: creline_layers[row.neuron_class], axis=1)

    filtered_dataframes["histogram_baseline"].loc[:, "creline"] = filtered_dataframes[
        "histogram_baseline"
    ].apply(lambda row: creline_mapping[row.neuron_class], axis=1)

    plot_2sim_PSTH_creline(
        filtered_dataframes["histogram_baseline"],
        baseline,
        filtered_dataframes["decay"],
        analysis_config["custom"]["plot_output"].joinpath(
            f"{vivo_key}_{subclass}_psth_{simulation_row.simulation_id.values[0]}"
        ),
        simulation_row=simulation_row,
        vivo_key=vivo_key,
    )


def silico_vivo_psth_comparison_creline(
    MultiAnalyzer: MultiAnalyzer,
    filtered_dataframes: Dict[str, pd.DataFrame],
    sigma: float = 1,
    vivo_key: str = "vivo_svoboda",
    offset: bool = False,
) -> None:
    """Compare SimulationCampaign analysis histograms to a baseline activity (in-vivo): visualization of peristimulus time histograms for each simulation.
    Performed for each creline.

    Args:
        MultiAnalyzer (blueetl.analysis.MultiAnalyzer): analyzer object
        filtered_dataframes (Dict[str, pd.DataFrame]): dictionary of pd.DataFrames with pre-extracted
            features to be used in the analysis
        sigma (float, optional): standard deviation of gaussian kernel to smooth firing rates. Defaults to 1.
        vivo_key (str, optional): name of the vivo data in analysis_config of Analyzer. Defaults to "vivo_reyesp".
        offset (bool, optional): boolean value to indicate if offset in simulation analysis should be considered.
            Defaults to False.
    """
    config = MultiAnalyzer.analyzers["spikes"].analysis_config.dict()

    invivo = pd.read_feather(config["custom"][vivo_key])
    if offset:
        offset = config["custom"][offset]

    post_analysis_window = config["custom"]["evoked_window_for_custom_post_analysis"]
    bounds = config["extraction"]["windows"][post_analysis_window]["bounds"]

    invivo = prepare_invivo(
        invivo,
        threshold_multiple=4,
        t_stim=0,
        sigma=sigma,
        offset=offset,
        bounds=bounds,
    ).etl.q(neuron_class=neuron_classes)

    call_by_simulation(
        MultiAnalyzer.analyzers["spikes"].repo.simulations.df,
        filtered_dataframes,
        partial(
            plot_psth_comparison_by_sim_creline,
            baseline=invivo,
            analysis_config=config,
            subclass="creline",
            vivo_key=vivo_key,
        ),
    )


def silico_vivo_ratio(
    MultiAnalyzer: MultiAnalyzer,
    filtered_dataframes: dict,
    sigma: int = 0,
    vivo_key: str = "vivo_reyesp",
    offset: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate the ratio of the mean firing rate of the simulation to the mean firing rate of each simulation
    instance in SimulationCampaign, compared against the in-vivo measurements.

    Args:
        MultiAnalyzer (blueetl.analysis.MultiAnalyzer): MultiAnalyzer object.
        filtered_dataframes (dict): Dictionary of pd.DataFrames with pre-extracted features.
        sigma (int, optional): Standard deviation of Gaussian filter applied to in-vivo data. Defaults to 0.
        vivo_key (str, optional): Name of the in-vivo data in analysis_config of MultiAnalyzer. Defaults to "vivo_reyesp".
        offset (str, optional): Name of the offset in analysis_config of MultiAnalyzer. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with mean_psth, max_psth, ratio, neuron_class and type (silico/vivo) information.
    """
    config = MultiAnalyzer.analyzers["ratios"].analysis_config.dict()

    invivo = pd.read_feather(config["custom"][vivo_key])
    if offset:
        offset = config["custom"][offset]

    post_analysis_window = config["custom"]["evoked_window_for_custom_post_analysis"]
    bounds = config["extraction"]["windows"][post_analysis_window]["bounds"]

    invivo = prepare_invivo(
        invivo,
        threshold_multiple=4,
        t_stim=0,
        sigma=sigma,
        offset=offset,
        bounds=bounds,
        normalize=False,
    )
    invivo["psth"] = invivo["psth_base"]
    stats = pd.merge(
        filtered_dataframes["mean_psth"].reset_index(),
        filtered_dataframes["max_psth"].reset_index(),
        on=["simulation_id", "circuit_id", "neuron_class", "window"],
    )
    stats["ratio"] = stats["max_psth"] / stats["mean_psth"]
    stats["type"] = "silico"

    vivo_max = (
        pd.DataFrame(
            invivo.groupby("neuron_class").apply(lambda df: get_max_psth(df)),
            columns=["max_psth"],
        )
        .reset_index()
        .drop(columns="level_1")
    )
    vivo_mean = (
        pd.DataFrame(
            invivo.groupby("neuron_class").apply(lambda df: get_mean_psth(df)),
            columns=["mean_psth"],
        )
        .reset_index()
        .drop(columns="level_1")
    )
    vivo_stats = pd.merge(vivo_max, vivo_mean, on="neuron_class")
    vivo_stats["ratio"] = vivo_stats["max_psth"] / vivo_stats["mean_psth"]
    vivo_stats["type"] = "vivo"

    return pd.concat([stats, vivo_stats])
