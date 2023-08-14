import blueetl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MultipleLocator
from .utils import synapse_type_colors, creline_colors


def smooth_histogram(histogram, sigma=1):
    """Smooths a histogram using a Gaussian filter.

    Args:
        histogram (ndarray): A one-dimensional numpy array representing the histogram to be smoothed.
        sigma (float): The standard deviation of the Gaussian kernel. Defaults to 1.

    Returns:
        ndarray: A one-dimensional numpy array representing the smoothed histogram.
    """
    return gaussian_filter(histogram, sigma=sigma)


def normalize_histogram(histogram):
    """Normalize a histogram by scaling all values by the maximum value of the histogram. Max value = 1.

    Args:
        histogram (ndarray): A one-dimensional numpy array representing the histogram to be normalized.

    Returns:
        ndarray: A one-dimensional numpy array representing the normalized histogram.
    """
    return histogram / np.max(histogram)


def histogram_distance(hist1, hist2):
    """Calculate the Euclidean distance between two histograms.

    Args:
        hist1 (ndarray): A one-dimensional numpy array representing the first histogram.
        hist2 (ndarray): A one-dimensional numpy array representing the second histogram.

    Returns:
        float: The Euclidean distance between the two histograms.
    """
    return distance.euclidean(hist1, hist2)


def onset_from_spikes(histogram, time, t_stim=0, threshold_multiple=4):
    """Calculates the onset of a "cortical" spiking response to an incoming stimulus, based on the binned spike counts histogram. The onset is defined as the point when the signal crosses a threshold, defined as the pre-stimulus window standard deviation multiplied by a factor threshold_multiple. `pre_stim_std` * `threshold_multiple`


    Args:
        histogram (np.array): binned spike counts
        time (np.array): Time array corresponding to the histogram
        t_stim (int, optional): time of stimulus. Defaults to 0.
        threshold_multiple (int, optional): Number of standard deviations to cross the threshold. Defaults to 4.

    Returns:
        onset_index: index of histogram array when onset occurs
    """
    stim_index = np.argwhere(time == t_stim)[0][0]
    pre_stimulus_histogram = histogram[:stim_index]
    post_stimulus_histogram = histogram[stim_index:]

    onset_index = np.argwhere(
        post_stimulus_histogram
        > pre_stimulus_histogram.mean()
        + threshold_multiple * pre_stimulus_histogram.std()
    ).flatten()

    if np.size(onset_index) > 0:
        stim_index += onset_index[0]

    return stim_index


def apply_on_neuron_class(data, function, key, new_key="new_feature", **kwargs):
    """Apply custom function on each neuron_class separately. Can be used with any
    function that should create a new column and. Generate a new pd.DataFrame with feature column.

    Args:
        data (pd.DataFrame): DataFrame with a histogram features to be process
        function (func): function/callable that takes an array as input and returns a scalar.
        key (string): Column name from which to extract the data to be processed.
        new_key (str, optional):Name for the new column created with the results of the function.
        Defaults to "new_feature".
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        pd.DataFrame: new feature from the frame
    """
    new_feature = (
        pd.DataFrame(
            data.groupby("neuron_class")
            .apply(lambda x: function(x[key].values, **kwargs))
            .explode(),
        )
        .reset_index()
        .rename(columns={0: new_key})
    )

    return new_feature


def find_onset_neuron_class(invivo, t_stim=0, threshold_multiple=2, key="psth_smooth"):
    """Finds the onset of the cortical spiking response to the incoming stimulus for each neuron class in the provided DataFrame.

    Args:
        invivo (pd.DataFrame): DataFrame with the histogram data and neuron classes.
        t_stim (int, optional): Time of simulation start. Defaults to 0.
        threshold_multiple (int, optional): Number of standard deviations to cross the threshold. Defaults to 2.
        key (str, optional): Name of the column in the DataFrame containing the histogram data. Defaults to "psth_smooth".

    Returns:
        pd.DataFrame: DataFrame containing the cortical onset histograms for each neuron class.
    """
    onsets = invivo.groupby("neuron_class").apply(
        lambda x: onset_from_spikes(
            x[key].values,
            x.time.values,
            t_stim=t_stim,
            threshold_multiple=threshold_multiple,
        )
    )
    invivo_onset = []
    for neuron_class, tempdf in invivo.groupby("neuron_class"):
        invivo_onset.append(tempdf.etl.q(time={"ge": onsets[neuron_class] - 50}))

    return pd.concat(invivo_onset)


def get_onset_PSTH(analysis_config, filtered_dataframes):
    """Works inplace for the given DataFame. Update the `filtered_dataframes` dictionary with a new key `"histogram_onset"`,
    containing the silico cortical onset histograms.

    Args:
        analysis_config (dict): A dictionary containing simulation configuration parameters.
        filtered_dataframes (dict): A dictionary of pandas DataFrame objects containing
            the filtered and processed simulation data.

    Returns:
        None

    Raises:
        KeyError: If any of the required keys (`"histogram"`, `"PSTH"`) are missing in the `analysis_config` dictionary.
    """
    filtered_dataframes["histogram"] = filtered_dataframes["histogram"].reset_index()[
        ["neuron_class", "psth", "time"]
    ]

    psth_smooth = apply_on_neuron_class(
        filtered_dataframes["histogram"],
        smooth_histogram,
        "psth",
        "psth_smooth",
        sigma=analysis_config["PSTH"]["sigma"],
    )
    psth_smooth["time"] = filtered_dataframes["histogram"].time.values

    filtered_dataframes["histogram"] = pd.merge(
        filtered_dataframes["histogram"], psth_smooth, on=["neuron_class", "time"]
    )

    filtered_dataframes["histogram_onset"] = find_onset_neuron_class(
        filtered_dataframes["histogram"],
        t_stim=analysis_config["PSTH"]["t_stim"],
        threshold_multiple=analysis_config["PSTH"]["threshold_multiple"],
    )


def plot_2sim_PSTH(
    silico,
    baseline,
    file_path,
    simulation_id,
    subclass="synapse_type",
    key="psth_smooth",
):
    """Plot a 2d-matrix-PSTH-comparison for each layer and neuron subclass.
    Size will be selected automatically, saves figure to the file_path directory.

    Args:
        silico (pd.DataFrame): DataFrame containing silico histograms
        baseline (pd.DataFrame): DataFrame containing baseline histograms
        file_path (pathlib.PosixPath): Path object representing the output directory path
        simulation_id (int): The id of the simulation
        subclass (str, optional): name of the neuron subcategory: "synapse_type"/"creline". Degault="synapse_type"
        key (str, optional): The name of the column containing the PSTH data to use. Default is 'psth_smooth'
    """
    subset_silico = silico.etl.q(neuron_class=baseline.neuron_class.values)

    fig, ax = plt.subplots(
        np.size(baseline.layer.unique()),
        np.size(baseline[subclass].unique()),
        figsize=(12, 20),
        sharex=True,
        sharey=True,
    )

    for n, layer in enumerate(baseline.layer.unique()):
        for m, subtype in enumerate(baseline[subclass].unique()):
            baseline_ = baseline[
                (baseline.layer == layer) & (baseline[subclass] == subtype)
            ]
            if np.size(baseline_.values) > 0:
                silico_ = subset_silico.etl.q(
                    neuron_class=baseline_.neuron_class.iloc[0]
                )
                ax[n, m].plot(baseline_[key].values, label="vivo")
                ax[n, m].plot(silico_[key].values, label="silico")
                ax[n, m].set_xlim(-5, 50)
                ax[n, m].set_title(baseline_[subclass].values[0])
        ax[n, 0].set_ylabel(f"L{layer}")

    ax[0, 0].legend()
    plt.suptitle("Simulation id:" + str(simulation_id))
    plt.savefig(file_path, bbox_inches="tight")


def plot_2sim_PSTH_synapse_type(
    silico,
    baseline,
    silico_decay,
    file_path,
    simulation_row,
    key="psth_norm",
    vivo_key="reyesp",
):
    """Plot a 2d-matrix-PSTH-comparison for each layer and neuron subclass for 2 simulations (or in-vivo and in-silico). Function plots PSTHs of different synapse types for simulated and experimental data, and also shows the decay of the synapse type in the simulation.

    Parameters:
        silico: DataFrame, simulated data.
        baseline: DataFrame, baseline data.
        silico_decay: DataFrame, simulated data of synapse type decay.
        file_path: str, the path to save the plot.
        simulation_row: DataFrame, the simulation parameters.
        key: str, optional, the column of the baseline data to plot. Default is 'psth_norm'.
        vivo_key: str, optional, the column of the experimental data to plot. Default is 'reyesp'.

    Returns:
        None.

    The function creates a subplot for each layer of the simulated data, with each subplot showing the PSTHs for different synapse types. For each synapse type, the function plots the PSTH of the simulated data and the baseline data (if available), as well as the decay of the synapse type in the simulation. The x-axis of each subplot is set to -10 to 50 ms and the y-axis is set to 0 to 1.1. The function also saves the plot as both a pdf and a png file at the given file_path.
    """
    layers = [1, 23, 4, 5, 6, 0]

    fig, ax = plt.subplots(
        np.size(layers),
        1,
        figsize=(4, 7),
        sharex=True,
        sharey=True,
    )

    for n, layer in enumerate(layers):
        for subtype in silico.etl.q(layer=layer).synapse_type.unique():
            silico_ = silico.etl.q(synapse_type=subtype, layer=layer)
            ax[n].plot(
                silico_["time"].values,
                silico_["psth"].values,
                color=synapse_type_colors(subtype),
                linewidth=1,
                label=subtype,
            )
            nc = silico_.neuron_class.iloc[0]
            ax[n].scatter(
                silico_decay.etl.q(neuron_class=nc, ratio=0.25).decay.values[0],
                0.25,
                color="mediumvioletred",
                s=15,
            )
            ax[n].scatter(
                silico_decay.etl.q(neuron_class=nc, ratio=0.5).decay.values[0],
                0.50,
                color="deeppink",
                s=15,
            )
            ax[n].scatter(
                silico_decay.etl.q(neuron_class=nc, ratio=0.75).decay.values[0],
                0.75,
                color="hotpink",
                s=15,
            )

            baseline_ = baseline.etl.q(layer=layer, synapse_type=subtype)

            if np.size(baseline_.values) > 0:
                ax[n].plot(
                    baseline_.time.values,
                    baseline_[key].values,
                    color=synapse_type_colors(subtype),
                    linewidth=1,
                    linestyle="--",
                )

        ax[n].set_ylabel(f"L{layer}", fontsize=12)

        if layer == 0:
            ax[n].set_ylabel(f"ALL", fontsize=12)

        ax[n].spines["top"].set_visible(False)
        ax[n].spines["right"].set_visible(False)
        ax[n].set_xlim(-10, 50)
        ax[n].set_ylim(0, 1.1)
        ax[n].vlines(
            0, -0, 1.1, color="black", linestyles="--", alpha=0.8, linewidth=0.8
        )

    ax[1].legend(loc=1)
    ax[n].set_xlabel("Time [ms]", fontsize=12)

    ax[3].xaxis.set_major_locator(MultipleLocator(10))
    ax[3].xaxis.set_major_formatter("{x:.0f}")
    ax[3].xaxis.set_minor_locator(MultipleLocator(5))

    fig.suptitle(
        f"{vivo_key} vs silico \n id: {str(simulation_row.simulation_id.values[0])},  ca: {simulation_row.ca.values[0]}, vpm_pct: {simulation_row.vpm_pct.values[0]}, ratio: {(simulation_row.depol_stdev_mean_ratio.values[0])}",
        fontsize=14,
    )
    fig.savefig(str(file_path) + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(str(file_path) + ".png", bbox_inches="tight", dpi=300)


def plot_2sim_PSTH_creline(
    silico,
    baseline,
    silico_decay,
    file_path,
    simulation_row,
    key="psth_norm",
    vivo_key="reyesp",
):
    """Plot a 2d-matrix-PSTH-comparison for each layer and neuron subclass for 2 simulations (or in-vivo and in-silico). Function plots PSTHs of different creline (PV, SST, VIP, EXC) types for simulated and experimental data, and also shows the decay of the synapse type in the simulation.

    Parameters:
        silico: DataFrame, simulated data.
        baseline: DataFrame, baseline data.
        silico_decay: DataFrame, simulated data of synapse type decay.
        file_path: str, the path to save the plot.
        simulation_row: DataFrame, the simulation parameters.
        key: str, optional, the column of the baseline data to plot. Default is "psth_norm".
        vivo_key: str, optional, the column of the experimental data to plot. Default is "reyesp".

    Returns:
        None

    Functionality:
        - Generates a plot comparing two sets of data (in vivo and in silico) for different neuron layers and subtypes
        - The plot is saved as both a PDF and PNG file
        - The plot includes multiple subplots, one for each layer, with each subplot showing the data for different neuron subtypes in that layer
        - Each subplot includes multiple scatter points and lines, representing the in silico data and baseline data, respectively
        - The x-axis of the plot represents time in milliseconds, and the y-axis represents the normalized PSTH value (ranging from 0 to 1.1)
        - The title of the plot includes information about the simulation, such as simulation ID, calcium concentration, VPM percentage, and depolarization standard deviation mean ratio.
    """
    layers = [23, 4, 5, 6, 0]

    fig, ax = plt.subplots(
        np.size(layers),
        1,
        figsize=(4, 7),
        sharex=True,
        sharey=True,
    )

    for n, layer in enumerate(layers):
        for subtype in silico.etl.q(layer=layer).creline.unique():
            silico_ = silico.etl.q(creline=subtype, layer=layer)
            ax[n].plot(
                silico_["time"].values,
                silico_["psth"].values,
                color=creline_colors(subtype),
                linewidth=1,
                label=subtype,
            )
            nc = silico_.neuron_class.iloc[0]
            ax[n].scatter(
                silico_decay.etl.q(neuron_class=nc, ratio=0.25).decay.values[0],
                0.25,
                color="mediumvioletred",
                s=15,
            )
            ax[n].scatter(
                silico_decay.etl.q(neuron_class=nc, ratio=0.5).decay.values[0],
                0.50,
                color="deeppink",
                s=15,
            )
            ax[n].scatter(
                silico_decay.etl.q(neuron_class=nc, ratio=0.75).decay.values[0],
                0.75,
                color="hotpink",
                s=15,
            )
            baseline_ = baseline.etl.q(layer=layer, creline=subtype)

            if np.size(baseline_.values) > 0:
                ax[n].plot(
                    baseline_.time.values,
                    baseline_[key].values,
                    color=creline_colors(subtype),
                    linewidth=1,
                    linestyle="--",
                )

        ax[n].set_ylabel(f"L{layer}", fontsize=12)

        if layer == 0:
            ax[n].set_ylabel(f"ALL", fontsize=12)

        ax[n].spines["top"].set_visible(False)
        ax[n].spines["right"].set_visible(False)
        ax[n].set_xlim(-10, 50)
        ax[n].set_ylim(0, 1.1)
        ax[n].vlines(
            0, -0, 1.1, color="black", linestyles="--", alpha=0.8, linewidth=0.8
        )

    ax[1].legend(loc=1)
    ax[n].set_xlabel("Time [ms]", fontsize=12)

    ax[3].xaxis.set_major_locator(MultipleLocator(10))
    ax[3].xaxis.set_major_formatter("{x:.0f}")
    ax[3].xaxis.set_minor_locator(MultipleLocator(5))

    fig.suptitle(
        f"{vivo_key} vs silico \n id: {str(simulation_row.simulation_id.values[0])},  ca: {simulation_row.ca.values[0]}, vpm_pct: {simulation_row.vpm_pct.values[0]}, ratio: {(simulation_row.depol_stdev_mean_ratio.values[0])}",
        fontsize=14,
    )
    fig.savefig(str(file_path) + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(str(file_path) + ".png", bbox_inches="tight", dpi=300)
