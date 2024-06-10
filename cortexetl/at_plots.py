import os
import logging
import blueetl
import argparse
import numpy as np
import pandas as pd
from glob import glob
from blueetl.utils import load_yaml
from blueetl.analysis import Analyzer
from barreletl.post_analysis import (
    silico_vivo_latency_distance,
    silico_vivo_decay_distance,
    silico_vivo_psth_comparison,
    silico_vivo_psth_comparison_creline,
)
from barreletl.utils import video_from_image_files

class ArgHolder:
    def __init__(self):
        """
        """


def at_plots(a):

    # Get prefix - sigma - bin_size
    # features_path = a.analysis_config.custom["output"].joinpath("features")
    # analysis_params = []
    # for setup in a.analysis_config.custom["analysis"]["features"]:
    #     prefix = ""
    #     if "prefix" in setup["params"]:
    #         prefix = setup["params"]["prefix"]
    #     params = setup["params"][f"{prefix}baseline_PSTH"]["params"]
    #     params["prefix"] = prefix

    #     analysis_params.append(params)
    # analysis_params = pd.DataFrame(analysis_params)

    args = ArgHolder()
    args.prefix = "s1_"

    print("Post analysis")
    simulations = a.repo.simulations.df
    frames = {
        "histogram_baseline": getattr(a.features, f"{args.prefix}baseline_PSTH").df.etl.q(
            window=a.analysis_config.custom["evoked_window_for_custom_post_analysis"]
        ),
        "latency": getattr(a.features, f"{args.prefix}latency").df.etl.q(
            window=a.analysis_config.custom["evoked_window_for_custom_post_analysis"]
        ),
        "decay": getattr(a.features, f"{args.prefix}decay").df.etl.q(
            window=a.analysis_config.custom["evoked_window_for_custom_post_analysis"]
        ),
    }
    # import pdb; pdb.set_trace()
    
    # # args = {"prefix": "s1"}
    # frames = {
    #     "histogram_baseline": a.features.baseline.etl.q(
    #         window=a.analysis_config.custom["evoked_window_for_custom_post_analysis"]
    #     ),
    #     "latency": a.features.latency.etl.q(
    #         window=a.analysis_config.custom["evoked_window_for_custom_post_analysis"]
    #     ),
    #     "decay": a.features.decay.etl.q(
    #         window=a.analysis_config.custom["evoked_window_for_custom_post_analysis"]
    #     ),
    # }


    # print(a.features.analysis_params)

    sigma = (
        a.features.analysis_params.etl.q(prefix=args.prefix).sigma.values[0]
        * a.features.analysis_params.etl.q(prefix=args.prefix).bin_size.values[0]
    )


    latency = silico_vivo_latency_distance(
        a, frames, sigma, vivo_key="vivo_reyesp", offset="offset_reyesp"
    )
    decay = silico_vivo_decay_distance(
        a, frames, sigma, ratio=0.25, vivo_key="vivo_reyesp", offset="offset_reyesp"
    )

    reyes = pd.merge(
        latency[
            [
                "neuron_class",
                "latency_diff",
                "latency_silico",
                "latency_vivo",
                "simulation_id",
            ]
        ],
        simulations,
        on="simulation_id",
    )
    reyes = pd.merge(
        reyes,
        decay[
            [
                "neuron_class",
                "decay_diff",
                "decay_silico",
                "decay_vivo",
                "simulation_id",
                "ratio",
            ]
        ],
        on=["simulation_id", "neuron_class"],
    )

    keys_to_save = [
        "neuron_class",
        "simulation_id",
        "ca",
        "desired_connected_proportion_of_invivo_frs",
        "depol_stdev_mean_ratio",
        "vpm_pct",
        "latency_silico",
        "latency_vivo",
        "latency_diff",
        "decay_silico",
        "ratio",
        "decay_vivo",
        "decay_diff",
        "simulation_path",
        "circuit_id",
    ]


    reyes[keys_to_save].to_parquet(
        a.analysis_config.custom["output"].joinpath(f"{args.prefix}vivo_reyes_comparison.parquet")
    )

    latency = silico_vivo_latency_distance(
        a, frames, sigma, vivo_key="vivo_svoboda", offset="offset_svoboda"
    )
    decay = silico_vivo_decay_distance(
        a,
        frames,
        sigma,
        ratio=[0.25, 0.5, 0.75],
        vivo_key="vivo_svoboda",
        offset="offset_svoboda",
    )

    svoboda = pd.merge(
        latency[
            [
                "neuron_class",
                "latency_diff",
                "latency_silico",
                "latency_vivo",
                "simulation_id",
            ]
        ],
        simulations,
        on="simulation_id",
    )
    svoboda = pd.merge(
        svoboda,
        decay[
            [
                "neuron_class",
                "decay_diff",
                "decay_silico",
                "decay_vivo",
                "simulation_id",
                "ratio",
            ]
        ],
        on=["simulation_id", "neuron_class"],
    )

    svoboda[keys_to_save].to_parquet(
        a.analysis_config.custom["output"].joinpath(
            f"{args.prefix}vivo_svoboda_comparison.parquet"
        )
    )


    print("Make plots")

    a.analysis_config.custom["plot_output"] = a.analysis_config.custom["output"].joinpath(
        f"{args.prefix}plots"
    )
    if not os.path.exists(a.analysis_config.custom["plot_output"]):
        os.makedirs(a.analysis_config.custom["plot_output"])

    silico_vivo_psth_comparison(
        a, frames, sigma, vivo_key="vivo_reyesp", offset="offset_reyesp"
    )
    silico_vivo_psth_comparison(
        a, frames, sigma, vivo_key="vivo_svoboda", offset="offset_svoboda"
    )

    file_list = list(
        np.sort(glob(str(a.analysis_config.custom["plot_output"]) + "/vivo_reyesp_synapse_type_*"))
    )
    output_file = str(a.analysis_config.custom["plot_output"]) + "/vid_synapse_type_reyes.mp4"
    video_from_image_files(file_list, output_file)

    file_list = list(
        np.sort(
            glob(str(a.analysis_config.custom["plot_output"]) + "/vivo_svoboda_synapse_type_*")
        )
    )
    output_file = str(a.analysis_config.custom["plot_output"]) + "/vid_synapse_type_svoboda.mp4"
    video_from_image_files(file_list, output_file)


    silico_vivo_psth_comparison_creline(
        a, frames, sigma, vivo_key="vivo_svoboda", offset="offset_svoboda"
    )
    file_list = list(
        np.sort(glob(str(a.analysis_config.custom["plot_output"]) + "/vivo_svoboda_creline_*"))
    )
    output_file = (
        str(a.analysis_config.custom["plot_output"]) + "/vid_synapse_type_svoboda_creline.mp4"
    )
    video_from_image_files(file_list, output_file)
