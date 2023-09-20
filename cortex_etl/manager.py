import logging
import sys
import shutil
import numpy as np
import seaborn as sns

from blueetl.analysis import Analyzer
from blueetl.constants import *
from blueetl.utils import load_yaml

import cortex_etl as c_etl

from blueetl.analysis import run_from_file


def analysis_initial_processing(analysis_config_file, loglevel="INFO"):

	sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3,
            "axes.titlesize": 7, "axes.spines.right": False, "axes.spines.top": False})

	ma = run_from_file(analysis_config_file, loglevel=loglevel)
	ma = ma.apply_filter()

	for a in ma.analyzers.values():
		c_etl.create_figure_dirs(a)

	return ma


def apply_analyses(a, a_name=''):

	if c_etl.one_option_true(a, ['unconnected_frs_df', 'unconnected_frs_plot']):
		c_etl.unconnected_analysis(a)

	if c_etl.one_option_true(a, ['extract_fr_df', 'print_coupled_coords_from_mask', 'plot_multi_sim_analysis', 'create_raster_videos', 'spike_pair_analysis', 'compare_campaigns', 'compare_to_missing_synapses']):
		c_etl.post_analysis(a)

	if c_etl.one_option_true(a, ['plot_multi_sim_analysis']):
		c_etl.plot_multi_sim_analysis(a)

	if c_etl.one_option_true(a, ['plot_rasters', 'create_raster_videos']):
		c_etl.plot_rasters(a)

	if (c_etl.one_option_true(a, ['create_flatspace_videos'])):
		c_etl.flatspace_videos(a)

	if (c_etl.one_option_true(a, ['compare_to_missing_synapses'])):
		c_etl.missing_synapses_analysis(a)

	if (c_etl.one_option_true(a, ['multi_hex_analysis'])):
		c_etl.multi_hex_analysis(a)

	if c_etl.one_option_true(a, ['spike_pair_analysis']):
		c_etl.spike_pair_analysis(a)

	if (c_etl.one_option_true(a, ['evoked_analysis'])):
		c_etl.evoked_analysis(a)

	if (c_etl.one_option_true(a, ['at_plots'])):
		c_etl.at_plots(a)

	if (c_etl.one_option_true(a, ['extract_fr_df'])):
		c_etl.extract_fr_df(a)

	if c_etl.one_option_true(a, ['create_multi_sim_summary_pdfs']):
		c_etl.multi_sim_summary_pdfs(a)

	if c_etl.one_option_true(a, ['compare_campaigns']):
		c_etl.compare_campaigns(a, a_name)


