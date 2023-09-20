import pandas as pd
import numpy as np
import blueetl
import bluepy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import sys
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.patheffects as mpe
import cortex_etl as c_etl


ng_and_sn_keys = {"L1_INH": "Stimulus_gExc_L1",
                        "L1_5HT3aR": "Stimulus_gExc_L1",
                      
                        "L23_EXC": "Stimulus_gExc_L23E",
                        "L23_INH": "Stimulus_gExc_L23I",
                        "L23_PV": "Stimulus_gExc_L23I",
                        "L23_SST": "Stimulus_gExc_L23I",
                        "L23_5HT3aR": "Stimulus_gExc_L23I",
                      
                        "L4_EXC": "Stimulus_gExc_L4E",
                        "L4_INH": "Stimulus_gExc_L4I",
                        "L4_PV": "Stimulus_gExc_L4I",
                        "L4_SST": "Stimulus_gExc_L4I",
                        "L4_5HT3aR": "Stimulus_gExc_L4I",
                      
                        "L5_EXC": "Stimulus_gExc_L5E",
                        "L5_INH": "Stimulus_gExc_L5I",
                        "L5_PV": "Stimulus_gExc_L5I",
                        "L5_SST": "Stimulus_gExc_L5I",
                        "L5_5HT3aR": "Stimulus_gExc_L5I",
                      
                        "L6_EXC": "Stimulus_gExc_L6E",
                        "L6_INH": "Stimulus_gExc_L6I",
                        "L6_PV": "Stimulus_gExc_L6I",
                        "L6_SST": "Stimulus_gExc_L6I",
                        "L6_5HT3aR": "Stimulus_gExc_L6I"
                     }

neuron_classes = ["L1_INH", "L1_5HT3aR",
                  "L23_EXC", "L23_INH", "L23_PV", "L23_SST", "L23_5HT3aR",
                  "L4_EXC", "L4_INH", "L4_PV", "L4_SST", "L4_5HT3aR", 
                  "L5_EXC", "L5_INH", "L5_PV", "L5_SST", "L5_5HT3aR", 
                  "L6_EXC", "L6_INH", "L6_PV", "L6_SST", "L6_5HT3aR"]
ca_key = "ca"; mean_key = "mean"; sd_key = "stdev"; seed_key = "seed"

neuron_classes = ["L1_INH", "L1_5HT3aR",
                          "L23_EXC", "L23_INH", "L23_PV", "L23_SST", "L23_5HT3aR",
                          "L4_EXC", "L4_INH", "L4_PV", "L4_SST", "L4_5HT3aR", 
                          "L5_EXC", "L5_INH", "L5_PV", "L5_SST", "L5_5HT3aR", 
                          "L6_EXC", "L6_INH", "L6_PV", "L6_SST", "L6_5HT3aR"]

neuron_class_groups = [["", "", "", "L1_5HT3aR"], 
                  ["L23_EXC", "L23_PV", "L23_SST", "L23_5HT3aR"],
                  ["L4_EXC", "L4_PV", "L4_SST", "L4_5HT3aR"], 
                  ["L5_EXC", "L5_PV", "L5_SST", "L5_5HT3aR"], 
                  ["L6_EXC", "L6_PV", "L6_SST", "L6_5HT3aR"]]

stim_type = "COND"

in_vivo_neuron_class_map = {
                        'L1_INH': 'L1I', 'L1_5HT3aR': 'L1I', 
    
                        'L23_EXC': 'L23E', 
                        'L23_INH': 'L23I', 'L23_PV': 'L23I', 'L23_SST': 'L23I', 'L23_5HT3aR': 'L23I', 
    
                        'L4_EXC': 'L4E', 
                        'L4_INH': 'L4I', 'L4_PV': 'L4I', 'L4_SST': 'L4I', 'L4_5HT3aR': 'L4I',
                        
                        'L5_EXC': 'L5E', 
                        'L5_INH': 'L5I', 'L5_PV': 'L5I', 'L5_SST': 'L5I', 'L5_5HT3aR': 'L5I',
                        
                        'L6_EXC': 'L6E', 
                        'L6_INH': 'L6I', 'L6_PV': 'L6I', 'L6_SST': 'L6I', 'L6_5HT3aR': 'L6I',
                        }

in_vivo_reference_frs = {"L1I": 1.500, "L23E": 0.070, "L23I": 0.961, "L4E": 0.619, "L4I": 1.184, "L5E": 1.252, "L5I": 2.357, "L6E": 0.248, "L6I": 1.500}


def unconnected_analysis(a):

    if c_etl.one_option_true(a, ['unconnected_frs_df', 'unconnected_frs_plot']):

        fr_df = a.features.by_neuron_class.df['mean_of_mean_firing_rates_per_second'].etl.q(window="unconn_2nd_half").droplevel(['window']).reset_index()
        fr_df = fr_df.rename(columns={"mean_of_mean_firing_rates_per_second": "data"})

        fr_df[ca_key] = np.nan; fr_df[mean_key] = np.nan; fr_df[sd_key] = np.nan; fr_df[seed_key] = np.nan

        for fr_index, fr_row in fr_df.iterrows():
            
            ng_key = fr_row["neuron_class"]    
            if (ng_key in neuron_classes):
                sn_key = ng_and_sn_keys[ng_key]

                simulation_row = a.repo.simulations.df.reset_index().etl.q(simulation_id=fr_row['simulation_id']).iloc[0]
                fr_df.loc[fr_index, mean_key] = float(simulation_row.simulation.instance.__dict__['config'][sn_key]['MeanPercent'])
                fr_df.loc[fr_index, sd_key] = float(simulation_row.simulation.instance.__dict__['config'][sn_key]['SDPercent'])
                fr_df.loc[fr_index, ca_key] = float(simulation_row.simulation.instance.__dict__['config']['Run_Default']['ExtracellularCalcium'])
                fr_df.loc[fr_index, seed_key] = int(simulation_row.simulation.instance.__dict__['config']['Run_Default']['BaseSeed'])
                

        fr_df = fr_df.etl.q(neuron_class=neuron_classes)

    if c_etl.one_option_true(a, ['unconnected_frs_df']):

        fr_df.to_parquet(a.figpaths.root / a.analysis_config.custom['fr_df_name'])

    if c_etl.one_option_true(a, ['unconnected_frs_plot']):

        data = fr_df

        mean_label = "$OU_{\mu}$"
        sd_label = "$OU_{\sigma}$"
        max_lim = 24.0

        grid_x, grid_y = np.mgrid[1:max_lim:500j, 1:max_lim:500j]

        cmap = cm.cool

        data_to_use = data
        neuron_classes_to_use = neuron_classes

        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(3*4, 3*5))

        max_fr = data_to_use['data'].max()
        for row_ind, neuron_class_group in enumerate(neuron_class_groups):
            
            for col_ind, neuron_class in enumerate(neuron_class_group):
                
                ax = axes[row_ind][col_ind]
                
                if (neuron_class==''):
                    ax.axis('off')
                else:
                    ng_data_for_plot = data_to_use.etl.q(ca=1.15, neuron_class=neuron_class)

                    ng_vivo_fr = in_vivo_reference_frs[in_vivo_neuron_class_map[neuron_class]]
                    lower_fr_lim = ng_vivo_fr *.01
                    upper_fr_lim = ng_vivo_fr*1.0
                    where_good = np.where((ng_data_for_plot["data"] < upper_fr_lim) & (ng_data_for_plot["data"] > lower_fr_lim))
                    where_out_of_range = np.where((ng_data_for_plot["data"] > upper_fr_lim) | (ng_data_for_plot["data"] < lower_fr_lim))

                    max_mean = ng_data_for_plot['mean'].max()
                    max_std = ng_data_for_plot['stdev'].max()

                    s=35

                    outside_in_vivo_ng_data =  ng_data_for_plot.iloc[where_out_of_range]
                    ax.scatter(outside_in_vivo_ng_data['mean'], outside_in_vivo_ng_data['stdev'], c=outside_in_vivo_ng_data['data'], cmap=cmap, s=s, linewidth=0.0, vmin=0.0, vmax=max_fr, alpha=0.5)

                    in_vivo_ng_data =  ng_data_for_plot.iloc[where_good]
                    ax.scatter(in_vivo_ng_data['mean'], in_vivo_ng_data['stdev'], c=in_vivo_ng_data['data'], cmap=cmap, s=s, linewidth=1, edgecolor='k', vmin=0.0, vmax=max_fr, alpha=0.5) 

                    ax.set_title(neuron_class, pad=-100)

                    shared_axis_processing(ax, stim_type, mean_label, sd_label, max_lim)

        norm = plt.Normalize(0.0, max_fr)
        sm =  ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        axins = inset_axes(axes[0][0],
                    width="100%",  
                    height="5%",
                    # loc='center right',
                    loc='center',
                    borderpad=-5
                   )

        cbar = fig.colorbar(sm, cax=axins, orientation="horizontal")
        cbar.set_label('FR (spikes/s)') # , rotation=270


        plt.savefig(a.figpaths.root / 'UnconnectedFRGrid_ALL.pdf', bbox_inches='tight')
plt.close()


def shared_axis_processing(ax, stim_type, mean_label, sd_label, max_lim):
    if (stim_type == "COND"):
        ax.set_xlim([0.0, max_lim + 1.0])
        ax.set_ylim([0.0, max_lim + 1.0])
        ax.xaxis.set_major_locator(MultipleLocator(max_lim + 1.0))
        ax.yaxis.set_major_locator(MultipleLocator(max_lim + 1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.labelpad = -9
    ax.yaxis.labelpad = -11
    ax.set_aspect('equal', 'box')
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_xlabel(mean_label); ax.set_ylabel(sd_label)
