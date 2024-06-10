import pandas as pd
import matplotlib as mpl
import cortexetl as c_etl
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pickle
from scipy import sparse as sp
import matplotlib.ticker as ticker


def plot_neuron_class_info_scatter(info_for_ncs, x_stat, y_stat, figspath_root, show_stats=True, ax_to_use=None, show_ei_fits=False, zero_lower_ylim=True):
    
    if (ax_to_use==None):
        plt.figure(figsize=(1.5, 1.5))
        ax = plt.gca()
    else:
        ax = ax_to_use
    
    for _, info_for_nc in info_for_ncs.iterrows():
        ax.scatter(info_for_nc[x_stat], info_for_nc[y_stat], c=info_for_nc['color'], marker=info_for_nc['marker'])
        
    

    if show_stats:
        lr = linregress(info_for_ncs[x_stat], info_for_ncs[y_stat])
        ax.set_title('LR: (p: ' + str(c_etl.round_to_n(lr.pvalue, 2)) + ', r: ' + str(c_etl.round_to_n(lr.rvalue, 2)) + ')')
        ax.plot(info_for_ncs[x_stat], lr.slope * info_for_ncs[x_stat] + lr.intercept, '-k', lw=0.5)
        
    if show_ei_fits:
        exc_info_for_ncs = info_for_ncs.etl.q(neuron_class=['L23_EXC', 'L4_EXC', 'L5_EXC', 'L6_EXC'])
        inh_info_for_ncs = info_for_ncs.etl.q(neuron_class=['L1_INH', 'L23_INH', 'L4_INH', 'L5_INH', 'L6_INH'])
        
        exc_lr = linregress(exc_info_for_ncs[x_stat], exc_info_for_ncs[y_stat])
        inh_lr = linregress(inh_info_for_ncs[x_stat], inh_info_for_ncs[y_stat])
        
        ax.plot(exc_info_for_ncs[x_stat], exc_lr.slope * exc_info_for_ncs[x_stat] + exc_lr.intercept, '-r', lw=0.5)
        ax.plot(inh_info_for_ncs[x_stat], inh_lr.slope * inh_info_for_ncs[x_stat] + inh_lr.intercept, '-b', lw=0.5)
        
    mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
    ax.set_xlabel(c_etl.label_for_key(x_stat), labelpad=-3)
    ax.set_ylabel(c_etl.label_for_key(y_stat), labelpad=-8)
    ax.tick_params(axis='both', pad=2)


    ax.set_xlim([0.0, ax.get_xticks()[-1]])
    if zero_lower_ylim:
        ax.set_ylim([0.0, ax.get_yticks()[-1]])
    
    c_etl.remove_intermediate_labels(ax.xaxis.get_major_ticks())
    c_etl.remove_intermediate_labels(ax.yaxis.get_major_ticks())
    
    if (ax_to_use==None):
        plt.savefig(str(figspath_root) + '/' + x_stat + '_VS_' + y_stat + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
    
    if ax_to_use:
        return ax


def missing_synapses_analysis(a):
    
    
    import seaborn as sns
    sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3,
            "axes.titlesize": 7, "axes.spines.right": False, "axes.spines.top": False})
    
    print("Compare depol parameters to missing synapses")
    missing_E_synapses_by_neuron_class_df = pd.read_parquet('/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/blueetl_ji_analyses/data/missing_E_synapses_by_neuron_class.parquet')
    input_conductance_by_neuron_class_df = pd.read_parquet('/gpfs/bbp.cscs.ch/project/proj147/home/isbister/blueetl_ji_1/blueetl_ji_analyses/data/input_conductance_by_neuron_class.parquet')
    mean_depol_by_nc = a.custom['by_neuron_class'].etl.q(neuron_class=c_etl.LAYER_EI_NEURON_CLASSES, window=a.custom['by_neuron_class'].window.unique()[0]).loc[:,['depol_mean', 'neuron_class']].groupby(a.custom['by_neuron_class'].neuron_class.astype(object)).mean().reset_index()  
    
    info_for_nc = pd.merge(pd.merge(missing_E_synapses_by_neuron_class_df, input_conductance_by_neuron_class_df), mean_depol_by_nc)
    info_for_nc['true_mean_conductance'] = info_for_nc['resting_conductance'] * info_for_nc['depol_mean'] / 100.0
    info_for_nc.loc[:, 'color'] = info_for_nc.apply(lambda row: c_etl.LAYER_EI_NEURON_CLASS_COLOURS[row['neuron_class']], axis = 1)
    info_for_nc.loc[:, 'marker'] = info_for_nc.apply(lambda row: c_etl.LAYER_EI_NEURON_CLASS_MARKERS[row['neuron_class']], axis = 1)
    
    plot_neuron_class_info_scatter(info_for_nc, 'missing_E_synapses', 'resting_conductance', a.figpaths.root)
    plot_neuron_class_info_scatter(info_for_nc, 'missing_E_synapses', 'depol_mean', a.figpaths.root)
    plot_neuron_class_info_scatter(info_for_nc, 'missing_E_synapses', 'true_mean_conductance', a.figpaths.root)
    plot_neuron_class_info_scatter(info_for_nc, 'resting_conductance', 'true_mean_conductance', a.figpaths.root)
    
#     lr = linregress(info_for_nc['resting_conductance'], info_for_nc['true_mean_conductance'])
#     info_for_nc.loc[:, 'resting_conductance_VS_true_mean_conductance_residuals'] = info_for_nc['true_mean_conductance'] - (info_for_nc['resting_conductance'] * lr.slope + lr.intercept)
#     plot_neuron_class_info_scatter(info_for_nc, 'resting_conductance', 'resting_conductance_VS_true_mean_conductance_residuals', a.figpaths.root, show_stats=False)
    
    lr = linregress(info_for_nc['missing_E_synapses'], info_for_nc['true_mean_conductance'])
    info_for_nc.loc[:, 'missing_E_synapses_VS_true_mean_conductance_residuals'] = info_for_nc['true_mean_conductance'] - (info_for_nc['missing_E_synapses'] * lr.slope + lr.intercept)
    plot_neuron_class_info_scatter(info_for_nc, 'missing_E_synapses', 'missing_E_synapses_VS_true_mean_conductance_residuals', a.figpaths.root, show_stats=False, zero_lower_ylim=False)
    
    root='/gpfs/bbp.cscs.ch/project/proj83/home/egas/SSCX_structure_vs_function/data/'
    
    participation=pd.read_pickle(f'{root}node_participation_full.pkl')
    k_in_deg=pd.read_pickle(f'{root}k_in_degree_full.pkl')
    nrn_info=pd.read_pickle(f'{root}hex0_nrn_info.pickle')
    mat=sp.load_npz(f'{root}hex0_local_mat.npz')
    nodes=nrn_info.query("layer==2 or layer == 3").index

    for df in [participation]:
        df['layer']=nrn_info['layer'].astype(str)
        df['synapse_class']=nrn_info['synapse_class'].astype(str)
        df['layer_grouped']=nrn_info['layer'].astype(str)
        df.loc[nodes, 'layer_grouped']='23'
    
    dim_counts = participation.groupby(['layer', 'synapse_class']).mean()
    dim_counts.iloc[1] = (dim_counts.iloc[1] + dim_counts.iloc[3]) / 2.0
    dim_counts.iloc[2] = (dim_counts.iloc[2] + dim_counts.iloc[4]) / 2.0
    dim_counts = dim_counts.reset_index().drop(index=[3,4], axis=1).reset_index()
    
    exc_lr_rs = []
    inh_lr_rs = []
    fig, axes = plt.subplots(1, 7, figsize=(12, 1.5))

    
    for dim in range(1, 7, 1):
        
        info_for_nc.loc[:, 'dim' + str(dim) + '_counts'] = dim_counts[dim]
        ax = axes[dim-1]
        plot_neuron_class_info_scatter(info_for_nc, 'dim' + str(dim) + '_counts', 'missing_E_synapses_VS_true_mean_conductance_residuals', a.figpaths.root, show_stats=False, ax_to_use=ax, show_ei_fits=True)
        
        if (dim > 1):
            ax.set_ylabel('')
            ax.get_yaxis().set_ticklabels([])
        ax.set_ylim([-0.00075, 0.0005])
        ax.set_box_aspect(1)
        
        exc_info_for_nc = info_for_nc.etl.q(neuron_class=['L23_EXC', 'L4_EXC', 'L5_EXC', 'L6_EXC'])
        inh_info_for_nc = info_for_nc.etl.q(neuron_class=['L1_INH', 'L23_INH', 'L4_INH', 'L5_INH', 'L6_INH'])
        
        exc_lr_r = linregress(exc_info_for_nc['missing_E_synapses_VS_true_mean_conductance_residuals'], exc_info_for_nc['dim' + str(dim) + '_counts']).rvalue
        inh_lr_r = linregress(inh_info_for_nc['missing_E_synapses_VS_true_mean_conductance_residuals'], inh_info_for_nc['dim' + str(dim) + '_counts']).rvalue
        
        exc_lr_rs.append(exc_lr_r)
        inh_lr_rs.append(inh_lr_r)
        
    ax = axes[-1]
    ax.plot(list(range(1, 7, 1)), inh_lr_rs, c='b')
    ax.plot(list(range(1, 7, 1)), exc_lr_rs, c='r')
    ax.set_ylim([-1.0, .2])
    ax.set_xlim([0.5, 6.5])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('R-Value')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_box_aspect(1)
    plt.savefig('SimplexDimsExplainingResidual.pdf', bbox_inches='tight')