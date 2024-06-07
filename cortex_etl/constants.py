# NEURON CLASS LISTS
LAYER_EI_NEURON_CLASSES = ['L1_INH', 'L23_EXC', 'L23_INH', 'L4_EXC', 'L4_INH', 'L5_EXC', 'L5_INH', 'L6_EXC', 'L6_INH']
LAYER_EI_NEURON_CLASSES_INH_FIRST = ['L1_INH', 'L23_INH', 'L4_INH', 'L5_INH', 'L6_INH', 'L23_EXC', 'L4_EXC', 'L5_EXC', 'L6_EXC']
LAYER_EI_RP_NEURON_CLASSES = ['L23_EXC', 'L23_INH', 'L4_EXC', 'L4_INH', 'L5_EXC', 'L5_INH']
LAYER_EI_SVO_NEURON_CLASSES = ['L23_EXC', 'L23_INH', 'L4_EXC', 'L4_INH', 'L5_EXC', 'L5_INH', "L6_EXC", "L6_INH"]
ALL_EI_NEURON_CLASSES = ['ALL_INH', 'ALL_EXC']
LAYER_E_AND_I_SUB_POP_NEURON_CLASSES = ["L1_5HT3aR",
                                        "L23_EXC",  "L23_PV", "L23_SST", "L23_5HT3aR", 
                                        "L4_EXC", "L4_PV", "L4_SST", "L4_5HT3aR", 
                                        "L5_EXC", "L5_PV", "L5_SST", "L5_5HT3aR", 
                                        "L6_EXC", "L6_PV", "L6_SST", "L6_5HT3aR"]

# NEURON CLASS GROUPINGS
LAYER_EI_NEURON_CLASS_GROUPINGS = [['L1_INH'], ['L23_EXC', 'L23_INH'], ['L4_EXC', 'L4_INH'], ['L5_EXC', 'L5_INH'], ['L6_EXC', 'L6_INH'], ['ALL_EXC', 'ALL_INH']]
LAYER_EI_NEURON_CLASS_GROUPINGS_WITH_ALL = [['L1_INH'], ['L23_EXC', 'L23_INH'], ['L4_EXC', 'L4_INH'], ['L5_EXC', 'L5_INH'], ['L6_EXC', 'L6_INH'], ['ALL_EXC', 'ALL_INH', 'ALL']]
NEURON_CLASS_NO_GROUPINGS = [['L1_INH'], ['L23_EXC'], ['L23_INH'], ['L4_EXC'], ['L4_INH'], ['L5_EXC'], ['L5_INH'], ['L6_EXC'], ['L6_INH']]
ALL_EI_NEURON_CLASS_GROUPINGS = [['ALL_INH', 'ALL_EXC']]
PYR_AND_IN_PSTH_NC_GROUPINGS_BY_LAYER = [["ALL", "L1_INH", "", ""], ["L23_EXC", "L23_INH", "L23_PV", "L23_SST"], ["L4_EXC", "L4_INH", "L4_PV", "L4_SST"], ["L5_EXC", "L5_INH", "L5_PV", "L5_SST"], ["L6_EXC", "L6_INH", "L6_PV", "L6_SST"]]

E_AND_I_SEPERATE_GROUPINGS = [['L6_INH', 'L5_INH', 'L4_INH', 'L23_INH', 'L1_INH'], ['L6_EXC', 'L5_EXC', 'L4_EXC', 'L23_EXC']]
E_AND_I_SUB_POP_SEPERATE_GROUPINGS = [['L6_PV', 'L5_PV', 'L4_PV', 'L23_PV'], ['L6_SST', 'L5_SST', 'L4_SST', 'L23_SST'], ['L6_5HT3aR', 'L5_5HT3aR', 'L4_5HT3aR', 'L23_5HT3aR', 'L1_5HT3aR'], ['L6_EXC', 'L5_EXC', 'L4_EXC', 'L23_EXC']]
E_AND_I_SUB_POP_BY_LAYER_GROUPINGS = [["", "L1_INH", "", "", "L1_5HT3aR"], ["L23_EXC", "L23_INH", "L23_PV", "L23_SST", "L23_5HT3aR"], ["L4_EXC", "L4_INH", "L4_PV", "L4_SST", "L4_5HT3aR"], ["L5_EXC", "L5_INH", "L5_PV", "L5_SST", "L5_5HT3aR"], ["L6_EXC", "L6_INH", "L6_PV", "L6_SST", "L6_5HT3aR"]]




blue_c = 'b'
blue_c = "#3271b8"

red_c = 'r'
red_c = "#e32b14"

RED = "#e32b14"
BLUE = "#3271b8"
GREEN = "#67b32e"
ORANGE = "#c9a021"
sst_c = 'lightskyblue'
sst_c = GREEN


NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES = {
	'ALL': {"layers": ["ALL"], "synapse_class": "ALL", "layer_string": "ALL", "color":'k', "marker":'.'},	

	'ALL_EXC': {"layers": ["ALL"], "synapse_class": "EXC", "layer_string": "ALL", "color":red_c, "marker":'.'},	
	'ALL_INH': {"layers": ["ALL"], "synapse_class": "INH", "layer_string": "ALL", "color":blue_c, "marker":'.'},	

	'L1_INH': {"layers": [1], "synapse_class": "INH", "layer_string": "L1", "color":blue_c, "marker":'o'},
    'L1_5HT3aR': {"layers": [1], "synapse_class": "5HT3aR", "layer_string": "L1", "color":"paleturquoise", "marker":'o'},

	'L23_EXC': {"layers": [2, 3], "synapse_class": "EXC", "layer_string": "L23", "color":red_c, "marker":'^'},
	'L23_INH': {"layers": [2, 3], "synapse_class": "INH", "layer_string": "L23", "color":blue_c, "marker":'^'},
	'L23_PV': {"layers": [2, 3], "synapse_class": "PV", "layer_string": "L23", "color":'midnightblue', "marker":'^'},
	'L23_SST': {"layers": [2, 3], "synapse_class": "SST", "layer_string": "L23", "color":sst_c, "marker":'^'},
	'L23_5HT3aR': {"layers": [2, 3], "synapse_class": "5HT3aR", "layer_string": "L23", "color":'paleturquoise', "marker":'^'},

	'L4_EXC': {"layers": [4], "synapse_class": "EXC", "layer_string": "L4", "color":red_c, "marker":'s'},
	'L4_INH': {"layers": [4], "synapse_class": "INH", "layer_string": "L4", "color":blue_c, "marker":'s'},
	'L4_PV': {"layers": [4], "synapse_class": "PV", "layer_string": "L4", "color":'midnightblue', "marker":'s'},
	'L4_SST': {"layers": [4], "synapse_class": "SST", "layer_string": "L4", "color":sst_c, "marker":'s'},
	'L4_5HT3aR': {"layers": [4], "synapse_class": "5HT3aR", "layer_string": "L4", "color":'paleturquoise', "marker":'s'},

	'L5_EXC': {"layers": [5], "synapse_class": "EXC", "layer_string": "L5", "color":red_c, "marker":'p'},
	'L5_INH': {"layers": [5], "synapse_class": "INH", "layer_string": "L5", "color":blue_c, "marker":'p'},
	'L5_PV': {"layers": [5], "synapse_class": "PV", "layer_string": "L5", "color":'midnightblue', "marker":'p'},
	'L5_SST': {"layers": [5], "synapse_class": "SST", "layer_string": "L5", "color":sst_c, "marker":'p'},
	'L5_5HT3aR': {"layers": [5], "synapse_class": "5HT3aR", "layer_string": "L5", "color":'paleturquoise', "marker":'p'},

	'L6_EXC': {"layers": [6], "synapse_class": "EXC", "layer_string": "L6", "color":red_c, "marker":'h'},
	'L6_INH': {"layers": [6], "synapse_class": "INH", "layer_string": "L6", "color":blue_c, "marker":'h'},
	'L6_PV': {"layers": [6], "synapse_class": "PV", "layer_string": "L6", "color":'midnightblue', "marker":'h'},
	'L6_SST': {"layers": [6], "synapse_class": "SST", "layer_string": "L6", "color":sst_c, "marker":'h'},
	'L6_5HT3aR': {"layers": [6], "synapse_class": "5HT3aR", "layer_string": "L6", "color":'paleturquoise', "marker":'h'}}

neuron_class_label_map = {
						"ALL": "All",
						"L1_INH": "L1 I", 
						'L23_EXC': 'L23 E', 
						'L23_INH': 'L23 I', 
						'L23_PV': 'L23 PV', 
						'L23_SST': 'L23 SST',
						'L4_EXC': 'L4 E', 
						'L4_INH': 'L4 I',
						'L4_PV': 'L4 PV', 
						'L4_SST': 'L4 SST',
						'L5_EXC': 'L5 E', 
						'L5_INH': 'L5 I', 
						'L5_PV': 'L5 PV', 
						'L5_SST': 'L5 SST',
						'L6_EXC': 'L6 E', 
						'L6_INH': 'L6 I', 
						'L6_PV': 'L6 PV', 
						'L6_SST': 'L6 SST',
						}
														
bluepy_neuron_class_map = {
						'L1_INH': 'L1I', 
						'L23_EXC': 'L23E', 
						'L23_INH': 'L23I', 
						'L4_EXC': 'L4E', 
						'L4_INH': 'L4I', 
						'L5_EXC': 'L5E', 
						'L5_INH': 'L5I', 
						'L6_EXC': 'L6E', 
						'L6_INH': 'L6I', 
                        "L1_5HT3aR": "L1_5HT3aR", 
                        "L23_PV": "L23_PV", 
                        "L23_SST": "L23_SST", 
                        "L23_5HT3aR": "L23_5HT3aR", 
                        "L4_PV": "L4_PV", 
                        "L4_SST": "L4_SST", 
                        "L4_5HT3aR": "L4_5HT3aR", 
                        "L5_PV": "L5_PV", 
                        "L5_SST": "L5_SST", 
                        "L5_5HT3aR": "L5_5HT3aR", 
                        "L6_PV": "L6_PV", 
                        "L6_SST": "L6_SST", 
                        "L6_5HT3aR": "L6_5HT3aR"
						}

                        


bluepy_neuron_class_map_2 = {
						'L1_INH': 'L1', 
						'L23_EXC': 'L23E', 
						'L23_INH': 'L23I', 
						'L4_EXC': 'L4E', 
						'L4_INH': 'L4I', 
						'L5_EXC': 'L5E', 
						'L5_INH': 'L5I', 
						'L6_EXC': 'L6E', 
						'L6_INH': 'L6I', 
                        "L1_5HT3aR": "L1_5HT3aR", 
                        "L23_PV": "L23_PV", 
                        "L23_SST": "L23_SST", 
                        "L23_5HT3aR": "L23_5HT3aR", 
                        "L4_PV": "L4_PV", 
                        "L4_SST": "L4_SST", 
                        "L4_5HT3aR": "L4_5HT3aR", 
                        "L5_PV": "L5_PV", 
                        "L5_SST": "L5_SST", 
                        "L5_5HT3aR": "L5_5HT3aR", 
                        "L6_PV": "L6_PV", 
                        "L6_SST": "L6_SST", 
                        "L6_5HT3aR": "L6_5HT3aR" 
						}

backup_ncs = {
    'L1_INH': 'L1_INH', 
    'L23_EXC': 'L23_EXC', 
    'L23_INH': 'L23_INH', 
    'L4_EXC': 'L4_EXC', 
    'L4_INH': 'L4_INH', 
    'L5_EXC': 'L5_EXC', 
    'L5_INH': 'L5_INH', 
    'L6_EXC': 'L6_EXC', 
    'L6_INH': 'L6_INH', 
    "L1_5HT3aR": "L1_INH", 
    "L23_PV": "L23_INH", 
    "L23_SST": "L23_INH", 
    "L23_5HT3aR": "L23_INH", 
    "L4_PV": "L4_INH", 
    "L4_SST": "L4_INH", 
    "L4_5HT3aR": "L4_INH", 
    "L5_PV": "L5_INH", 
    "L5_SST": "L5_INH", 
    "L5_5HT3aR": "L5_INH", 
    "L6_PV": "L6_INH", 
    "L6_SST": "L6_INH", 
    "L6_5HT3aR": "L6_INH" 
}

vivo_neuron_class_map = {
						'L1_INH': 'L1 INH', 
						'L23_EXC': 'L23 EXC', 
						'L23_INH': 'L23 INH', 
						'L4_EXC': 'L4 EXC', 
						'L4_INH': 'L4 INH', 
						'L5_EXC': 'L5 EXC', 
						'L5_INH': 'L5 INH', 
						'L6_EXC': 'L6 EXC', 
						'L6_INH': 'L6 INH', 
						}

silico_layer_strings = ['L1', 'L23', 'L4', 'L5', 'L6']



# awake_vivo_fr_dict = {'L23_EXC':0.237, 
# 						'L23_INH':0.386,
# 						'L4_EXC':0.445, 
# 						'L4_INH':0.962,
# 						'L5_EXC':1.355, 
# 						'L5_INH':1.346}


LAYER_MARKERS = {
	'L1': 'o',
	'L23': '^',
	'L4': 's',
	'L5': 'p',
	'L6': 'h'

}

LAYER_EI_NEURON_CLASS_COLOURS = {'L1_INH':'b', 
								'L23_EXC':'r', 
								'L23_INH':'b', 
								'L4_EXC':'r', 
								'L4_INH':'b', 
								'L5_EXC':'r', 
								'L5_INH':'b', 
								'L6_EXC':'r', 
								'L6_INH':'b'}

LAYER_EI_NEURON_CLASS_MARKERS = {'L1_INH':'o', 
								'L23_EXC':'^', 
								'L23_INH':'^', 
								'L4_EXC':'s', 
								'L4_INH':'s', 
								'L5_EXC':'p', 
								'L5_INH':'p', 
								'L6_EXC':'h', 
								'L6_INH':'h'}

ring_targets = ["ALL_RING_005", 
				"ALL_RING_010",
				"ALL_RING_015", 
				"ALL_RING_020",
				"ALL_RING_025", 
				"ALL_RING_030",
				"ALL_RING_035", 
				"ALL_RING_040",
				"ALL_RING_045", 
				"ALL_RING_050",
				"ALL_RING_055", 
				"ALL_RING_060",
				"ALL_RING_065", 
				"ALL_RING_070",
				"ALL_RING_075", 
				"ALL_RING_080",
				"ALL_RING_085", 
				"ALL_RING_090"]



parameter_constants = {

    # 'ca':
    # {
    #     'axis_label':'$[Ca^{2+}]_o$',
    #     'unit_string': 'mM',
    #     'unit_scale_up': 1.0
    # },

    'ca':
    {
        'axis_label':'$Ca^{2+}$',
        'unit_string': 'mM',
        'unit_scale_up': 1.0
    },

    'depol_stdev_mean_ratio':
    {
        'axis_label':'$R_{OU}$',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'fr_scale':
    {
        'axis_label':'$P_{FR}$',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'vpm_pct':
    {
        'axis_label':'$F_{P}$',
        'unit_string': '%',
        'unit_scale_up': 1.0
    },

    'none':
    {
        'axis_label':'',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'desired_connected_proportion_of_invivo_frs':
    {
    	'axis_label':'$P_{FR}$',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'freq':
    {
    	'axis_label':'Freq',
        'unit_string': 'Hz',
        'unit_scale_up': 1.0
    },

    'missing_E_synapses':
    {
        'axis_label':'Mean number of\nmissing synapses',
        'unit_string': '',
        'unit_scale_up': 1.0
    },
    
	'missing_E_synapses_VS_true_mean_conductance_residuals':
    {
        'axis_label':'Residual conductance \ninjection',
        'unit_string': '$\mu S$',
        'unit_scale_up': 1.0
    },
    
    'dim1_counts':
    {
        'axis_label':'Mean simplex count\n(Dim 1)',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'dim2_counts':
    {
        'axis_label':'Mean simplex count\n(Dim 2)',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'dim3_counts':
    {
        'axis_label':'Mean simplex count\n(Dim 3)',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'dim4_counts':
    {
        'axis_label':'Mean simplex count\n(Dim 4)',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'dim5_counts':
    {
        'axis_label':'Mean simplex count\n(Dim 5)',
        'unit_string': '',
        'unit_scale_up': 1.0
    },

    'dim6_counts':
    {
        'axis_label':'Mean simplex count\n(Dim 6)',
        'unit_string': '',
        'unit_scale_up': 1.0
    },


    'resting_conductance':
    {
        'axis_label':'Resting conductance',
        'unit_string': '',
        'unit_scale_up': 1.0
    },
    
    'depol_mean':
    {
        'axis_label':'$OU_{\mu}$',
        'unit_string': '%',
        'unit_scale_up': 1.0
    },
    
    'true_mean_conductance':
    {
        'axis_label':'Mean conductance\ninjection',
        'unit_string': '$\mu S$',
        'unit_scale_up': 1.0
    },
    
    'connection_vs_unconn_proportion':
    {
        'axis_label':'Conn MFR /\nUnconn MFR',
        'unit_string': '',
        'unit_scale_up': 1.0
    },
    
    'mean_of_mean_firing_rates_per_second':
    {
        'axis_label':'Conn MFR',
        'unit_string': 'Hz',
        'unit_scale_up': 1.0
    },

    'desired_unconnected_fr':
    {
        'axis_label':'Unconn MFR',
        'unit_string': 'Hz',
        'unit_scale_up': 1.0
    },

    'ei_corr_rval':
    {
        'axis_label':'R-value',
        'unit_string': '',
        'unit_scale_up': 1.0
    }



}