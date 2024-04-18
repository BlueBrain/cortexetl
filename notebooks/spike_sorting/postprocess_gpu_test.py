import math 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from matplotlib.ticker import MultipleLocator

sns.set(style="ticks", context="paper", font="Helvetica Neue",
        rc={"axes.labelsize": 7, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.3,
            "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.3,
            "axes.titlesize": 7, "axes.spines.right": False, "axes.spines.top": False})


def remove_intermediate_axis_labels(ax, y_or_x='x'):
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ticks = ax.get_xticks()
    if y_or_x == 'y':
        labels = [item.get_text() for item in ax.get_yticklabels()]
        ticks = ax.get_yticks() 
    
    num_labels = len(labels)
    for i in range(1, num_labels-1):
        labels[i] = ''
        
    if y_or_x == 'x': ax.set_xticks(ticks, labels)
    if y_or_x == 'y': ax.set_yticks(ticks, labels)



file = open('/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/pickles/pickles_20-4-24/gpu_training_return_dicts.pkl','rb')
return_dicts = pickle.load(file)

outroot = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/comparison/"


options = {
	'ground_truth': {
		'c': 'g',
		'linestyle': '-',
		'label': 'Silico (1836)',
		'zorder': -1,
		'alpha': 1.0,
		'test_vs_target': True
	},
	'ground_truth_nss': {
		'c': 'g',
		'linestyle': '--',
		'label': 'Silico (433)',
		'zorder': -1,
		'alpha': 1.5,
		'test_vs_target': False
	},
	'spike sorted': {
		'c': 'grey',
		'linestyle': '-',
		'label': 'Kilosort 3 (433)',
		'zorder': -1,
		'alpha': 0.5,
		'test_vs_target': True
	}


}


plt.figure(figsize=(2,2))
max_errors = []
return_dicts = [return_dicts[2], return_dicts[0], return_dicts[1]]
for return_dict in return_dicts:

	opts = options[return_dict['neuron_class_str']]

	errors = [math.degrees(math.acos(1.0 - l)) for l in return_dict['test_losses']]
	num_epochs = len(errors)
	max_errors.append(np.max(errors))
	plt.plot([i for i in range(len(return_dict['test_losses']))], errors, c=opts['c'], linestyle=opts['linestyle'], lw=1, label=opts['label'])

plt.gca().set_xlabel('Epoch / 50', labelpad=-3)
plt.gca().set_ylabel('Mean error (\N{DEGREE SIGN})', labelpad=-5)
plt.gca().set_xlim([0, num_epochs])
plt.gca().set_ylim([0, np.max(max_errors)])
remove_intermediate_axis_labels(plt.gca(), y_or_x='x')
remove_intermediate_axis_labels(plt.gca(), y_or_x='y')
plt.legend(frameon=False, bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig(outroot + 'mean_error_degrees.pdf')
plt.close()


plt.figure(figsize=(2,2))

for return_dict in return_dicts:

	opts = options[return_dict['neuron_class_str']]
	if opts['test_vs_target']:

		target_degrees = np.rad2deg(return_dict['target_angles']).flatten() + 180.0
		test_degrees = np.rad2deg(return_dict['test_angles']).flatten() + 180.0

		nvals = len(target_degrees)
		sample_indices = random.sample(range(nvals), int(math.ceil(nvals * 0.4)))

		plt.scatter(target_degrees[sample_indices], test_degrees[sample_indices], c=opts['c'], s=.2, zorder=opts['zorder'], alpha=opts['alpha'], label=opts['label'])

# plt.legend(frameon=False, bbox_to_anchor=(1.0, 1.0))


major_ticks = [0, 360]
plt.gca().set_xticks(major_ticks)
plt.gca().set_yticks(major_ticks)
minor_locator = MultipleLocator(60)
plt.gca().xaxis.set_minor_locator(minor_locator)
plt.gca().yaxis.set_minor_locator(minor_locator)
plt.plot([0, 360], [0, 360], ls='-', zorder=1, c='k', lw=0.5)
plt.gca().set_xlabel('True rotation', labelpad=-3)
plt.gca().set_ylabel('Predicted rotation', labelpad=-5)
# plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_xlim([0, 360])
plt.gca().set_ylim([0, 360])
plt.axis('scaled')
plt.tight_layout()
plt.savefig(outroot + 'test_target_vs_observed.pdf')
plt.close()


# remove_intermediate_axis_labels(plt.gca(), y_or_x='x')
# remove_intermediate_axis_labels(plt.gca(), y_or_x='y')





