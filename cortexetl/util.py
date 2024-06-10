import numpy as np

import cv2
def images_from_filenames(list_of_filenames):
	img_array = []
	size=(0,0)	
	for filename in list_of_filenames:
		# filename_str = str(filename)
		# print(filename_str)
		img = cv2.imread(filename)
		if (img is not None):
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)
	return img_array, size

import os
def video_from_image_files(list_of_filenames, output_filename, delete_images=False):

	if (list_of_filenames != []):

		img_array, size = images_from_filenames(list_of_filenames)

		if (img_array != []):

			out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 1, size)
			for i in range(len(img_array)):
				frame = cv2.resize(img_array[i], size)
				out.write(frame)
			out.release()

		if delete_images:
			for f in list_of_filenames: os.remove(f)


def one_option_true(a, options):
	for opt in options:
		if (opt in list(a.analysis_config.custom.keys())):
			if (a.analysis_config.custom[opt] == True):
				return True
	return False


def hist_elements(hist_df):

    bin_indices = np.asarray(hist_df.index.unique(level='bin'))
    hist_array = np.asarray(hist_df['hist']).flatten()

    return bin_indices, hist_array

def print_useful_dfs(a):
	# print("\nnumber of missing/incomplete simulations: ", len(a.repo.missing_simulations()))

	print("\nsimulations\n", a.repo.simulations.df.columns.values.tolist(), "\n", a.repo.simulations.df, "\n")
	print("\nneuron_classes\n", a.repo.neuron_classes.df)
	print("\nneurons\n", a.repo.neurons.df)
	# print("\ntrial_steps\n", a.repo.trial_steps.df)
	print("\nwindows\n", a.repo.windows.df)    
	print("\nspikes\n", a.repo.spikes.df)

	print("\nby_gid\n", a.features.by_gid.df)
	print("\nby_gid_and_trial\n", a.features.by_gid_and_trial.df)
	print("\nby_neuron_class\n", a.features.by_neuron_class.df)
	print("\nby_neuron_class_and_trial\n", a.features.by_neuron_class_and_trial.df)
	print("\nhistograms\n", a.features.histograms.df)


from scipy.optimize import curve_fit
def fit_exponential(ca_stat1, ca_stat2):

	popt, pcov = curve_fit(
		lambda t, a, b, c: a * np.exp(b * t) + c,
		ca_stat1, ca_stat2, p0=(1.0, 0.5, ca_stat2.min() - 1), 
		maxfev=20000
	)

	model_preds = popt[0] * np.exp(popt[1] * ca_stat1) + popt[2]
	error = np.linalg.norm(model_preds - ca_stat2)

	return popt, error

import math
def calculate_suggested_unconnected_firing_rate(target_connected_fr, a, b, c):

	y = target_connected_fr
	log_domain = max((y - c) / a, 1.0)
	# print(y, c, a, log_domain)
	suggested_unconnected_fr = math.log(log_domain) / b

	return suggested_unconnected_fr


from math import log10, floor
round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))


def flatten(lol):
	return [x for xs in lol for x in xs]

def set_xy_labels_and_title(ax, xlabel, ylabel, title=''):
    ax.set_xlabel(xlabel, labelpad=-3)
    ax.set_ylabel(ylabel, labelpad=-4)
    ax.set_title(title)
    
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


