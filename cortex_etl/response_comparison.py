import cortex_etl as c_etl

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import blueetl as etl
from scipy.interpolate import interp1d
import seaborn as sns
from matplotlib import cm
import sys
import matplotlib.patches as mpatches



def compare_metrics_of_two_layerwise_groups(df, layers, group_1, group_2, stat, prefix, mask_key=''):

	if (mask_key != ''):
		mask_dict = {mask_key:False}
		df = df.etl.q(mask_dict)

	cmap = cm.get_cmap('rocket', 5)
	colors = cmap(np.linspace(0, 1, 5))

	layer_colours = {
				"L23": colors[0],
				"L4": colors[1],
				"L5": colors[2],
				"L6": colors[3]}

	group_1_df = df.etl.q(neuron_class=[layer + "_" + group_1 for layer in layers])
	group_2_df = df.etl.q(neuron_class=[layer + "_" + group_2 for layer in layers])

	group_1_df.loc[:, 'layer_colour'] = group_1_df.apply(lambda row: layer_colours[row['layer']], axis = 1)

	layer_df = pd.merge(group_1_df, group_2_df, on=['simulation_id', "layer"])

	layer_df[stat + "_silico_x"] = layer_df[stat + "_silico_x"] + np.random.normal(0.0, 0.1, len(layer_df[stat + "_silico_x"]))
	layer_df[stat + "_silico_y"] = layer_df[stat + "_silico_y"] + np.random.normal(0.0, 0.1, len(layer_df[stat + "_silico_y"]))



	plt.figure()
	plt.scatter(layer_df[stat + "_silico_x"], layer_df[stat + "_silico_y"], c=np.asarray(layer_df['layer_colour']).tolist(), s=0.2)

	# PLOT IN VIVO REFERENCES
	for layer in layers:
		single_layer_df = layer_df.etl.q(layer=layer)
		single_layer_df = single_layer_df.iloc[0]
		plt.scatter(single_layer_df[stat + "_vivo" + '_x'], single_layer_df[stat + "_vivo" + '_y'], c=np.asarray(single_layer_df['layer_colour']).tolist(), label=single_layer_df['layer'])


	plt.gca().legend()
	plt.gca().set_xlim([0.0, 20.0])
	plt.gca().set_ylim([0.0, 20.0])
	plt.gca().set_xlabel(group_1)
	plt.gca().set_ylabel(group_2)
	plt.gca().plot([0.0, 20.0], [0.0, 20.0], 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
	plt.savefig(prefix + '_' + group_1 + '_v_' + group_2 + '_' + stat + '_' + mask_key + '.pdf')
	plt.close()

def compare_metrics_to_in_vivo_for_neuron_classes(df, neuron_classes, prefix, mask_key=''):

	if (mask_key != ''):
		mask_dict = {mask_key:False}
		df = df.etl.q(mask_dict)


	cmap = cm.get_cmap('rocket', 9)
	colors = cmap(np.linspace(0, 1, 9))

	neuron_class_colours = {
				"L23_EXC": colors[0],
				"L4_EXC": colors[1],
				"L5_EXC": colors[2],
				"L6_EXC": colors[3],
				"L23_INH": colors[4],
				"L4_INH": colors[5],
				"L5_INH": colors[6],
				"L6_INH": colors[7]}

	plt.figure()

	df = df.etl.q(neuron_class=neuron_classes)

	unique_latencies = list(df['latency_vivo'].unique())
	offset_dict = {}
	for unique_latency in unique_latencies:
		neuron_classes_with_latency = df.etl.q(latency_vivo=unique_latency)['neuron_class'].unique()
		for i, neuron_class in enumerate(neuron_classes_with_latency):
			offset_dict[neuron_class] = i
	df["plot_offset"] = df.apply(lambda row: offset_dict[row['neuron_class']], axis = 1)
	df["latency_vivo"] = df["latency_vivo"] + np.random.normal(0.0, 0.02, len(df["latency_vivo"])) + df["plot_offset"] * 0.03
	df["latency_silico"] = df["latency_silico"] + np.random.normal(0.0, 0.1, len(df["latency_silico"]))
	df.loc[:, 'layer_colour'] = df.apply(lambda row: neuron_class_colours[row['neuron_class']], axis = 1)



	# FOR LEGEND
	for neuron_class in neuron_classes:
		single_nc_df = df.etl.q(neuron_class=neuron_class)
		single_nc_df = single_nc_df.iloc[0]
		plt.scatter(single_nc_df["latency_vivo"], single_nc_df["latency_silico"], c=np.asarray(single_nc_df['layer_colour']).tolist(), label=single_nc_df['neuron_class'], s=0.2)
	plt.gca().legend()

	plt.scatter(df["latency_vivo"], df["latency_silico"], c=np.asarray(df['layer_colour']).tolist(), s=0.2)
	plt.gca().set_xlim([0.0, 20.0])
	plt.gca().set_ylim([0.0, 20.0])
	plt.gca().plot([0.0, 20.0], [0.0, 20.0], 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
	plt.savefig(prefix + "_VS_VIVO" + '_' + mask_key + ".pdf")
	plt.close()


def compare_metrics_to_in_vivo_for_neuron_classes_VIOLIN(df, neuron_classes, prefix, mask_key=''):

	if (mask_key != ''):
		mask_dict = {mask_key:False}
		df = df.etl.q(mask_dict)


	cmap = cm.get_cmap('rocket', 9)
	colors = cmap(np.linspace(0, 1, 9))

	neuron_class_colours = {
				"L23_EXC": colors[0],
				"L4_EXC": colors[1],
				"L5_EXC": colors[2],
				"L6_EXC": colors[3],
				"L23_INH": colors[4],
				"L4_INH": colors[5],
				"L5_INH": colors[6],
				"L6_INH": colors[7]}

	plt.figure()

	df = df.etl.q(neuron_class=neuron_classes)

	unique_latencies = list(df['latency_vivo'].unique())
	offset_dict = {}
	for unique_latency in unique_latencies:
		neuron_classes_with_latency = df.etl.q(latency_vivo=unique_latency)['neuron_class'].unique()
		for i, neuron_class in enumerate(neuron_classes_with_latency):
			offset_dict[neuron_class] = i
	df["plot_offset"] = df.apply(lambda row: offset_dict[row['neuron_class']], axis = 1)

	# df.loc[:, 'layer_colour'] = df.apply(lambda row: row['latency_vivo'], axis = 1)

	df["latency_vivo"] = df["latency_vivo"] + np.random.normal(0.0, 0.02, len(df["latency_vivo"])) + df["plot_offset"] * 0.03
	df["latency_silico"] = df["latency_silico"] + np.random.normal(0.0, 0.1, len(df["latency_silico"]))
	df.loc[:, 'layer_colour'] = df.apply(lambda row: neuron_class_colours[row['neuron_class']], axis = 1)



	# # FOR LEGEND
	# for neuron_class in neuron_classes:
	# 	single_nc_df = df.etl.q(neuron_class=neuron_class)
	# 	single_nc_df = single_nc_df.iloc[0]
	# 	plt.scatter(single_nc_df["latency_vivo"], single_nc_df["latency_silico"], c=np.asarray(single_nc_df['layer_colour']).tolist(), label=single_nc_df['neuron_class'], s=0.2)
	# plt.gca().legend()
	labels = []
	def add_label(violin, label):
	    color = violin["bodies"][0].get_facecolor().flatten()
	    labels.append((mpatches.Patch(color=color), label))


	for neuron_class in neuron_classes:
		nc_df = df.etl.q(neuron_class=neuron_class)
		add_label(plt.violinplot(nc_df['latency_silico'], [nc_df.iloc[0]['latency_vivo'] + nc_df.iloc[0]['plot_offset']]), neuron_class)
	plt.legend(*zip(*labels), loc=2)

	# plt.scatter(df["latency_vivo"], df["latency_silico"], c=np.asarray(df['layer_colour']).tolist(), s=0.2)
	plt.gca().set_xlim([0.0, 20.0])
	plt.gca().set_ylim([0.0, 20.0])
	plt.gca().plot([0.0, 20.0], [0.0, 20.0], 'k--', alpha=0.75, zorder=0, lw=0.5, dashes=(5, 5))
	plt.savefig(prefix + "_VIOLIN_" + mask_key + ".pdf")
	plt.close()

