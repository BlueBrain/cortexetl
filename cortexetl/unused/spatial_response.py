import matplotlib.pyplot as plt
from matplotlib import cm

def spatial_response(a):

	for ind, simulation_row in a.repo.simulations.df.iterrows():

		if (ind == len(a.repo.simulations.df) - 1):

			cells_df = simulation_row['circuit'].cells.get({'$target': 'hex0'})

			fig = plt.figure()
			ax = plt.axes(projection='3d')
			window_features_df = a.features.by_gid.df.etl.q(simulation_id=simulation_row['simulation_id'], window='evoked_SOZ_25ms').reset_index()
			window_features_df = window_features_df[window_features_df['mean_firing_rates_per_second'] > 5.0]
			ax.scatter(cells_df.loc[window_features_df.gid, "x"], cells_df.loc[window_features_df.gid, "y"], cells_df.loc[window_features_df.gid, "z"], c=window_features_df['mean_firing_rates_per_second']) 
			plt.savefig("3D_Plot.png")
			plt.close()

			max_fr = np.max(a.features.by_gid.df.etl.q(simulation_id=simulation_row['simulation_id'])['mean_firing_rates_per_second'])

			fig, axes = plt.subplots(5, 2, figsize=(4, 12))
			axes[0][0].axis('off')

			for neuron_class in constants.LAYER_EI_NEURON_CLASSES:
				layers = constants.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[neuron_class]['layers']
				synapse_class = constants.NEURON_CLASS_LAYERS_AND_SYNAPSE_CLASSES[neuron_class]['synapse_class']

				neuron_group_cells_df = cells_df.query("layer == @layers")
				neuron_group_cells_df = neuron_group_cells_df[neuron_group_cells_df["synapse_class"] == synapse_class]

				window_and_neuron_class_df = a.features.by_gid.df.etl.q(simulation_id=simulation_row['simulation_id'], neuron_class=neuron_class, window='evoked_SOZ_25ms').reset_index()

				# max_fr = np.max(window_and_neuron_class_df['mean_firing_rates_per_second'])

				cmap = cm.cool

				thresholded_features = window_and_neuron_class_df[window_and_neuron_class_df['mean_firing_rates_per_second'] > 5.0]
				
				syn_classes = ["EXC", "INH"]
				lays = [1, 2, 4, 5, 6]
				e_i_ind = syn_classes.index(synapse_class)
				layer_ind = lays.index(layers[0])
				ax = axes[layer_ind][e_i_ind]
				ax.set_title(neuron_class)
				ax.axes.xaxis.set_visible(False)
				ax.axes.yaxis.set_visible(False)

				# plt.figure()
				# ax.scatter(neuron_group_cells_df["x"], neuron_group_cells_df["y"], c=window_and_neuron_class_df['mean_firing_rates_per_second'], cmap=cmap, vmin=0.0, vmax=max_fr, s=5, alpha=1.0, zorder=3)
				# ax.scatter(neuron_group_cells_df["x"], neuron_group_cells_df["y"], c='grey', s=5, alpha=1.0, zorder=3)
				ax.scatter(neuron_group_cells_df.loc[thresholded_features.gid, "x"], neuron_group_cells_df.loc[thresholded_features.gid, "y"], c=thresholded_features['mean_firing_rates_per_second'], cmap=cmap, vmin=0.0, vmax=max_fr, s=5, alpha=1.0, zorder=4)
				# plt.scatter(neuron_group_cells_df["x"], neuron_group_cells_df["y"], c=window_and_neuron_class_df.etl.q(gid=neuron_group_cells_df.index)['mean_firing_rates_per_second'], cmap=cmap, vmin=0.0, vmax=max_fr, s=5, alpha=1.0)
			plt.savefig("TOP.png", bbox_inches='tight')
			plt.close()