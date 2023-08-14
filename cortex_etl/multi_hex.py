import pandas as pd
import blueetl as etl
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import os
import cortex_etl as c_etl


def get_flatspace_centre_of_hex(a, hex_ind):

	gids = a.repo.neurons.df.etl.q(circuit_id=0, neuron_class=["ALL_EXC_" + str(hex_ind), "ALL_INH_" + str(hex_ind)])['gid']
	locations = a.repo.simulations.df.iloc[0]['circuit'].cells.get(gids, ["x", "y", "z"])
	flat_locations = c_etl.flatten_locations(locations, "/gpfs/bbp.cscs.ch/project/proj83/home/bolanos/BB_Rat_SSCX_flatmap_v2.nrrd")
	x = np.mean(flat_locations[0])
	y = np.mean(flat_locations[1])
	return x,y

def get_xy_for_single_hex(hex_mean_flatspace_coords, hex_ind):

	hex_x = hex_mean_flatspace_coords.etl.q(hex=hex_ind).iloc[0]['x']
	hex_y = hex_mean_flatspace_coords.etl.q(hex=hex_ind).iloc[0]['y']

	return hex_x, hex_y


def plot_evoked_hists_by_hex(a, simulation_id, simulation_hists, hex_inds, hex_mean_flatspace_coords, stim_hex_ind):
	plt.figure(figsize=(10, 10))
	ax = plt.gca()

	for hex_ind in hex_inds:		
		hex_x, hex_y = get_xy_for_single_hex(hex_mean_flatspace_coords, hex_ind)

		for g, c in zip(["ALL_EXC_", "ALL_INH_"], ['r', 'b']):

			if (hex_ind == stim_hex_ind):
				c='g'

			neuron_class_hist = simulation_hists.etl.q({"neuron_class": g + str(hex_ind), "bin": {"le": 50, "ge": 0}})

			min_val = np.min(neuron_class_hist['hist'])
			min_arg = np.argmin(neuron_class_hist['hist'])

			max_val = np.max(neuron_class_hist['hist'])
			max_arg = np.argmax(neuron_class_hist['hist'])

			ax.plot(hex_x + neuron_class_hist['bin'] * 0.2, hex_y + ((neuron_class_hist['hist'] - min_val) / max_val)*3.0, c=c)

	plt.savefig(str(a.figpaths.multi_hex) + '/' + 'hex_psths_' + str(simulation_id) + '.pdf')
	plt.close()


def plot_evoked_hexes_by_hist(a, hex_mean_flatspace_coords):

	hists_for_plot = a.features.histograms.df.etl.q(window="evoked_SOZ_100ms", bin_size=1.0, smoothing_type='Gaussian', kernel_sd=1.0).reset_index()
	for simulation_id in hists_for_plot.simulation_id.unique():
		simulation_hists = hists_for_plot.etl.q(simulation_id=simulation_id)
		plot_evoked_hists_by_hex(a, simulation_id, simulation_hists, list(range(0, 77)), hex_mean_flatspace_coords, 0)


def calulate_flatspace_mean_coord_for_each_hex(a, hex_inds):
	
	xs = []; ys = []; 
	for hex_ind in hex_inds:
		x, y = get_flatspace_centre_of_hex(a, hex_ind)
		print(x, y)
		xs.append(x); ys.append(y); 

	df = pd.DataFrame(list(zip(hex_inds, xs, ys)), columns =['hex', 'x', 'y'])
	df.to_parquet(hex_mean_flatspace_coords_path)


def plot_hex_correlation_connectivity(a, hex_mean_flatspace_coords):

	# plt.figure()
	cmap = sns.cm.vlag
	cmap_center = 0.0
	fixed_lims = [-1.0, 1.0]

	rval_mat = np.zeros((78, 78))
	pval_mat = np.zeros((78, 78))

	spont_hists = a.features.histograms.df.etl.q(window="conn_spont", bin_size=1.0, smoothing_type='Gaussian', kernel_sd=1.0)

	for hex_ind_a in list(range(0, 77)):
		# plt.figure()
		hex_a_hist = spont_hists.etl.q(neuron_class="ALL_EXC_" + str(hex_ind_a)).reset_index()['hist']	
		# for hex_ind_b in list(range(hex_ind_a, 77)):
		for hex_ind_b in list(range(hex_ind_a + 1, 77)):
			hex_b_hist = spont_hists.etl.q(neuron_class="ALL_EXC_" + str(hex_ind_b)).reset_index()['hist']

			a_hex_x, a_hex_y = get_xy_for_single_hex(hex_mean_flatspace_coords, hex_ind_a)
			b_hex_x, b_hex_y = get_xy_for_single_hex(hex_mean_flatspace_coords, hex_ind_b)

			lr = linregress(hex_a_hist, hex_b_hist)
			rval = lr.rvalue
			pval = lr.pvalue
			if (pval < 0.05):
				print(hex_ind_a, hex_ind_b, rval)
				# plt.plot([a_hex_x, b_hex_x], [a_hex_y, b_hex_y], lw=3.0*abs(rval), c=cmap((rval/2.0) + 0.5))

			rval_mat[hex_ind_a, hex_ind_b] = rval
			pval_mat[hex_ind_a, hex_ind_b] = pval

		print(str(a.figpaths.root) + "/" + 'spatial_corr.pdf')

		plt.figure()
		plt.imshow(rval_mat)
		plt.savefig(str(a.figpaths.root) + "/spatial_corr_r_mat.pdf")
	plt.close()


def plot_corr_rval_by_hex(a, hex_mean_flatspace_coords, simulation_ids):

	for simulation_id in np.unique(a.repo.simulations.df.simulation_id):
	
		rvals = []
		xs = []
		ys = []

		spont_hists = a.features.histograms.df.etl.q(simulation_id=simulation_id, window="conn_spont", bin_size=1.0, smoothing_type='Gaussian', kernel_sd=1.0).reset_index()

		for hex_ind in list(range(0, 77)):
			try:
				spont_hist_EXC = spont_hists.etl.q(neuron_class="ALL_EXC_" + str(hex_ind))['hist']
				spont_hist_INH = spont_hists.etl.q(neuron_class="ALL_INH_" + str(hex_ind))['hist']

				if (not np.all(spont_hist_EXC == spont_hist_EXC.iloc[0])):
					rval = linregress(spont_hist_EXC, spont_hist_INH).rvalue
					rvals.append(rval)
					hex_x, hex_y = get_xy_for_single_hex(hex_mean_flatspace_coords, hex_ind)
					xs.append(hex_x)
					ys.append(hex_y)				
			except:
				print(str(hex_ind) + " didn't work")

		plt.figure()
		plt.scatter(xs, ys, c=rvals, s=400, marker=(6, 0, 0), cmap=sns.cm.vlag, vmin=-1, vmax=1) # 'H'
		plt.axis('off')
		plt.gca().set_aspect('equal', 'box')
		plt.colorbar(label='Correlation r value')

		hex0_x, hex0_y = get_xy_for_single_hex(hex_mean_flatspace_coords, 0)
		hex59_x, hex59_y = get_xy_for_single_hex(hex_mean_flatspace_coords, 59)
		plt.scatter([hex0_x], [hex0_y], c=[[0.0, 0.0, 0.0, 0.0]], s=400, marker=(6, 0, 0), linewidth=1, edgecolor='k')
		plt.scatter([hex59_x], [hex59_y], c=[[0.0, 0.0, 0.0, 0.0]], s=400, marker=(6, 0, 0), linewidth=1, edgecolor='k')

		plt.savefig(str(a.figpaths.multi_hex) + '/' + 'rvals_by_hex_' + str(simulation_id) + '.pdf')
		plt.close()




def multi_hex_analysis(a):

	print('multi_hex_analysis')

	hex_mean_flatspace_coords_path = 'scripts/hex_mean_flatspace_coords.parquet'
	if not os.path.exists(hex_mean_flatspace_coords_path):
		calulate_flatspace_mean_coord_for_each_hex(a, list(range(0, 77)), hex_mean_flatspace_coords_path)
	
	hex_mean_flatspace_coords = pd.read_parquet(hex_mean_flatspace_coords_path)
	
	if ('evoked_SOZ_100ms' in np.unique(a.repo.windows.df.window)): #JI_TBR
		plot_evoked_hexes_by_hist(a, hex_mean_flatspace_coords)

	if ('conn_spont' in np.unique(a.repo.windows.df.window)): #JI_TBR
		plot_corr_rval_by_hex(a, hex_mean_flatspace_coords, np.unique(a.repo.simulations.df.simulation_id))



