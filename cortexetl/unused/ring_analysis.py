import cortexetl as c_etl

import matplotlib.pyplot as plt
import numpy as np

def ring_analysis(a):

	simulation_id = 9
	window = "conn_spont"

	simulation = a.repo.simulations.df.etl.q(simulation_id=simulation_id).iloc[0]
	window_0 = a.repo.windows.df.etl.q(simulation_id=simulation_id, window=window).iloc[0]

	
	plt.figure()
	ring_stats = a.features.by_neuron_class.df.etl.q(simulation_id=simulation_id, window=window, neuron_class=c_etl.ring_targets).reset_index()
	ring_stats.plot('neuron_class', 'mean_of_mean_firing_rates_per_second')
	plt.gca().set_ylim([0.0, np.max(ring_stats['mean_of_mean_firing_rates_per_second']) * 1.05])
	plt.savefig('TEST')





