import numpy as np
import itertools
import pandas as pd

def create_pairwise_dfs(a):

	np.random.seed(42)
	sample_gids = np.random.choice(a.repo.neurons.df['gid'].unique(), 200)
	gid_pairs = np.asarray(list(itertools.combinations(sample_gids, 2)))
	df = pd.DataFrame(gid_pairs, columns=['gid_1', 'gid_2'])
	df['circuit_id'] = 0
	dff = a.features.by_gid_and_trial.df['first'].reset_index()
	dffx = dff.etl.q(gid=df['gid_1'].unique())
	dffy = dff.etl.q(gid=df['gid_2'].unique())
	neuron_pairs_df = (df
	.merge(dffx, left_on=['circuit_id', 'gid_1'], right_on=['circuit_id', 'gid'], how='left')
	.merge(dffy, left_on=['circuit_id', 'gid_2', 'simulation_id', 'window', 'trial'], right_on=['circuit_id', 'gid', 'simulation_id', 'window', 'trial'], how='left')
	.drop(['gid_1', 'gid_2'], axis=1)
	)

	neuron_pairs_df['diff'] = neuron_pairs_df['first_y'] - neuron_pairs_df['first_x']

	print(neuron_pairs_df)
	print(neuron_pairs_df.groupby(by=["circuit_id", "simulation_id", "window", "gid_x", "gid_y", "neuron_class_x", "neuron_class_y"])[['first_x','first_y']].corr().unstack().iloc[:,1])
	print(neuron_pairs_df.groupby(by=["circuit_id", "simulation_id", "window", "gid_x", "gid_y", "neuron_class_x", "neuron_class_y"]).mean().drop(['trial']))