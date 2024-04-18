import pickle

# with open('/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/pickles/pickles_20-4-24/gpu_training_return_dicts.pkl', 'r') as f:
# 	return_dicts = pickle.load(f)


file = open("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/pickles/pickles_20-4-24/gpu_training_return_dicts.pkl",'rb')
pickle.load(file)

print(return_dicts[0])