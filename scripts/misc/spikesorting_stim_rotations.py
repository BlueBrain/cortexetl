import numpy as np

stims_per_sim = 10
num_sims = 36

string = "\"rotations\": ["


for sim_i in range(num_sims):
	string += "\""

	for stim_i in range(stims_per_sim):

		rotation = float(sim_i * stims_per_sim + stim_i)
		string += str(rotation)

		if stim_i < stims_per_sim - 1:
			string  += ", "

	string +="\""

	if sim_i < num_sims - 1:
		string += ",\n"

string += "]"
print(string)