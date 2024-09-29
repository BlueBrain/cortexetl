# from bluepy import Simulation
# sim = Simulation('/gpfs/bbp.cscs.ch/project/proj83/home/isbister/sonata_test/sontata_test_0/simulation_config.json')
# print(sim)


# import h5py
# filename = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/sonata_test/sim_data/sonata_test_0/3a06e400-6920-4277-af36-5e53248fe192/0/reporting/spikes.h5"
# h5 = h5py.File(filename,'r')


# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(h5['spikes']['S1nonbarrel_neurons']['timestamps'], h5['spikes']['S1nonbarrel_neurons']['node_ids'])
# plt.savefig('test_conn_blocks_removed.png')



import matplotlib.pyplot as plt
import bluepysnap
simulation_path = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/sonata_test/sim_data/sonata_test_0/86525ee5-9123-4b08-9e51-7b356a5c7002/0/simulation_config.json"
simulation = bluepysnap.Simulation(simulation_path)
spikes = simulation.spikes

plt.figure()
spikes.filter().raster()
plt.savefig('test_conn_blocks_removed.png')