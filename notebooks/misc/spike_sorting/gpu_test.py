import numpy as np

import torch
from torch import nn
from torch import optim

class DeepNetReLU(nn.Module):
    """ network with a single hidden layer h with a RELU

    Args:
    n_inputs (int): number of input units
    n_hidden (int): number of units in hidden layer

    Attributes:
    in_layer (nn.Linear): weights and biases of input layer
    out_layer (nn.Linear): weights and biases of output layer

    """

    def __init__(self, n_inputs, n_hidden):
        super().__init__()  # needed to invoke the properties of the parent class nn.Module
        self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
        self.mid_layer = nn.Linear(n_hidden, n_hidden) # JI
        self.mid2_layer = nn.Linear(n_hidden, n_hidden) # JI
        self.out_layer = nn.Linear(n_hidden, 2) # hidden units --> output

    def forward(self, r):
        """Decode stimulus orientation from neural responses

        Args:
          r (torch.Tensor): vector of neural responses to decode, must be of
            length n_inputs. Can also be a tensor of shape n_stimuli x n_inputs,
            containing n_stimuli vectors of neural responses

        Returns:
          torch.Tensor: network outputs for each input provided in r. If
            r is a vector, then y is a 1D tensor of length 1. If r is a 2D
            tensor then y is a 2D tensor of shape n_stimuli x 1.

        """
        h = self.in_layer(r)  # hidden representation
        rect_h = torch.relu(h)
        
        m = self.mid_layer(rect_h)
        rect_m = torch.relu(m)
        
        m2 = self.mid2_layer(rect_m)
        rect_m2 = torch.relu(m2)

        # m3 = self.mid_layer(rect_m2)
        # rect_m3 = torch.relu(m3)
        
        y = self.out_layer(rect_m2)
        return y




import torch
import torch.utils.data as data

class SpikeDataLoader(data.Dataset):

    def __init__(self, histograms_by_rotation_and_trial, ids, start_trial, final_trial):
        
        self.start_trial = start_trial
        self.final_trial = final_trial
        self.num_trials = final_trial - start_trial + 1
        self.uniq_rotations = list(histograms_by_rotation_and_trial.keys())

        examples_to_stack = []
        labels = []

        for rotation in self.uniq_rotations:

            tr = histograms_by_rotation_and_trial[rotation][start_trial:final_trial+1, ids, :] * 100.0

            examples_to_stack.append(tr.reshape((self.num_trials, -1)))
            labels.extend([[rotation] for i in range(self.num_trials)])

        data = np.vstack(tuple(examples_to_stack))


        
        self.data = torch.Tensor(data)
        self.labels = torch.deg2rad(torch.Tensor(labels))


        labels_real = torch.cos(self.labels)
        labels_imaginary = torch.sin(self.labels)

        self.labels = torch.cat((labels_real, labels_imaginary), dim=1)


    def __getitem__(self, index):
        
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]



import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Queue, Process
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import math
import matplotlib.pyplot as plt

import os

import sys
sys.path.append('../..')
import cortexetl as c_etl
def train_model(queue, model_package, model_package_i):

    num_samples = len(model_package.train_loader)

    num_epochs = 300
    num_iterations_per_epoch = 1
    batch_size = 256
    batches_per_iteration = math.floor(num_samples / batch_size)

    num_samples_per_iteration = batches_per_iteration * batch_size

    data = model_package.train_loader.data
    target = model_package.train_loader.labels
    data, target = data.to(model_package.device), target.to(model_package.device)

    test_data_cpu = model_package.test_loader.data
    test_target_cpu = model_package.test_loader.labels
    test_targets_min = np.min(test_target_cpu.detach().numpy())
    test_targets_max = np.max(test_target_cpu.detach().numpy())
    test_data_gpu, test_target_gpu = test_data_cpu.to(model_package.device), test_target_cpu.to(model_package.device)

    test_losses = []
    epoch_fig_paths = []

    return_target_angles = None
    return_test_angles = None

    for epoch in range(-1, num_epochs):
        if epoch >= 0:

            for iteration in range(num_iterations_per_epoch):

                sample_indices = np.random.choice(num_samples, num_samples_per_iteration, replace=False)

                for batch_i in range(batches_per_iteration):

                    batch_indices = sample_indices[batch_i*batch_size:batch_i*batch_size + batch_size]

                    output = model_package.model(data[batch_indices, :])

                    loss = model_package.loss_criterion(output, target[batch_indices, :])

                    model_package.optimizer.zero_grad()
                    loss.backward()
                    model_package.optimizer.step()

        test_output = model_package.model(test_data_gpu)
        test_loss = model_package.loss_criterion(test_output, test_target_gpu)
        test_losses.append(test_loss.item())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # fig.suptitle(log_str)        
        # axes[0].plot(train_losses, c='b')
        axes[0].plot(test_losses, c='g')
        axes[0].set_xlim([0, num_epochs + 1])
        # axes[0].set_ylim([0, np.max(train_losses + test_losses)])
        # axes[0].set_ylim([0, np.max(test_losses)])
        axes[0].set_xlabel('Epoch / 50')
        axes[0].set_ylabel('Loss')


        complex_tensor = test_target_cpu.detach()
        real_part = complex_tensor[:, 0]
        imag_part = complex_tensor[:, 1]
        target_angles = torch.atan2(imag_part, real_part).unsqueeze(1).numpy()

        complex_tensor = test_output.cpu().detach()
        real_part = complex_tensor[:, 0]
        imag_part = complex_tensor[:, 1]
        test_angles = torch.atan2(imag_part, real_part).unsqueeze(1).numpy()

        return_target_angles = target_angles
        return_test_angles = test_angles
        
        axes[1].scatter(target_angles, test_angles)
        axes[1].set_xlim([test_targets_min - 1, test_targets_max + 1])
        axes[1].set_xlim([test_targets_min - 1, test_targets_max + 1])
        axes[1].plot([test_targets_min, test_targets_max], [test_targets_min, test_targets_max], ls='--')
        axes[1].set_xlabel('True rotation')
        axes[1].set_ylabel('Predicted rotation')
            
        # axes[2].scatter(test_target_cpu.detach().numpy(), test_output.cpu().flatten().detach().numpy())
        # axes[2].set_xlim([test_targets_min - 1, test_targets_max + 1])
        # axes[2].set_xlim([test_targets_min - 1, test_targets_max + 1])
        # axes[2].plot([test_targets_min, test_targets_max], [test_targets_min, test_targets_max], ls='--')
        # axes[2].set_xlabel('True rotation')
        # axes[2].set_ylabel('Predicted rotation')
            
        epoch_fig_path = model_package.outdir + str(model_package_i) + '_epoch' + str(epoch) + '.png'
        epoch_fig_paths.append(epoch_fig_path)
        plt.savefig(epoch_fig_path)
        plt.close()

        c_etl.video_from_image_files(epoch_fig_paths, model_package.outdir + str(model_package_i) + '_simulation_spikes_training_video.mp4', delete_images=False)



        if (epoch == 0) | (epoch == num_epochs - 1):     
            print(f'loss: {loss.item():.3f}')

    # model.state_dict() # Send the model parameters

    return_dict = {"test_losses": test_losses,
                    "neuron_class_str": model_package.neuron_class_str,
                    "target_angles": return_target_angles,
                    "test_angles": return_test_angles}

    queue.put(return_dict)
    print("FINISHED")


class ModelPackage(object):

    def __init__(self, data_with_rotation_info, ids, neuron_class_str, device, num_layers, layer_width, outdir, learning_rate=1e-4):

        self.neuron_class_str = neuron_class_str
        self.ids = ids
        self.device = device
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.learning_rate = learning_rate
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        if len(ids):
            _, _, filtered_hist_indices = np.intersect1d(ids, data_with_rotation_info['histogram_gids'], return_indices=True)

        self.train_loader = SpikeDataLoader(data_with_rotation_info['histograms_by_rotation_and_trial'], filtered_hist_indices, 0, 44)
        self.test_loader = SpikeDataLoader(data_with_rotation_info['histograms_by_rotation_and_trial'], filtered_hist_indices, 45, 49)

        self.example_width = self.train_loader[0][0].shape[0]

        self.model = DeepNetReLU(self.example_width, self.layer_width)
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        # self.loss_criterion = nn.MSELoss()
        self.loss_criterion = CSS_LOSS()


# def pos_loss(output, target):
#     loss = torch.mean(1.0 - torch.cos(output - target))
#     return loss



class CSS_LOSS(torch.nn.modules.loss._Loss):

    def forward(self, input, target):
        return 1.0 - torch.mean(torch.nn.functional.cosine_similarity(input, target))
        


            
import  pickle
import pandas as pd
def main():

    # Look at:
    # https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/4

    print("Num GPUs: " + str(torch.cuda.device_count()))
    torch.multiprocessing.set_start_method("spawn")
    # Define two devices
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")
    devices = [device0, device1, device2, device3]

    model_packages = []

    
    sorted_unit_prob_for_each_ground_truth_gid = pd.read_csv('/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/for_James/unit_matching.csv').rename(columns={"Unnamed: 0": "true_gid"})
    postprocessed_ss_ids = [48, 51, 61, 106, 130, 132, 166, 181, 189, 236, 240, 248, 261, 274, 316, 321, 326, 382, 400, 406, 407, 408, 409, 417, 418, 419, 421, 422, 437, 439, 440, 442, 448, 449, 453, 454, 455, 456, 457, 458, 461, 462, 467, 468, 469, 479, 480, 482, 485, 486, 496, 497, 501, 504, 505, 508, 513, 531, 533, 536, 539, 542, 548, 549, 551, 558, 562, 563, 564, 567, 568, 569, 570, 574, 575, 576, 577, 578, 579, 580, 581, 582, 584, 595, 596, 598, 599, 606, 607, 611, 614, 619, 623, 636, 638, 639, 641, 642, 643, 644, 648, 649, 653, 657, 658, 659, 671, 672, 687, 688, 695, 707, 714, 716, 718, 719, 721, 725, 727, 732, 747, 758, 762, 771, 772, 794, 796, 800, 801, 811, 812, 821, 830, 831, 842, 846, 851, 853, 857, 864, 866, 868, 874, 877, 881, 882, 884, 889, 904, 905, 906, 907, 910, 911, 912, 913, 917, 929, 930, 931, 932, 941, 942, 943, 944, 945, 946, 950, 957, 961, 963, 965, 966, 967, 968, 969, 970, 977, 978, 979, 991, 998, 1011, 1021, 1023, 1024, 1030, 1034, 1051, 1060, 1061, 1065, 1066, 1067, 1073, 1077, 1078, 1110, 1117, 1127, 1137, 1139, 1141, 1144, 1151, 1154, 1155, 1157, 1165, 1178, 1179, 1182, 1184, 1186, 1199, 1200, 1201, 1204, 1206, 1208, 1209, 1213, 1223, 1224, 1241, 1246, 1249, 1256, 1257, 1259, 1270, 1271, 1280, 1282, 1290, 1293, 1295, 1296, 1298, 1299, 1301, 1303, 1308, 1309, 1310, 1316, 1320, 1322, 1328, 1329, 1342, 1346, 1347, 1348, 1356, 1358, 1361, 1363, 1368, 1370, 1376, 1377, 1378, 1380, 1381, 1391, 1394, 1395, 1403, 1407, 1413, 1418, 1419, 1420, 1426, 1428, 1431, 1433, 1434, 1436, 1439, 1442, 1444, 1448, 1453, 1459, 1479, 1480, 1485, 1498, 1500, 1504, 1512, 1515, 1528, 1532, 1533, 1535, 1536, 1538, 1539, 1543, 1544, 1546, 1552, 1561, 1566, 1570, 1577, 1579, 1580, 1597, 1601, 1605, 1608, 1609, 1615, 1620, 1632, 1637, 1644, 1646, 1652, 1654, 1659, 1660, 1663, 1664, 1668, 1671, 1678, 1679, 1681, 1682, 1683, 1685, 1690, 1691, 1722, 1723, 1725, 1737, 1738, 1740, 1751, 1762, 1763, 1764, 1765, 1768, 1771, 1776, 1777, 1780, 1791, 1796, 1801, 1817, 1820, 1831, 1846, 1854, 1855, 1856, 1858, 1863, 1868, 1874, 1875, 1890, 1891, 1897, 1905, 1913, 1915, 1947, 1954, 1958, 1978, 1992, 1997, 2001, 2003, 2012, 2013, 2015, 2026, 2028, 2049, 2053, 2072, 2073, 2080, 2081, 2096, 2100, 2115, 2116, 2122, 2123, 2148, 2154, 2156, 2160, 2161, 2170, 2179, 2203, 2208, 2237, 2253, 2256, 2293, 2300, 2310, 2342, 2374, 2384, 2386, 2401, 2411, 2414, 2426, 2442, 2445, 2473, 2479, 2510, 2571, 2585, 2604, 2616, 2622]
    postprocessed_ss_ids_str = ['true_gid'] + [str(gid) for gid in postprocessed_ss_ids]
    sorted_unit_prob_for_each_ground_truth_gid = sorted_unit_prob_for_each_ground_truth_gid.loc[:, postprocessed_ss_ids_str]
    ground_truth_gids = np.asarray(sorted_unit_prob_for_each_ground_truth_gid['true_gid'])

    print(len(ground_truth_gids))
    print(len(postprocessed_ss_ids))
    # import sys; sys.exit()

    file = open('/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/pickles/pickles_20-4-24/data_with_rotation_info_simulation_spikes.pickle', 'rb')
    gt_data_with_rotation_info = pickle.load(file)
    file.close()

    file = open('/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/pickles/pickles_20-4-24/data_with_rotation_info_spike_sorted.pickle', 'rb')
    sorted_data_with_rotation_info = pickle.load(file)
    file.close()
    
    outroot = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/comparison/"


    model_package_0 = ModelPackage(sorted_data_with_rotation_info, postprocessed_ss_ids, "spike sorted", devices[0], 1, 2000, outroot + "ss_nn/")
    model_packages.append(model_package_0)

    model_package_1 = ModelPackage(gt_data_with_rotation_info, ground_truth_gids, "ground_truth", devices[1], 1, 2000, outroot + "gt_nn/")
    model_packages.append(model_package_1)

    model_package_2 = ModelPackage(gt_data_with_rotation_info, np.random.choice(ground_truth_gids, size=len(postprocessed_ss_ids), replace=False), "ground_truth_nss", devices[2], 1, 2000, outroot + "gt_nss_nn/")
    model_packages.append(model_package_2)


    
    # Create a queue for communication between processes
    queue = Queue()

    # Create four worker processes
    processes = []
    for i, model_package in enumerate(model_packages):
        process = Process(target=train_model, args=(queue, model_package, i))
        processes.append(process)
        # if i == 1:
        #     break

    print("START TRAINING")
    # Start all worker processes
    for process in processes:
        process.start()
    print("RETURNED")

    # Wait for all worker processes to complete
    for process in processes:
        process.join()

    print("JOINED")

    # Process the gradients and update the models
    return_dicts = []
    for _ in range(len(processes)):
        return_dict = queue.get()
        return_dicts.append(return_dict)

    #     # model_state_dict, loss = queue.get()
    #     # model.load_state_dict(model_state_dict)

    plt.figure()
    for return_dict in return_dicts:
        plt.plot([i for i in range(len(return_dict['test_losses']))], return_dict['test_losses'])
    plt.gca().set_xlabel('Epoch / 50')
    plt.gca().set_ylabel('Loss')
    plt.savefig(outroot + 'test_losses.pdf')
    plt.close()

    with open('/gpfs/bbp.cscs.ch/project/proj83/home/isbister/spike_sorting_bias_output_data/pickles/pickles_20-4-24/gpu_training_return_dicts.pkl', 'wb') as f:
        pickle.dump(return_dicts, f)


    print("DONE")

if __name__ == "__main__":
    main()