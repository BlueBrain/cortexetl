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
        self.out_layer = nn.Linear(n_hidden, 1) # hidden units --> output

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

    def __init__(self, histograms_by_nc, neuron_class, start_trial, final_trial):
        
        self.neuron_class = neuron_class
        self.start_trial = start_trial
        self.final_trial = final_trial
        self.num_trials = final_trial - start_trial + 1
        self.uniq_rotations = list(histograms_by_nc[neuron_class].keys())

        examples_to_stack = []
        labels = []

        for rotation in self.uniq_rotations:

            tr = histograms_by_nc[neuron_class][rotation][start_trial:final_trial+1]

            examples_to_stack.append(tr.reshape((self.num_trials, -1)))
            labels.extend([[rotation] for i in range(self.num_trials)])

        data = np.vstack(tuple(examples_to_stack))


        
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)


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

import sys
sys.path.append('../..')
import cortex_etl as c_etl
def train_model(queue, model_package, model_package_i):

    num_samples = len(model_package.train_loader)

    num_epochs = 100
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

    for epoch in range(num_epochs):

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




        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # fig.suptitle(log_str)        
        # axes[0].plot(train_losses, c='b')
        axes[0].plot(test_losses, c='g')
        axes[0].set_xlim([0, num_epochs])
        # axes[0].set_ylim([0, np.max(train_losses + test_losses)])
        axes[0].set_ylim([0, np.max(test_losses)])
        axes[0].set_xlabel('Epoch / 50')
        axes[0].set_ylabel('Loss')
        
        axes[1].scatter(test_target_cpu.detach().numpy(), test_output.cpu().flatten().detach().numpy())
        axes[1].set_xlim([test_targets_min - 1, test_targets_max + 1])
        axes[1].set_xlim([test_targets_min - 1, test_targets_max + 1])
        axes[1].plot([test_targets_min, test_targets_max], [test_targets_min, test_targets_max], ls='--')
        axes[1].set_xlabel('True rotation')
        axes[1].set_ylabel('Predicted rotation')
            
        epoch_fig_path = str(model_package_i) + '_epoch' + str(epoch) + '.png'
        epoch_fig_paths.append(epoch_fig_path)
        plt.savefig(epoch_fig_path)
        plt.close()

        c_etl.video_from_image_files(epoch_fig_paths, str(model_package_i) + '_simulation_spikes_training_video.mp4')



        if (epoch == 0) | (epoch == num_epochs - 1):     
            print(f'loss: {loss.item():.3f}')

    # model.state_dict() # Send the model parameters
    queue.put(test_losses)
    print("FINISHED")


class ModelPackage(object):

    def __init__(self, histograms_by_nc, neuron_class, device, num_layers, layer_width, learning_rate=1e-4):

        self.neuron_class = neuron_class
        self.device = device
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.learning_rate = learning_rate

        # self.model_package_str = 

        self.train_loader = SpikeDataLoader(histograms_by_nc, self.neuron_class, 0, 44)
        self.test_loader = SpikeDataLoader(histograms_by_nc, self.neuron_class, 45, 49)

        self.example_width = self.train_loader[0][0].shape[0]

        self.model = DeepNetReLU(self.example_width, self.layer_width)
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.loss_criterion = nn.MSELoss()
        


            

def main():

#     import pickle
#     file = open('histograms_by_nc.pickle', 'rb')
#     histograms_by_nc = pickle.load(file)
#     file.close()
    # neuron_class = "L5_EXC"

    # Look at:
    # https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579/4

    import pickle
    file = open('pickles_31-1-24/data_with_rotation_info_simulation_spikes.pickle', 'rb')
    data_with_rotation_info = pickle.load(file)
    file.close()
    histograms_by_nc = data_with_rotation_info['histograms_by_nc']
    neuron_class = "L4_EXC"

    # import pickle
    # file = open('pickles_31-1-24/data_with_rotation_info_spike_sorted.pickle', 'rb')
    # data_with_rotation_info = pickle.load(file)
    # file.close()
    # histograms_by_nc = data_with_rotation_info['histograms_by_nc']
    # neuron_class = "all"

    print("Num GPUs: " + str(torch.cuda.device_count()))
    torch.multiprocessing.set_start_method("spawn")
    # Define two devices
    device1 = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    devices = [device1, device2, device1, device2]

    

    model_packages = []
    model_package_1 = ModelPackage(histograms_by_nc, neuron_class, devices[0], 1, 2000)
    model_packages.append(model_package_1)

    
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
    for _ in range(len(processes)):
        print(queue.get())
    #     # model_state_dict, loss = queue.get()
    #     # model.load_state_dict(model_state_dict)

    print("DONE")

if __name__ == "__main__":
    main()