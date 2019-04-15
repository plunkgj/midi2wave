# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# *****************************************************************************

# forked from nv-wavenet/pytorch:
# https://github.com/NVIDIA/nv-wavenet/blob/master/pytorch/train.py
#
# Modified January 2018 by Gary Plunkett for use on the Maestro dataset
# Implemented aggressive inference training April 2019 (https://arxiv.org/abs/1901.05534)

import argparse
import json
import os
import time
from csv import DictWriter

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io.wavfile import write

import utils
from utils import as_variable, mu_law_encode
from maestro_dataloader import MaestroDataloader
from scheduled_sampling  import ScheduledSamplerWithPatience
import debug
from nn.wavenet import Wavenet
from nn import discretized_mix_logistics as DML
from nn.wavenet_autoencoder import WavenetAutoencoder

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = wavenet_config["n_out_channels"]

    def forward(self, inputs, targets):
        """
        inputs are batch by num_classes by sample
        targets are batch by sample
        torch CrossEntropyLoss needs
            input = batch * samples by num_classes
            targets = batch * samples
        """
        targets = targets.view(-1)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.contiguous()
        inputs = inputs.view(-1, self.num_classes)
        return torch.nn.CrossEntropyLoss()(inputs, targets)

    
class L2DiversityLoss(torch.nn.Module):
    """
    L2  diversity loss as detailed in section 3.2 of "The Challenge of 
    Realistic Music Generation: Modelling Raw Audio at Scale".
    https://arxiv.org/abs/1806.10474
    
    This term encourages the midi autoencoder distribution to be uniform across 
    dimensions. This output distribution is quantized into a one-hot via argmax,
    so making the output distribuion uniform ensures each one-hot vector has an 
    equal chance of occuring. 
    """
    def __init__(self):
        super(L2DiversityLoss, self).__init__()
        
    def forward(self, q_bar):
        """
        Notes on how this works:

        q_bar is the continous autoencoder output distribution averaged across batch and time
        Each q is normalized so sum(q)=1
        Let k be the dimensionality of q
        If q_bar is distributed uniformly, q_bar[i] = 1/k
        k*q_bar[i] is encouraged to equal 1 using L2 loss
        q_bar will be well-estimated, as there are 16,000 instances of q from every 
            second of training data
        """

        k = q_bar.size(0)
        loss = torch.sum((k*q_bar - 1) ** 2)
        return loss
    
def load_checkpoint(checkpoint_path, model, encoder_optimizer, decoder_optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    aggressive = checkpoint_dict['aggressive']    
    decoder_optimizer.load_state_dict(checkpoint_dict['decoder_optimizer'])
    encoder_optimizer.load_state_dict(checkpoint_dict['encoder_optimizer'])    
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, encoder_optimizer, decoder_optimizer, aggressive, iteration


def save_checkpoint_autoencoder(model, device, use_VAE, encoder_optimizer, decoder_optimizer, aggressive,  learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WavenetAutoencoder(wavenet_config, cond_wavenet_config, use_VAE).to(device)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'aggressive': aggressive,
                'learning_rate': learning_rate}, filepath)

    
def train(num_gpus, rank, group_name, device, output_directory, epochs, learning_rate,
          iters_per_checkpoint, batch_size, seed, checkpoint_path,
          use_scheduled_sampling=False,
          use_wavenet_autoencoder=True, use_variational_autoencoder=False, diversity_scale=0.005,
          use_logistic_mixtures=False, n_mixtures=3,
          audio_hz=16000, midi_hz=250, aggressive_loss_threshold=3.0, encoder_error_thresh=0.0):

    assert use_wavenet_autoencoder is True
    
    if num_gpus > 1:
        device = init_distributed(rank, num_gpus, group_name, **dist_config)
    device = torch.device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
        
    if use_logistic_mixtures:
        sampler = DML.SampleDiscretizedMixLogistics()
        criterion = DML.DiscretizedMixLogisticLoss()
    else:
        sampler = utils.CategoricalSampler()
        criterion = CrossEntropyLoss()

    model = WavenetAutoencoder(wavenet_config, cond_wavenet_config, use_variational_autoencoder).to(device)
    if use_variational_autoencoder:
        diversity_loss = L2DiversityLoss()
        
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if use_scheduled_sampling:
        scheduled_sampler = ScheduledSamplerWithPatience(model, sampler, **scheduled_sampler_config)
    
    encoder_optimizer = torch.optim.Adam(model.encoder_wavenet.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(model.wavenet.parameters(), lr=learning_rate)

    # Train state params
    aggressive = True
    train_encoder = False
    
    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, encoder_optimizer, decoder_optimizer, aggressive, iteration = load_checkpoint(checkpoint_path, model, encoder_optimizer, decoder_optimizer)
        iteration += 1

    # Dataloader
    trainset = MaestroDataloader(**data_config)
    if num_gpus > 1:
        train_sampler = DistributedSampler(trainset)
    else:
        train_sampler = None
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready for distributed
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    # Initialize training variables
    epoch_offset = max(0, int(iteration / len(train_loader)))
    start_iter = iteration
    
    loss_idx = 0
    loss_sum = 0
    prev_loss = 999999999
    
    print("output directory: " + output_directory)
    
    # write loss to csv file
    loss_writer = DictWriter(open(output_directory + "/train.csv", 'w', newline=''),
                             fieldnames=['iteration', 'loss'])
    loss_writer.writeheader()

    signal_writer = DictWriter(open(output_directory + "/signal.csv", "w", newline=''),
                               fieldnames=['iteration', 'cosim', 'p-dist', 'forwardMagnitude', 'midiMagnitude'])
    signal_writer.writeheader()
    
    model.train()    
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()                

            x, y = batch

            x = as_variable(x, device)
            y = as_variable(y, device)
            y_true = y.clone()

            if use_scheduled_sampling:
                y = scheduled_sampler(x, y)                

            y_preds = model((x, y))
            
            if use_wavenet_autoencoder:
                q_bar = y_preds[1]
                y_preds = y_preds[0]
                
            loss = criterion(y_preds, y_true)
            if use_variational_autoencoder:
                div_loss = diversity_loss(q_bar)
                loss = loss + (diversity_scale * div_loss)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.data.item()
            loss.backward()
                
            if aggressive and train_encoder:
                encoder_optimizer.step()
                print("Encoder step")                    
                
            elif aggressive:
                decoder_optimizer.step()
                print("Decoder step")

            else: # normal training
                encoder_optimizer.step()
                decoder_optimizer.step()
                
            print("total loss:     {}:\t{:.9f}".format(iteration, reduced_loss))
            if use_variational_autoencoder:
                print("    diversity loss: {:.9f}".format(div_loss))

            if use_scheduled_sampling:
                scheduled_sampler.update(reduced_loss)            

            # record running average of loss
            loss_sum += reduced_loss
            loss_idx += 1
            if (iteration % 10 == 0):
                loss_avg = loss_sum/loss_idx
                print("floating avg of 10: " + str(loss_avg))
                #loss_writer.writerow({"iteration": str(i),
                #                     "loss": str(reduced_loss)})

                if aggressive and loss_avg < aggressive_loss_threshold:
                    agressive = False
                elif aggressive and train_encoder and loss_avg >= (prev_loss + encoder_error_thresh):
                    train_encoder = False
                elif aggressive:
                    train_encoder = True
                    
                prev_loss = loss_avg
                loss_sum = 0
                loss_idx = 0

            # save model
            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/wavenet_{}".format(output_directory, iteration)
                    save_checkpoint_autoencoder(model, device, use_variational_autoencoder,
                                                encoder_optimizer, decoder_optimizer,
                                                aggressive, learning_rate, iteration, checkpoint_path)

            iteration += 1
            del loss
        # end loop
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()
    
    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global scheduled_sampler_config
    scheduled_sampler_config = config["scheduled_sampler_config"]
    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config 
    wavenet_config = config["wavenet_config"]
    global cond_wavenet_config
    cond_wavenet_config = config["cond_wavenet_config"]
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1
    
    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, **train_config)
