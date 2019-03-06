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

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.io.wavfile import write

import utils
from utils import as_Variable, mu_law_encode
from midi2multihot import Midi2SampMultihot
import debug
from nn.wavenet import Wavenet
from nn import discretized_mix_logistics as DML
from nn.wavenet_with_condnet import Wavenet_With_Condnet

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

    
def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, device, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = Wavenet(**wavenet_config).to(device)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def save_checkpoint_cond(model, device, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = Wavenet_With_Condnet(wavenet_config, cond_wavenet_config).to(device)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def add_noise(y):
    """
    Work-in-progress, just removed from train loop for readability
    """
    noise = torch.randint(low=0, high=255, size=y.size())
    y_noise = y.clone()
    mask = torch.FloatTensor(y.size(0), y.size(-1)).uniform_() > (1 - randsamp_freq)
    noise_idx = mask.nonzero()
    for b, t in noise_idx:
        y_noise[b, t] = noise[b, t]
    
def train_on_preds(x, y, y_preds, model, mask_freq=0):
    """
    Work-in-progress, just removed from train loop for readability
    """
    # if not batched add batch dim
    if (y_preds.size() != 3):
        y_preds.unsqueeze(0)

    gumbel_noise = utils.gumbel_noise_like(y_preds)
    y_next = torch.argmax(y_preds+gumbel_noise, dim=1)

    model.zero_grad()    
    y_preds = model((x, y_next))
    return y_preds    
    
    
def train(num_gpus, rank, group_name, device, output_directory, epochs, learning_rate,
          iters_per_checkpoint, batch_size, seed, checkpoint_path, use_cond_wavenet,
          use_logistic_mixtures=False, n_mixtures=3,
          min_randsamp_freq=0, max_randsamp_freq=0, max_randsamp_iter=1,
          audio_hz=16000, midi_hz=250,
          pred_loops_min=1, pred_loops_max=1, incr_pred_loops_iters=1000, under_pred_loops_prob=0):

    device = torch.device(device)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    if use_logistic_mixtures:
        criterion = DML.DiscretizedMixLogisticLoss()
    else:
        criterion = CrossEntropyLoss()

    if (use_cond_wavenet):
        model = Wavenet_With_Condnet(wavenet_config, cond_wavenet_config).to(device)
    else:
        model = Wavenet(**wavenet_config).to(device)
        
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0004)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1

    # Dataloader
    trainset = Midi2SampMultihot(**data_config)
    if num_gpus > 1:
        train_sampler = DistributedSampler(trainset)
    else:
        train_sampler = None
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    #Initialize training variables
    loss_idx = 0
    loss_sum = 0
    randsamp_freq = min_randsamp_freq
    epoch_offset = max(0, int(iteration / len(train_loader)))
    start_iter = iteration
    pred_loops = pred_loops_min

    # update learning rates
    lrs = learning_rate * (torch.ones(pred_loops) / pred_loops)
    
    model.train()    
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):

            model.zero_grad()
            x, y = batch

            #FLAG option to add noise here
            
            x = as_Variable(x, device).float()
            y = as_Variable(y, device)

            x_in = (x, y)
            y_preds = model(x_in)

            #debug.VerifyData(x, y, y_pred, iteration)
            
            loss = criterion(y_preds, y)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.data.item()
            loss.backward()
            #optimizer.lr = lrs[0]
            optimizer.step()
            #lrs[0] = reduced_loss
            print("loss:        {}:\t{:.9f}".format(iteration, reduced_loss))
            
            if (pred_loops > 1):
                for p in range(pred_loops-1):
                    y_preds = train_on_preds(x, y, y_preds, model)
                    loss = criterion(y_preds, y)
                    reduced_loss = loss.data.item()
                    loss.backward()
                    optimizer.lr = lrs[p+1]
                    optimizer.step()
                    lrs[p+1] = reduced_loss
                    print("pred loss " + str(p+1)  + ": {}:\t{:.9f}".format(iteration, reduced_loss))
                          
                # update learning rates
                # softmax(exp(-loss))
                # highest lr for iteration with most loss
                lrs = learning_rate * F.softmax(lrs, dim=0)

            loss_sum += reduced_loss
            loss_idx += 1
            if (iteration % 50 == 0):
                print("floating avg: " + str(loss_sum/loss_idx))
                loss_sum = 0
                loss_idx = 0

            # save model
            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/wavenet_{}".format(output_directory, iteration)
                    if use_cond_wavenet:
                        save_checkpoint_cond(model, device, optimizer, learning_rate,
                                             iteration, checkpoint_path)
                    else:
                        save_checkpoint(model, device, optimizer, learning_rate, iteration,
                                        checkpoint_path)

            iteration += 1

            # Update randsamp freq
            """
            if (iteration < max_randsamp_iter):
                iter_inc = max_randsamp_iter - start_iter
                freq_inc = max_randsamp_freq - min_randsamp_freq
                randsamp_freq = ((freq_inc/iter_inc)*(iteration - start_iter))
            """
            if (pred_loops < pred_loops_max) and (iteration % incr_pred_loops_iters == 0):
                pred_loops = pred_loops+1
                lrs = torch.cat([lrs, torch.FloatTensor([0])])
            
            del loss


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
