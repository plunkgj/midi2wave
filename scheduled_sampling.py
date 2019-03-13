# Gary Plunkett, March 2019
# Scheduled Sampling module (https://arxiv.org/abs/1506.03099)

# to try - backprop through scheduled sampler via torch's categorical sampler

from collections import deque
import numpy as np
import torch


class LinDecay(torch.nn.Module):
    def __init__(self, start_y, end_y, dx):
        super(LinDecay, self).__init__()
        self.M = (start_y - end_y) / (dx - 1)
        self.B = start_y

    def forward(self, x):
        # input is scalar
        return self.B - (self.M * x)


class ScheduledSamplerWithPatience(torch.nn.Module):

    def __init__(self, model, sampler,
                 start_loops=0, end_loops=1,
                 start_epsilon=1., end_epsilon=0.1, decay_iters=2500,
                 patience=150, threshold=0.2, underloop_prob=0.1,
                 start_decaying=True):
        super(ScheduledSamplerWithPatience, self).__init__()

        self.model = model
        self.sampler = sampler
        self.start_loops = start_loops
        self.end_loops = end_loops
        self.sample_loops = start_loops
        self.patience = patience
        self.threshold = threshold
        self.underloop_prob = underloop_prob

        # initialize epsilon for each loop
        self.decaying = start_decaying
        self.decay_iters = decay_iters
        self.decay = LinDecay(start_epsilon, end_epsilon, decay_iters)
        self.start_epsilon = start_epsilon
        self.epsilon = []
        for i in range(start_loops-1):
            self.epsilon += [end_epsilon]
        if start_decaying:
            self.epsilon += [start_epsilon]
        else:
            self.epsilon += [end_epsilon]

        self.iteration = 0

        self.loss_sum = 0
        self.loss_memory = deque()
        for i in range(patience):
            self.loss_memory.append(0)
        self.prev_loss_check = None

    #FLAG add a training param
    def forward(self, x, y, training=True):
        """
        X and Y tensors for Wavenet training
        """
        
        # sample the number of sample loops to do
        if (np.random.random() < self.underloop_prob):
            # loops [0, sample_loops-1] each have an equal chance of occuring                
            if (self.sample_loops == 0):
                tmp_sample_loops = 0
            else:
                tmp_sample_loops = np.random.randint(self.sample_loops)
        else:
            tmp_sample_loops = self.sample_loops

        device = y.device

        for p in range(tmp_sample_loops):
            if (self.epsilon[p] == 1):
                continue
            y_preds = self.model((x, y), training=training)                
            y_samples = self.sampler(y_preds)

            mask = torch.zeros(y.size()).uniform_() > self.epsilon[p]
            mask = mask.long().to(device)            
            y = (y_samples * mask) + (y * -(mask - 1.))
            
        return y

    def update(self, loss):
        """
        Check loss this iteration, update epsilon
        """
        if self.decaying is None:
            # done increasing sample loops and final epsilon decay
            return
        
        self.loss_memory.appendleft(loss)
        self.loss_sum += loss        
        self.loss_sum -= self.loss_memory.pop()
        av_loss = self.loss_sum / self.patience
        
        if self.decaying:
            # update sample chance (epsilon)            
            self.epsilon[-1] = self.decay(self.iteration)
            
            # if done decaying:
            if (self.iteration == self.decay_iters-1):

                # if done with all sample loops
                if (self.sample_loops == self.end_loops):
                    self.decaying = None
                    print("########done with sched sampling#########")
                    
                else: # start waiting
                    self.decaying = False
                    self.iteration = -1
                    self.prev_loss_check = av_loss
                    print("########waiting for loss to plateau#########")
                    
        # if not decaying, check loss every "patience" iterations
        elif (self.iteration != 0) and (self.iteration%self.patience == 0):
            if self.prev_loss_check is not None:
                
                if av_loss > (self.prev_loss_check+self.threshold):
                    self.sample_loops += 1
                    self.epsilon += [self.start_epsilon]
                    self.iteration = -1
                    self.decaying = True
                    print("########starting new sched sample loop#########")
                    
            self.prev_loss_check = av_loss        

        self.iteration += 1
