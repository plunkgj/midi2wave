"""
 Gary Plunkett, March 2019
 adapted from the PixelCNN++ TF implementation:
 https://github.com/openai/pixel-cnn/blob/ac8b1eb1703737a9664555182ce35264f8c6f88c/pixel_cnn_pp/nn.py

"""

import torch
import torch.nn.functional as F
import numpy as np

import utils
import debug

class DiscretizedMixLogisticLoss(torch.nn.Module):
    """
    Todo...
    """
    
    def __init__(self):
        super(DiscretizedMixLogisticLoss, self).__init__()
        
    def forward(self, l, y, sum_all=True):
    
        """
        x input is B x (3*n_gaus) x T
        assumes data is unscaled
        """
        assert(l.size(1) % 3 == 0)

        n_gaus = l.size(1) // 3
        #length = l.size(-1)
        
        if (len(y.size()) == 2):
            y = y.unsqueeze(1)

        # translate y [0, 255] to [-1, 1]
        y = (y.float() / 127.5) - 1
        
        # set lowest allowable probability value (prevent NaNs)
        floor = torch.FloatTensor([1e-8]).to(y.device)
        
        mix_logits = l[:,         :n_gaus,   :]
        means      = l[:,   n_gaus:2*n_gaus, :]
        log_var    = l[:, 2*n_gaus:        , :]

        # activate input
        mix_logits = F.softmax(mix_logits, dim=1)
        means = F.tanh(means)

        # max prob of logistic distribution w/ log_var=-7 is 0.97324        
        log_var = -7 * F.sigmoid(log_var)
        
        centered_x = y - means
        inv_stdv = torch.exp(-log_var)
        plus_in = inv_stdv * (centered_x + 1./255.)
        min_in = inv_stdv * (centered_x - 1./255.)
        
        cdf_plus = F.sigmoid(plus_in)
        cdf_min = F.sigmoid(min_in)
        cdf_delta = cdf_plus - cdf_min
        log_cdf_plus = plus_in - F.softplus(plus_in) # equiv to log(sig(plus_in))
        log_one_minus_cdf_min = -F.softplus(min_in)  # equiv to log(1 - sig(min_in))
        
        # log prob in bin center, used for extreme cases
        mid_in = inv_stdv * centered_x
        log_pdf_mid = mid_in - log_var - 2*F.softplus(mid_in)
        
        log_probs_plus = (y < -0.999).float() * cdf_plus
        log_probs_min = (y > 0.999).float() * cdf_min
        log_probs_extreme = (cdf_delta < 1e-5).float() * (log_pdf_mid - np.log(127.5))
        log_probs = (cdf_delta > 1e-5).float() * torch.log(torch.max(cdf_delta, floor))
        
        log_probs = log_probs + log_probs_extreme 
        log_probs = log_probs * (y > -0.999).float() * (y < 0.999).float()
        log_probs = log_probs + log_probs_plus + log_probs_min

        log_probs = log_probs + torch.log(mix_logits)
        
        # prob at time t = Sum(mix_probs[b, :, t]) = Sum(exp(log_probs[b, :, t]))
        # return avg -log_prob 
        prob_sum = torch.max(torch.sum(torch.exp(log_probs), dim=1), floor)

        NLLLoss = torch.sum(-torch.log(prob_sum)) / y.size(-1)

        return NLLLoss


class SampleDiscretizedMixLogistics(torch.nn.Module):
    """
    Todo. Can I train in inference mode here???
    """
    
    def __init__(self):
        super(SampleDiscretizedMixLogistics, self).__init__()
        self.categorical_sampler = utils.CategoricalSampler()
        self.uniform_sampler = utils.UniformSampler()
        
    def forward(self, l, quantize_output=True):
        """
        Expects (B x 3*n_mix x T) input
        """

        device = l.device
        # maybe add batch dim
        if (len(l.size()) == 1):
            l = l.unsqueeze(0)
        # maybe add time dimension
        if (len(l.size()) == 2):
            l = l.unsqueeze(-1)
        
        assert(l.size(1) % 3 == 0)
        n_gaus = l.size(1) // 3
        n_batch = l.size(0)
        length = l.size(-1)        
        
        # unpack params
        mix_logits = l[:,         :n_gaus,   :]
        means      = l[:,   n_gaus:2*n_gaus, :]
        log_var    = l[:, 2*n_gaus:        , :]

        # activate params
        means = F.tanh(means)
        log_var = -7 * F.sigmoid(log_var) # max prob of logistic distribution w/ log_var=-7 is 0.97324 ()
        
        # pick which mixture component to use from mix_logits
        # uses "Gumbel-max trick" to sample from raw logits:
        # argmax(logits_i - gumbel_noise_i) is equal to sampling from softmax
        logit_sel = self.categorical_sampler(mix_logits)

        # get means and log_var at idx of selected logits
        selected_means = torch.zeros(0, length).to(device)
        selected_log_var = torch.zeros(0, length).to(device)
        time = torch.arange(length, dtype=torch.long)
        for b in range(n_batch):
            logits_b = logit_sel[b].squeeze()
            batch_num_vec = torch.full([length], b, dtype=torch.long)
            idx = (batch_num_vec, logits_b, time)
            means_b = means[idx].unsqueeze(0)
            log_var_b = log_var[idx].unsqueeze(0)

            selected_means = torch.cat((selected_means, means_b), dim=0)
            selected_log_var = torch.cat((selected_log_var, log_var_b), dim=0)            

        means = selected_means
        log_var = selected_log_var

        # sample from logit_fcn to get devation from mixture mean
        # logit_fcn = inverse(sigmoid)
        # random uniform sampling of logit_fcn is equal to sampling from logistic distribution
        u = self.uniform_sampler(means.size()).to(device)
        logit_fcn_sampls = torch.exp(log_var) * (torch.log(u) - torch.log(1. - u))
        x = means + logit_fcn_sampls
        x = torch.clamp(x, -1, 1)

        x = (x+1) * 127.5
        
        if quantize_output:
            x = torch.round(x).long()
        
        return x
