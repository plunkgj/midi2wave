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
import os
import torch
import numpy as np
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def as_variable(x, device, requires_grad=True):
    x = x.contiguous()
    if device.type == "cuda":
        x = x.cuda(device, non_blocking=True)
    return torch.autograd.Variable(x)

def mu_law_decode_numpy(x, mu_quantization=256):
    assert(np.amax(x) <= mu_quantization)
    assert(np.amin(x) >= 0)
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

def mu_law_decode(x, mu_quantization=256):
    assert(torch.max(x) <= mu_quantization)
    assert(torch.min(x) >= 0)
    x = x.float()
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**torch.abs(signal) - 1)
    return torch.sign(signal) * magnitude

def mu_law_encode(x, mu_quantization=256):
    assert(torch.max(x) <= 1.0)
    assert(torch.min(x) >= -1.0)
    mu = mu_quantization - 1.
    scaling = np.log1p(mu)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / scaling
    encoding = ((x_mu + 1) / 2 * mu + 0.5).long()
    return encoding

def mu_law_encode_numpy(x, mu_quantization=256):
    assert(np.amax(x) <= 1.0)
    assert(np.amin(x) >= -1.0)
    mu = mu_quantization - 1.
    scaling = np.log1p(mu)
    x_mu = np.sign(x) * np.log1p(mu * np.absolute(x)) / scaling
    encoding = ((x_mu + 1) / 2 * mu + 0.5).astype("int64")
    return encoding

def gumbel_noise_like(X, floor=1e-5):
    u = torch.zeros(X.size()).uniform_(1e-5, (1 - 1e-5)).to(X.device)
    return -torch.log(-torch.log(u))


class CategoricalSampler(torch.nn.Module):

    def __init__(self):
        super(CategoricalSampler, self).__init__()
    
    def forward(self, X):
        """
        X is a unscaled probability distribution (B x C x T)
        """
        # add batch dim if needed
        if (len(X.size()) == 2):
            X = X.unsqueeze(0)

        if (len(X.size()) > 1):
            X = torch.transpose(X, 1, 2)
            
        return torch.distributions.categorical.Categorical(logits=X).sample()

class UniformSampler(torch.nn.Module):

    def __init__(self):
        super(UniformSampler, self).__init__()
        
    def forward(self, size, floor=1e-5):
        """
        shape is an array of Tensor shape
        """

        low = torch.full(size, floor)
        high = torch.full(size, 1-floor)
        return torch.distributions.uniform.Uniform(low, high).sample()
 
    
