# Gary Plunkett
# Feburary 2019
# Wavenet with a context wavenet used to preprocess midi features,
# as in "Fatcorized Music Modelling with the Maestro Dataset"

import torch
import torch.nn.functional as F
from nn.wavenet import Wavenet

class Wavenet_With_Condnet(torch.nn.Module):
    def __init__(self, wavenet_params, condwavenet_params):

        super(Wavenet_With_Condnet, self).__init__()
        
        self.cond_wavenet = Wavenet(**condwavenet_params)
        self.wavenet = Wavenet(**wavenet_params)

    def forward(self, forward_input, training=True):

        midi_features = forward_input[0]
        forward_input = forward_input[1]
        device = midi_features.device
        
        #Conditioning wavenet takes in null features
        null_features = torch.zeros(midi_features.size(0), 1, midi_features.size(2)).to(device)
        cond_features = self.cond_wavenet((null_features, midi_features), training)

        q, q_bar = self.argmax_autoencode(cond_features)

        y_preds = self.wavenet((q, forward_input), training)
        
        return (y_preds, q_bar)

    def export_weights(self):
        """
        Returns a dictionary for conditioning and audio wavenets seperately
        """
        model = {}
        model["wavenet"] = self.wavenet.export_weights()
        model["cond_wavenet"] = self.cond_wavenet.export_weights()
        return model

    def argmax_autoencode(self, condnet_output):
        # argmax autoencoder (https://arxiv.org/abs/1806.10474)
        q = F.relu(condnet_output)
        q = q / (torch.sum(q, dim=1) + 1e-5) # divisive normalization w/ NaN safeguard if all qs negative
        q_bar = torch.mean(q, dim=1)

        # Sample a onehot from distribution with hard gumbel-softmax
        # requires 2D input :(        
        q = q.transpose(1, 2)
        for b in range(condnet_output.size(0)):
            q[b] = F.gumbel_softmax(q[b], hard=True) 
        q = q.transpose(1, 2)

        return q, q_bar
    
    def inference(self, midi_features, **kwargs):

        batch_size = midi_features.size(0)
        null_features = torch.zeros(batch_size, 1, midi_features.size(2)).to(midi_features.device)
        cond_features = self.cond_wavenet((null_features, midi_features))
        
        q, _ = self.argmax_autoencode(cond_features)
        
        return self.wavenet.inference(q, **kwargs)
