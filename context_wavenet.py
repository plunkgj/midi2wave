# Gary Plunkett
# Feburary 2019
# Wavenet with a conditioning wavenet used for upsampling.

import torch
from wavenet import WaveNet

class WaveNet_With_CondWaveNet(torch.nn.Module):
    def __init__(self, wavenet_params, condwavenet_params):

        super(WaveNet_With_CondWaveNet, self).__init__()
        
        self.cond_wavenet = WaveNet(**condwavenet_params)
        self.wavenet = WaveNet(**wavenet_params)

    def forward(self, forward_input):

        midi_features = forward_input[0]
        forward_input = forward_input[1]

        #Conditioning wavenet takes in null features
        null_features = torch.zeros(midi_features.size(0), 1, midi_features.size(2)).to(midi_features.device)

        context_features = self.cond_wavenet.forward((null_features, forward_input))        

        # FLAG need to add support for adding context features
        model_input = (cond_features, forward_input, context_features)
        return self.wavenet.forward(model_input)

    def export_weights(self):
        """
        Returns a dictionary for conditioning and audio wavenets seperately
        """
        model = {}
        model["wavenet"] = self.wavenet.export_weights()
        model["cond_wavenet"] = self.cond_wavenet.export_weights()
        return model

