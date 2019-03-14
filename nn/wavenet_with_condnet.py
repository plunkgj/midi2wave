# Gary Plunkett
# Feburary 2019
# Wavenet with a context wavenet used to preprocess midi features,
# as in "Fatcorized Music Modelling with the Maestro Dataset"

import torch
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
        cond_features = self.cond_wavenet.forward((null_features, midi_features), training)
        assert(cond_features.size(2) == midi_features.size(2))

        model_input = (cond_features, forward_input)
        return self.wavenet.forward(model_input, training)

    def export_weights(self):
        """
        Returns a dictionary for conditioning and audio wavenets seperately
        """
        model = {}
        model["wavenet"] = self.wavenet.export_weights()
        model["cond_wavenet"] = self.cond_wavenet.export_weights()
        return model

    def inference(self, midi_features, **kwargs):

        print(midi_features.size())
        null_fetaures = torch.zeros(midi_3features.size(0), 1, midi_features.size(2)).to(midi_features.device)
        cond_features = self.cond_wavenet((null_features, midi_features))
        print(cond_features.size())
        
        return  self.wavenet.inference(cond_features, **kwargs)
