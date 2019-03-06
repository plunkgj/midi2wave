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

# Gary Plunkett
# Modified version of inference.py from nv-wavenet
# https://github.com/NVIDIA/nv-wavenet/blob/master/pytorch/inference.py

import os
import numpy as np
from scipy.io.wavfile import write
import torch
import nv_wavenet
import wavenet_utils as utils

midi_channels = 89

def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def sample_output(output):
    output = output.view(-1, 3)
    
    
def main(midi_files, model_filename, output_dir, batch_size, implementation):

    midi_files = utils.files_to_list(midi_files)
    model = torch.load(model_filename)['model']
    wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))
    
    for files in chunker(midi_files, batch_size):
        midi_batch = []
        for file_path in files:
            print(file_path)
            midi = torch.load(file_path)#.pin_memory()
            midi = utils.to_gpu(midi)
            midi_batch.append(torch.unsqueeze(midi, 0))

        #Get conditional input for inference wavenet
        cond_input = model.get_cond_input(torch.cat(midi_batch, 0))

        audio_data = wavenet.infer(cond_input, implementation)        
        print(audio_data)
        print(audio_data.size())
        print(np.max(audio_data.cpu().numpy()))
        print(np.min(audio_data.cpu().numpy()))            

        
        for i, file_path in enumerate(files):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            audio = utils.mu_law_decode_numpy(audio_data[i,:].cpu().numpy(), wavenet.A)

            write("{}/{}_infer_noMul.wav".format(output_dir, file_name),
                  16000, audio)

            print(audio.shape)
            print(np.max(audio))
            print(np.min(audio))            
            audio = utils.MAX_WAV_VALUE * audio
            print(np.max(audio))
            print(np.min(audio))            
            wavdata = audio.astype('int16')
            print(np.max(wavdata))
            print(np.min(wavdata))            
            
            write("{}/{}_infer.wav".format(output_dir, file_name),
                  16000, wavdata)

            exit()

            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', "--checkpoint_path", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument('-b', "--batch_size", default=1)
    parser.add_argument('-i', "--implementation", type=str, default="single",
                        help="""Which implementation of NV-WaveNet to use.
                        Takes values of single, dual, or persistent""" )
    
    args = parser.parse_args()
    if args.implementation == "auto":
        implementation = nv_wavenet.Impl.AUTO
    elif args.implementation == "single":
        implementation = nv_wavenet.Impl.SINGLE_BLOCK
    elif args.implementation == "dual":
        implementation = nv_wavenet.Impl.DUAL_BLOCK
    elif args.implementation == "persistent":
        implementation = nv_wavenet.Impl.PERSISTENT
    else:
        raise ValueError("implementation must be one of auto, single, dual, or persistent")
        
    main(args.filelist_path, args.checkpoint_path, args.output_dir, args.batch_size, implementation)
