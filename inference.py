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
import math
from scipy.io.wavfile import write
import numpy as np
import torch
import torch.nn.functional as F
import utils
import debug
from nn.discretized_mix_logistics import SampleDiscretizedMixLogistics

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def sample_output(output):
    output = output.view(-1, 3)    

# Infer one midi-upsample-field chunk at a time
def py_infer(midi_files, model_filename, output_dir, device, midi_hz, audio_hz, upsample_field_size, use_logistic_mix):

    # Load midi files and wavenet model
    midi_files = utils.files_to_list(midi_files)
    model = torch.load(model_filename)['model'].to(device)

    fig, ax = plt.subplots()
    
    # Infer audio for each midi file
    for f, filename in enumerate(midi_files):
        
        print("generating file " + filename)
        
        # Load this midi file
        midiX = torch.load(filename + ".midiX")
        if no_pedal:
            midi = midi[:88]                
        midiX = torch.nn.functional.relu(midiX)
        
        plt.cla()
        ax.spy(midiX, markersize=3, aspect="auto", origin='lower')
        plt.savefig(filename + "_inInfer.png")        
        
        # Add batch dimension
        midiX = midiX.unsqueeze(0).to(device)

        audio = model.inference(midiX, use_logistic_mix=use_logistic_mix)
        print(audio.size())
        audio = audio.squeeze().cpu().numpy()
        write(filename + "_pyinfer.wav", args.audio_hz, audio)


def train_mode_output(midi_files, model_filename, output_dir, device,
                      use_logistic_mix=False, audio_hz=16000, teacher_length=-1):

    midi_files = utils.files_to_list(midi_files)
    model = torch.load(model_filename)['model'].to(device)
    teacher_samples = teacher_length*audio_hz

    if use_logistic_mix:
        sampler = SampleDiscretizedMixLogistics()
    else:
        sampler = utils.CategoricalSampler()
    
    for filename in midi_files:

        print(filename)
        midi = torch.load(filename + ".midiX")
        if no_pedal:
            midi = midi[:88]
        midi = torch.nn.functional.relu(midi)
        
        audio_target = torch.load (filename + ".audioX")

        # Add a batch dimension
        midi = midi.unsqueeze(0).to(device)
        audio_input = audio_target.unsqueeze(0).to(device)
        if (teacher_samples > 0) and (teacher_samples < audio_target.size(-1)):
            silence = torch.full_like(audio_input, 128)
            audio_input = torch.cat([audio_input[:, :teacher_samples], silence[:, teacher_samples:]], dim=-1)

        model_output = model((midi, audio_input), training=False)

        if not use_logistic_mix:
            model_output = model_output.squeeze()
            model_output = torch.nn.Softmax(dim=0)(model_output)

            np.save("test1s/0_logits_train", model_output.detach().cpu().numpy())
            plt.cla()
            plt.imshow(model_output.data.cpu(), cmap="inferno", interpolation="nearest", aspect="auto", origin="lower")
            plt.savefig("verifyData/logit_probs_train.png")

            _, mu_audio = model_output.max(dim=0)

        else:
            # Returns floats 
            mu_audio = sampler(model_output).squeeze().detach()
            filename += "_DML"

        debug.tprobe(mu_audio, "mu_audio")            
        train_audio = utils.mu_law_decode(mu_audio)
        debug.tprobe(train_audio, "train_audio_postdecode")
        print(train_audio.dtype)
        #train_audio = utils.MAX_WAV_VALUE * train_audio
        write(filename + "_train.wav", 16000, train_audio.cpu().numpy())

        exit()
        
def pyinfer_teacher_forcing(midi_files, model_filename, output_dir, device, use_logistic_mix=False, audio_hz=16000, teacher_length=-1, wavenet_num=0):

    midi_files = utils.files_to_list(midi_files)
    model = torch.load(model_filename)['model'].to(device)
    
    fig, ax = plt.subplots()
    for filename in midi_files:
        
        print(filename)
        midi = torch.load(filename + ".midiX")
        if no_pedal:
            midi = midi[:88]
        midi = torch.nn.functional.relu(midi)
        
        audio_target = torch.load (filename + ".audioX")

        plt.cla()
        ax.spy(midi, markersize=3, aspect="auto", origin='lower')
        plt.savefig(filename + "_inTeacherForce.png")
        
        # Add a batch dimension
        midi = midi.unsqueeze(0).to(device)
        audio_target = audio_target.unsqueeze(0).to(device)

        #Set length to use training audio
        if (teacher_length > 0) and ((teacher_length*audio_hz) < audio_target.size(-1)):
            teacher_samples = int(teacher_length*audio_hz)
            filename = filename + "_pyinfer_teacher" + str(teacher_length) + "s_" + str(wavenet_num) + ".wav"
        else:
            teacher_samples = audio_target.size(-1)
            filename = filename + "_pyinfer_teacherforce.wav"

        print(filename)
        print(teacher_samples)

        audio = model.inference(midi, use_logistic_mix=use_logistic_mix,
                                teacher_audio=audio_target[:, :teacher_samples])
        print(audio.size())
        audio = audio.squeeze().cpu().numpy()
        write(filename, audio_hz, audio)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', "--checkpoint_path", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument('-i', "--infer_mode", choices=['default', 'train', 'pytrain', 'slow', 'rand'],
                        default='default')
    parser.add_argument('-d', "--device",  default='cuda', help="Device name. ex 'cpu', 'cuda:1', 'cuda'")
    parser.add_argument('-s', "--sampler", choices=['softmax', 'logistic_mix'], default='softmax')
    parser.add_argument('-a', "--audio_hz", type=int, default=16000,
                        help="default=16000")
    parser.add_argument('-m', "--midi_hz", type=int, default=250,
                        help="default=250")
    parser.add_argument('-r', "--upsample_field", type=int, default=256,
                        help="Size of context upsample field. "
                        "Value of upsample_window in default wavenet params. "
                        "default=256")
    parser.add_argument("--teacher_length", type=float, default=-1,
                        help="How long to input teacher audio for. default=-1 (full length)")
    parser.add_argument("--no_pedal", action="store_true")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    use_logistic_mix = False
    if args.sampler == "logistic_mix":
        use_logistic_mix = True
        
    #torch.set_printoptions(profile="full")

    global no_pedal
    no_pedal = args.no_pedal
    
    if (args.infer_mode=='train'):
        train_mode_output(args.filelist_path, args.checkpoint_path, args.output_dir, device, use_logistic_mix, teacher_length=3)
    elif (args.infer_mode=='pytrain'):
        pyinfer_teacher_forcing(args.filelist_path, args.checkpoint_path, args.output_dir, device, use_logistic_mix, args.audio_hz, args.teacher_length)
    elif (args.infer_mode=='slow'):
        py_infer_slow(args.filelist_path, args.checkpoint_path, args.output_dir, args.midi_hz, args.audio_hz)
    elif (args.infer_mode=='rand'):
        train_mode_random(args.filelist_path, args.checkpoint_path, args.output_dir, args.midi_hz, args.audio_hz)
    else: #default
        py_infer(args.filelist_path, args.checkpoint_path, args.output_dir, device, args.midi_hz, args.audio_hz, args.upsample_field, use_logistic_mix)
