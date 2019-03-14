# ****************************
#
# Gary Plunkett
# January 2019
# Random sampling of midi and audio data from a data directory
#
# ****************************

import os
import time
import csv
import random

import numpy as np
import scipy as sp
import torch
import torch.utils.data

import utils

class Midi2SampMultihot(torch.utils.data.Dataset):

    def __init__(self, dataset_path, segment_length, midi_hz, audio_hz, only_onsets, midi_channels, no_pedal, print_file_nums=False, epoch_length=None):
        
        self.dataset_path = dataset_path
        self.segment_length = segment_length
        self.audio_segment_length = int(segment_length * audio_hz)
        self.midi_segment_length = int(segment_length * midi_hz)

        #Open the metadata csv and get file information
        metadata = csv.DictReader(open(dataset_path + "filenames.csv"))
        total_audio = 0
        file_weights = []
        midi_samples = []
        file_names = []
        for filedata in metadata:
            total_audio += int(filedata["audio_samples"])
            file_weights.append(int(filedata["audio_samples"]))
            midi_samples.append(int(filedata["midi_samples"]))
            file_names.append(filedata["index"])
        file_weights = np.array(file_weights) / total_audio

        self.midi_samples = midi_samples
        self.file_weights = file_weights
        self.file_names = file_names
        self.n_files = len(file_weights)
        self.file_nums = range(self.n_files)
        n_segments = total_audio / (segment_length * audio_hz)
        self.epoch_length = int(n_segments)
        self.print_file_nums = print_file_nums

        self.midi_channels=midi_channels
        self.only_onsets = only_onsets
        self.audio_hz = audio_hz
        self.midi_hz = midi_hz
        self.no_pedal = no_pedal

    def __getitem__(self, index):

        # loop until midi with midi onsets found:
        midi=None

        #FLAG change to only load one file
        while midi is None:

            # pick a random file
            file_num = np.random.choice(self.file_nums, p=self.file_weights)

            midi = np.load(self.dataset_path + self.file_names[file_num] + ".npz")
            midi = sp.sparse.csc_matrix((midi["data"], midi["indices"], midi["indptr"]),
                                        shape=(self.midi_channels, self.midi_samples[file_num]))
        
            assert (midi.shape[1] >= self.midi_segment_length),\
                "segment_length > length of midi file"+str(file_num)

            midi_start_pos = random.randint(0, (midi.shape[1] - self.midi_segment_length))

            midi = midi[:, midi_start_pos:(midi_start_pos + self.midi_segment_length)]
            midi = midi.todense()
            if self.only_onsets:
                np.maximum(midi, 0, midi)

            #if no midi onsets try again (disregard pedal input)
            if midi[0:88, :].nonzero()[0].shape[0] == 0: 
                midi=None

        if self.no_pedal:
            midi = midi[:88]
        audio_start_pos = int(midi_start_pos * (self.audio_hz / self.midi_hz))
        audio = np.load(self.dataset_path + self.file_names[file_num] + ".npy", mmap_mode='r')
        audio = self.slice_audio(audio, audio_start_pos, midi)
        
        midiX = torch.from_numpy(midi)
        audioX = torch.from_numpy(audio)
        
        return (midiX, audioX)


    def slice_audio(self, audio, audio_start_pos, midi):
        """
        Set audio signal to 0 until first midi onset
        """

        midi_acts = midi[:88, :].transpose().nonzero()
        if len(midi_acts[0]) is not 0:
            first_midi_samp = midi_acts[0][0]
            first_audio_samp = int(first_midi_samp * (self.audio_hz/self.midi_hz))
        else:
            first_audio_samp = 0
            
        audio = audio[(audio_start_pos+first_audio_samp):(audio_start_pos + self.audio_segment_length)]
        if first_audio_samp is not 0:
            silence = utils.mu_law_encode_numpy(np.zeros(first_audio_samp))
            audio = np.concatenate((silence, audio))

        return audio

    
    def __len__(self):
        return self.epoch_length
