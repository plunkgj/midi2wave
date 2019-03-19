"""
Gary Plunkett, January 2018
Script to preprocess maestro audio and midi files for wavenet training.
Expects audio to have been resampled and mono-ized by resample_audio.py

This script requires two arguments:

-c "config file"
-d "data_usage", options are "train" and "test"


The config file options explained:

maestro_dir:        Location of the root maetsro folder
split:              "train", "test", or "validate". Only preprocesses one at a time
out_dir:            Directory to save matrix data to

audio_hz:            16000 by default
midi_hz:             250 by default
mu_law_encode:       If true, audio is saved mu-encoded. Default is true
mu_quantization:     Default is 256
test_segment_length: Length of data to save in seconds. Only aplicable when saving test data, not train.
only_audio:          Only save audio data. Default false
only_midi:           Only save midi data. Default false
no_output_csv:       Dont write an output csv. By default if one exists it is overwrittenn. Default false
seperate_audio_dir:  If resampled audio saved somehwere other than the maestro dir, specify that here. Default None

"""

import argparse
import json
import csv
import torch
import torch.utils.data
import numpy as np
import pretty_midi
import random
import utils
import scipy as sp
from scipy.io.wavfile import read, write
import matplotlib
# Set non-graphical backend for saving plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def Midi2Tensor(filename, midi_hz):
    """
    Return midi onsets as a sparse numpy matrix.
    Resamples midi to the specified Hz by decimation
    Onset value is normalized note velociity
    """
    pm = pretty_midi.PrettyMIDI(filename)

    num_ticks = int(np.floor(pm.get_end_time() * midi_hz))

    notes = pm.instruments[0].notes
    pedals = pm.instruments[0].control_changes
    vel = []
    tick = []
    pitch = []
    
    # make a onehot vector for each notes' on set and offset
    for note in notes:        
        tick.append( int(np.floor(note.start * midi_hz)) )
        assert(note.pitch>20 and note.pitch<110)
        pitch.append(note.pitch - 21)
        vel.append(note.velocity / 127)
            
    # make a onehot vector for each pedal change
    for ped in pedals:
        assert(ped.number==64)
        tick.append( int(np.floor(ped.time * midi_hz)) )
        pitch.append(88)
        vel.append(ped.value / 127)

    midiX = sp.sparse.csc_matrix((vel, (pitch, tick)), shape=(89, num_ticks+1), dtype="float32")
    return midiX
    

def Audio2Vec(filename, hz, mu_law_encode, mu_quantization):
    """
    Return mu-law encoded audio as a numpy array
    """
    sampling_rate, audio = read(filename)
    assert(sampling_rate==hz)
    if mu_law_encode:
        audio = utils.mu_law_encode_numpy(audio, mu_quantization)
    return audio


def SaveTestData(audioX, midiX, fileNum, output_dir, test_segment_length, audio_hz, midi_hz, mu_law_encode=True):
    """
    Save torch tensors for inference.py
    A random segment in the piece will be chosen. The length is specified by test_segment_length

    This also plots a visualization of the midi roll, and the ground truth audio segment
    """
    
    fig, ax = plt.subplots()

    filename = output_dir + "/" + str(fileNum)
        
    # save midi tensor
    if midiX is not None:
        segment_samples = int(np.floor(midi_hz * test_segment_length))
        starting_pos = random.randint(0, midiX.shape[1] - segment_samples)
        midiX = midiX[:, starting_pos:(starting_pos + segment_samples)]
        midiX = midiX.todense()
        torch.save(torch.from_numpy(midiX), filename + ".midiX")

        # plot midi roll
        plt.cla()
        ax.spy(midiX[:89, :], markersize=3, aspect="auto", origin='lower')
        plt.savefig(filename + ".png")

    # save ground truth audio
    if audioX is not None:
        segment_samples = int(audio_hz * test_segment_length)
        audio_start_pos = int(starting_pos * (audio_hz / midi_hz))
        audioX = audioX[audio_start_pos : (audio_start_pos + segment_samples)]
        torch.save(torch.from_numpy(audioX), filename + ".audioX")        

        # save ground truth audio
        if mu_law_encode:
            raw_audio = utils.mu_law_decode_numpy(audioX)
        else:
            raw_audio = audioX.numpy()
        raw_audio = utils.MAX_WAV_VALUE * raw_audio
        wavdata = raw_audio.astype('int16')
        write(filename + "_groundTruth.wav", 16000, wavdata)

    
def PreprocessMaestro(train_or_test, maestro_dir, split, out_dir,
                       audio_hz=16000, midi_hz=250,
                       mu_law_encode=True, mu_quantization=256, test_segment_length=4,
                       only_audio=False, only_midi=False, no_output_csv=False, separate_audio_dir=None):
    """
    Save audio and midi tensors in output dir
    """
    
    if (only_audio) and (only_midi):
        print("Cannot set true both \"only_midi\" and \"only_audio\". Exiting.")
        exit()

    if separate_audio_dir is None:
        audio_dir = maestro_dir
    else:
        audio_dir = separate_audio_dir
        
    # Read maestro metadata
    metadata = csv.DictReader(open(maestro_dir + '/maestro-v1.0.0.csv'))
    test = []
    validate = []
    train = []
    for file in metadata:
        if (file['split']=='train'):
            train.append(file)
        elif (file['split']=='validation'):
            validate.append(file)
        elif (file['split']=='test'):
            test.append(file)

    if (split=="train"):
        split_data=train
    elif (split=="validate"):
        split_data=validate
    elif (split=="test"):
        split_data=test

    # Write file information to the output directory, essential for dataloader
    if not no_output_csv:
        csvwriter = csv.DictWriter(open(out_dir + "/filenames.csv", 'w', newline=''),
                                   fieldnames=["index",
                                               "audio_samples",
                                               "midi_samples",
                                               "audio_filename",
                                               "midi_filename"])
        csvwriter.writeheader()

    # start message
    t_str = "midi and audio"
    if only_midi:
        t_str=t_str[:4]
    elif only_audio:
        t_str=t_str[-5:]
    print("Making " + t_str + " tensors for " + train_or_test  + ", " + str(len(split_data)) + " files")

    # save audio as numpy array, and midi as sparse numpy matrix
    for i, piece in enumerate(split_data):

        print("file" + str(i), end='\r', flush=True)

        # audio downsampled by resample_audio.py will have "_Hz" appended to filename
        audio_suffix =  "_" + str(audio_hz) + ".wav"
        audio_filename = audio_dir + "/" + piece["audio_filename"][:-4] + audio_suffix

        midi_filename = maestro_dir + "/" + piece["midi_filename"]        

        # Save audio array
        audioX = None
        if not only_midi:
            audioX = Audio2Vec(audio_filename, audio_hz,
                               mu_law_encode, mu_quantization)
            if train_or_test=="train": 
                np.save(out_dir + str(i), audioX)

        # Save sparse midi tensor
        midiX = None
        if not only_audio:
            midiX = Midi2Tensor(midi_filename, midi_hz)
            if train_or_test=="train":
                np.savez(out_dir + str(i),
                         data=midiX.data,
                         indices=midiX.indices,
                         indptr=midiX.indptr)

        if train_or_test=="test":
            SaveTestData(audioX, midiX, i, out_dir, test_segment_length, audio_hz, midi_hz, mu_law_encode)
                
        if not no_output_csv:
            csvwriter.writerow({"index": str(i),
                                "audio_samples": audioX.shape[0],
                                "midi_samples": midiX.shape[1],
                                "audio_filename": piece["audio_filename"],
                                "midi_filename": piece["midi_filename"] })


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="Location of configuration file")
    parser.add_argument('-d', '--data_usage', type=str, choices=["train", "test"],
                        help="What is this data going to be used for? Train and test data saved differently.")
    
    args = parser.parse_args()

    # Parse config file
    with (open(args.config)) as f:
        data = f.read()
    config = json.loads(data)["preprocess_config"]

    PreprocessMaestro(args.data_usage, **config)
