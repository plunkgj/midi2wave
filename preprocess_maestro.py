# Gary Plunkett
# January 2018
# Script to preprocess maestro audio and midi files for input to
# the nv-wavenet. Expects mono audio

import argparse
import csv
import torch
import torch.utils.data
import numpy as np
import librosa
import pretty_midi
import random
import utils
import scipy as sp

import matplotlib
# Set non-graphical backend for saving plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io.wavfile import read, write

MAKE_TEST = True


def Midi2Tensor(filename, midi_hz, offset_vel=0):
    """
    Return midi onsets as a sparse numpy matrix.
    Can record offsets by specifying a nonzero offset value 
    """
    pm = pretty_midi.PrettyMIDI(filename)

    num_ticks = int(np.floor(pm.get_end_time() * midi_hz))

    notes = pm.instruments[0].notes
    pedals = pm.instruments[0].control_changes
    vel = []
    tick = []
    pitch = []
    
    #Make a onehot vector for each notes' onset and offset
    for note in notes:
        
        tick.append( int(np.floor(note.start * midi_hz)) )
        assert(note.pitch>20 and note.pitch<110)
        pitch.append(note.pitch - 21)
        vel.append(note.velocity / 127)

        tick.append( int(np.floor(note.end * midi_hz)) )
        pitch.append(note.pitch - 21)
        # Greater than 0 will mirror onset vel
        if (offset_vel>0):
            vel.append(-note.velocity/127)
        else:
            vel.append(offset_vel)
            
    #Make a onehot vector for each pedal change
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


def MakeTestData(dataset_path, output_dir, metadata, test_segment_length, audio_hz, midi_hz, offset_vel):

    filelist = open(output_dir + "/filenames.txt", 'w')    
    fig, ax = plt.subplots()

    for i, piece in enumerate(metadata):

        print(str(i) + ", " + piece["audio_filename"] )

        filename = output_dir + "/" + str(i)
        filelist.write(filename + "\n")
        
        # save midi tensor
        midi = Midi2Tensor(dataset_path + piece["midi_filename"], midi_hz, offset_vel)
        segment_samples = int(np.floor(midi_hz * test_segment_length))
        starting_pos = random.randint(0, midi.shape[1] - segment_samples)
        midi = midi[:, starting_pos:(starting_pos + segment_samples)]
        midi = midi.todense()
        midi = torch.from_numpy(midi)
        torch.save(midi, filename + ".midiX")

        # plot midi roll
        plt.cla()
        ax.spy(midi[:89, :], markersize=3, aspect="auto", origin='lower')
        plt.savefig(filename + ".png")

        # save ground truth audio
        _, audio = read(dataset_path + piece["audio_filename"][:-4] + "_" + str(audio_hz) + ".wav")
        audio_seg_samps = int(audio_hz * test_segment_length)
        audio_start_pos = int(starting_pos * (audio_hz / midi_hz))
        audio = audio[audio_start_pos : (audio_start_pos + audio_seg_samps)]
        audioX = utils.mu_law_encode_numpy(audio) # save for tensor
        audio = utils.mu_law_decode_numpy(audioX)
        audio = utils.MAX_WAV_VALUE * audio
        wavdata = audio.astype('int16')
        write(filename + "_target.wav", 16000, wavdata)
        torch.save(torch.from_numpy(audioX), filename + ".audioX")

    filelist.close()
        
#~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--maestro_dir', type=str,
                        help='Location of meastro-v1.0.0 directory.')
    parser.add_argument('-s', '--split', type=str,
                        choices=['train', 'validate', 'test'],
                        help='Which training split to preprocess.')    
    parser.add_argument('-a', '--audio_dir', type=str,
                        default=None,
                        help='Location of audio directory. Audio should be mono'
                        ' and have ["_" + str(hz)] appended to filename. '
                        'Default is same as maestro_dir.')
    parser.add_argument('-o', '--out_dir', type=str,
                        default="./processed_data/",
                        help='Default is ./processed_data/' )
    parser.add_argument('--audio_hz', type=int,
                        default=16000,
                        help='Sample rate of audio files being read.'
                        'Default is 16000')
    parser.add_argument('--midi_hz', type=int,
                        default=250,
                        help='Sample rate to convert midi files to. Default is 250')
    parser.add_argument('-v', '--offset_vel', type=float,
                        default=-1.0,
                        help='Normalized velocity value indicating a note_off in '
                         'midi vectors. Suggested value range is [-1, 0), but any'
                         ' float is possible. offset_val>0 means mirror onset_vel.')
    parser.add_argument('--mu_law_encode', type=bool, default=True,
                        help='Whether to save audio using mu law encoding.'
                        'Default is True')
    parser.add_argument('--mu_quantization', type=int, default=256,
                        help='Number of possible values for mu-law qunatization. '
                        'Default is 256')
    parser.add_argument('--only_audio', action='store_true')
    parser.add_argument('--only_midi', action='store_true')
    parser.add_argument('--no_csv', action='store_true',
                        help='Dont overwrite csv. Useful if its already been created.')
    args = parser.parse_args()
    
    if (args.only_audio) and (args.only_midi):
        print("Cannot specify both --only_midi and --only_audio. Exiting script")
        exit()

    audio_dir = args.audio_dir
    if (audio_dir==None):
        audio_dir = args.maestro_dir
        
    #Read metadata
    metadata = csv.DictReader(open(args.maestro_dir + '/maestro-v1.0.0.csv'))
    test = []
    validate = []
    train = []
    for data in metadata:
        if (data['split']=='train'):
            train.append(data)
        elif (data['split']=='validation'):
            validate.append(data)
        elif (data['split']=='test'):
            test.append(data)

    split_name=args.split
    if (args.split=="train"):
        split_data=train
    elif (args.split=="validate"):
        split_data=validate
    elif (args.split=="test"):
        split_data=test

        
    if (MAKE_TEST):
        MakeTestData(args.maestro_dir, args.out_dir, validate, 1, args.audio_hz, args.midi_hz, args.offset_vel)
        exit()
        
    if not args.no_csv:
        csvwriter = csv.DictWriter(open(args.out_dir + "/filenames.csv", 'w', newline=''),
                                   fieldnames=["index",
                                               "audio_samples",
                                               "midi_samples",
                                               "audio_filename",
                                               "midi_filename"])
        csvwriter.writeheader()

    t_str = "midi and audio"
    if args.only_midi:    t_str=t_str[:4]
    elif args.only_audio: t_str=t_str[-5:]
    print("Making " + t_str + " tensors for " + str(len(split_data)) + " files")
        
    #Save midi and audio data as tensors
    for i, piece in enumerate(split_data):

        print("file" + str(i), end='\r', flush=True)
        
        audio_suffix =  "_" + str(args.audio_hz) + ".wav"
        audio_filename = audio_dir + "/" + piece["audio_filename"][:-4] + audio_suffix
        midi_filename = args.maestro_dir + "/" + piece["midi_filename"]        

        #Save audio array
        if not args.only_midi:
            audioX = Audio2Vec(audio_filename,
                               args.audio_hz,
                               args.mu_law_encode,
                               args.mu_quantization)

            np.save(args.out_dir + str(i), audioX)

        #Save sparse midi tensor
        if not args.only_audio:
            midiX = Midi2Tensor(midi_filename,
                                args.midi_hz,
                                args.offset_vel)

            np.savez(args.out_dir + str(i),
                     data=midiX.data,
                     indices=midiX.indices,
                     indptr=midiX.indptr)
            
        if not args.no_csv:
            csvwriter.writerow({"index": str(i),
                                "audio_samples": audioX.shape[0],
                                "midi_samples": midiX.shape[1],
                                "audio_filename": piece["audio_filename"],
                                "midi_filename": piece["midi_filename"] })
