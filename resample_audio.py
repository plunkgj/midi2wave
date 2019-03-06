'''
Gary Plunkett
Jan 2018
Script to downsample and mono-ize maestro audio using librosa.
Saves output audio in the same directory as original audio unless otherwise specified. Appends ["_" + str(new_hz)] to the filename.
If saving to a different location than original audio, have to recreate the maestro directory structure there.
'''

import argparse
import csv
import librosa

'''
Resample and mono-ize audio
'''
def resample_audio(dataset, dataset_path, output_path, sample_hz, resample_type):

    print("resampling " + str(len(dataset)) + " pieces...")
    
    for i, piece in enumerate(dataset):

        print("file " + str(i), end='\r', flush=True)

        audio, sampling_rate = librosa.load(dataset_path + piece["audio_filename"],
                                            sr=sample_hz,
                                            mono=True,
                                            res_type=resample_type)
        assert(sampling_rate==sample_hz)

        filename_suffix = "_" + str(sample_hz) + ".wav"
        filename = output_path + piece["audio_filename"][:-4] + filename_suffix
        librosa.output.write_wav(filename, audio, sample_hz)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str,
                        help='/path/to/maetsro-v1.0.0/')
    parser.add_argument('--out_dir', type=str,
                        default=None,
                        help='Where to save downsampled audio. Defaults to data_dir. '
                        'Appends ["_" + str(hz)] to the filename so no overwrite occurs.')
    parser.add_argument('--hz', type=int,
                        help='Hz to downsample audio to. Default is 16000', default=16000)
    parser.add_argument('-s', '--split', type=str,
                        choices=['train', 'validate', 'test', 'all'],
                        default='all',
                        help='Which training split to downsample. Default is all')
    parser.add_argument('-r', '--res_type', type=str,
                        choices=['kaiser_best', 'kaiser_fast'],
                        default='kaiser_fast',                         
                        help='Audio resampling technique. Default is kaiser_fast. '
                        '(kaiser_best takes a very long time)' )
    args=parser.parse_args()
    
    #Read file metadata from maestro csv and sort train/validate/test files
    metadata = csv.DictReader(open(args.data_dir + 'maestro-v1.0.0.csv'))
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

    #Set out_dir to data_dir if unspecified
    out_dir=args.out_dir
    if (out_dir==None):
        out_dir=args.data_dir
            
    #Resample selected split (or all)
    if (args.split=='all') or (args.split=='train'):
        print("Resampling training set")
        resample_audio(train, args.data_dir, out_dir, args.hz, args.res_type)

    if (args.split=='all') or (args.split=='validate'):
        print("Resampling validation set")
        resample_audio(validate, args.data_dir, out_dir, args.hz, args.res_type)

    if (args.split=='all') or (args.split=='test'):
        print("Resampling test set")
        resample_audio(test, args.data_dir, out_dir, args.hz, args.res_type)
