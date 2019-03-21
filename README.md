# midi2wave
This is a pytorch implementation of the "midi2wave" Wavenet component outlined in the paper [Enabling Factorized Music Modeling and Generation with the Maestro Dataset](https://arxiv.org/abs/1810.12247). midi2wave is a Wavenet conditioned on midi data, which can synthesize professional-sounding piano audio. The preprocessing tools provided generate wavenet input from the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro), a large, high-qaulity dataset consisting of piano audio and midi data.


# State of project:

* Discretized logistic mixture loss and sampling implemented 
* Teacher-forced audio synthesis sounds good 
* Inference in python or with [nv-wavenet](https://github.com/NVIDIA/nv-wavenet) (need to take care that wavenet configration is compatible with nv-wavenet) 
* [Fast Wavenet](https://arxiv.org/abs/1611.09482) inference with dilated convolution caching implemented.
* Implemented an autoregressive Wavenet autoencoder for learning a useful latent representation of midi data. The encoder Wavenet is the 'context stack' mentioned in the Maestro paper.
* Can optionally turn the autoencoder into a discretized variational autoencoder with a uniform latent code distribution, following the specification of the argmax autoencoder in [Modelling Raw Audio at Scale](https://arxiv.org/abs/1806.10474)


Training issues:
* WaveNet not expected to train without the encoder Wavenet (midi features too sparse)
* Decoder Wavenet learns to ignore input from the encoder Wavenet, as its powerful enough to learn audio itself as long as its teacher-forced. This is known as posterior collapse.
* Conducted experiments with schduled sampling but didn't help. [Scheduled Sampling](https://arxiv.org/pdf/1610.09038.pdf)
* Will continue troubleshooting the posterior collapse issue now that its been identified

Components of midi2wav unimplemented:
* Global conditioning features (waiting on normal conditioning features to work well first)

# Dependencies
* pytorch 1.0
* librosa
* pretty-midi
* numpy
* scipy

# Preprocessing

* Download the Maestro dataset (>100 GBs) [here](https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip). 
* Run resample.py to downsample to 16kHz and mono-ize audio. This step takes a very long time which is why I've separated it, you should only ever have to do it once. It stores resampled audio in same directory as original maestro data by default (with altered names from original so no overwrite)
* Prepare a processed data directory: mkdir data 
* Run preprocess_maestro.py with the provided config files to make training and test data.

```
python resample_audio.py -d /path/to/maestro-v1.0.0
mkdir data
mkdir data/train data/validation data/test
python preprocess_audio.py -c preprocess_train.py -d train
python preprocess_audio.py -c preprocess_test.py -d test
```

# Training

Training parameters are stored in config.json. The Wavenet configuration in the provided config.json is a WaveNet autoencoder, with a Wavenet to turn midi features into latent code, and an autoregressive decoder Wavenet to generate audio conditioned on that latent code. The audio Wavenet is nearly the same as the default [nv-wavenet](https://github.com/NVIDIA/nv-wavenet/tree/master/pytorch), but with logistic mixture output. The decoder Wavenet follows all specifications provided in the Maestro paper about their "context stack" Wavenet.

To train the Wavenet run:
```
python train.py -c config.json
```

# Testing

I've provided an inference module for audio generation. Its possible to use the nv-wavenet cuda inference module, but one should take care that the specified Wavenet parameters are compatible with the available nv-wavenet architectures. 

My testing procedure has to generate 4s audio samples, the first 2s using teacher forcing, the second 2s in autoregressive mode. This should help the Wavenet generate audio by providing it with a history of 'good' samples to begin autoregressison from. For this mode of inference, do:

```
python inference.py -f /path/to/filelist.csv -c /path/to/wavenet_checkpoint -o /path/to/outdir -s logistic_mix -i pytrain --teacher_force 2.0
```

There are other command-line options available but this is probably the most useful way to test while fixing the posterior collapse problem. In the near future I will change this method so it requires a config file instead of command line arguments.
