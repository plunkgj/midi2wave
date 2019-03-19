# midi2wave
This is a pytorch implementation of the "midi2wave" Wavenet component outlined in the paper [Enabling Factorized Music Modeling and Generation with the Maestro Dataset](https://arxiv.org/abs/1810.12247). midi2wave is a Wavenet conditioned on midi data, which can synthesize professional-sounding piano audio. The preprocessing tools provided generate wavenet input from the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro), a large, high-qaulity dataset consisting of piano audio and midi data.


# State of project:

* Discretized logistic mixture loss and sampling implemented 
* Teacher-forced audio synthesis sounds good 
* Inference in python or with [nv-wavenet](https://github.com/NVIDIA/nv-wavenet) (need to take care that wavenet configration is compatible with nv-wavenet) 
* [Fast Wavenet](https://arxiv.org/abs/1611.09482) inference with dilated convolution caching implemented.
* Context wavenet preprocesses midi input before input to audio wavenet as conditioning vectors.


Training issues:
* Inference audio doesn't respond to midi input at all. 
** Conducted experiments with schduled sampling but didn't help. [Scheduled Sampling](https://arxiv.org/pdf/1610.09038.pdf)
** Experimenting with making the context wavenet output follow a uniform distirbution, turning the two wavenets into a variational autoencoder. Following quatization schemes in sections 3.1 and 3.2 in [Modelling Raw Audio at Scale](https://arxiv.org/pdf/1806.10474.pdf). No success yet.

Components of midi2wav unimplemented:
* Global conditioning features (waiting on normal conditioning features to work well first)

#---More detail coming 3/21/19 ---

# Preprocessing

//TODO: Add more detail here about resample and preprocess_maestro commands

* Download the Maestro dataset (>100 GBs) [here](https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip). 
* Run resample.py to downsample and mono-ize audio 
* Prepare a processed data directory: mkdir test_data 
* Run preprocess_maestro.py to save wavenet-ready data to disk 

# Training

'python train.py -c config.json'

//TODO: Add details about config.json parameters

# Testing

Still need to separate MakeTestData scrit from preprocess_maestro.py
