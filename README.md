# Wavenet
The WaveNet neural network architecture directly generates a raw audio waveform, showing excellent results in text-to-speech and general audio generation.

Moreover It can be used almost all sequence generation even text or image.

This repository provides some works related to [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/).

## Features
- [x] Local conditioning
- [x] Generalized fast generation algorithm
- [x] Mixture of discretized logistics loss
- [ ] Parallel Wavenet

### Generalized fast generation algorithm
We generalized [Fast wavenet](https://github.com/tomlepaine/fast-wavenet) to filter width > 1 by using Multi-Queue structured matrix which has size of (Dilation x (filter_width - 1) x batch_size x channel_size).

When you generate a sample, you must feed the number of samples that have generated to the function. This is because the queue has to choose before queueing operation.

You can find easily the modified points and details of the algorithm in <a href="./notebook/eval algorithm.ipynb">here</a>.

**Check the usage of the incremental generator in <a href="./notebook/demo fast generation.ipynb">here</a>.**

## Applications

### Vocoder

Neural Vocoder can generate high quality raw speech samples conditioned on linguistic or acoustic features.

We tested our model followed [@r9y9](https://github.com/r9y9)'s [works](https://github.com/r9y9/wavenet_vocoder).

Audio samples are available at https://twidddj.github.io/docs/vocoder. See the issue for the result in [here](https://github.com/twidddj/wavenet/issues/1).

#### Pre-trained models
| Model URL                                                                 | Data       | Steps         |
|---------------------------------------------------------------------------|------------|---------------|
| [link](https://drive.google.com/file/d/1tptZDDZ8rOWUdse0HWn9W2cm0yN0xhx6) | LJSpeech   | 680k steps    |


#### Getting Start
#### 0. Download dataset
+ the voice conversion dataset(for multi speaker, 16k): [cmu_arctic](http://festvox.org/cmu_arctic/)
+ the single speaker dataset(22.05k): [LJSpeech-1.0](https://keithito.com/LJ-Speech-Dataset/)

#### 1. Preprocess data
> `python -m apps.vocoder.preprocess --num_workers 4 --name ljspeech --in_dir /your_path/LJSpeech-1.0 --out_dir /your_outpath/` 

#### 2. Train model
> `python -m apps.vocoder.train --metadata_path {~/yourpath/train.txt} --data_path {~/yourpath/npy} --log_dir {~/log_dir_path}`

#### 3. Test model
You can find the codes for testing trained model in <a href="./apps/vocoder/notebook/test vocoder.ipynb">here</a>.

## Requirements
Code is tested on TensorFlow version 1.4 for Python 3.6.

## References
- [Wavenet](https://github.com/ibab/tensorflow-wavenet)
- [Wavenet vocoder in Pytorch](https://github.com/r9y9/wavenet_vocoder)
- [Wavenet vocoder in Tensorflow](https://github.com/azraelkuan/tensorflow_wavenet_vocoder/tree/dev)
- [Fast wavenet](https://github.com/tomlepaine/fast-wavenet)

## Related Papers
- ["WaveNet: A Generative Model for Raw Audio"](https://arxiv.org/abs/1609.03499)
- ["Fast Wavenet Generation Algorithm"](https://arxiv.org/pdf/1611.09482.pdf)
- ["PixelCNN++"](https://arxiv.org/pdf/1701.05517.pdf)
- ["Parallel WaveNet: Fast High-Fidelity Speech Synthesis"](https://arxiv.org/abs/1711.10433)
