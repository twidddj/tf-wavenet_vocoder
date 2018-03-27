# Wavenet
The WaveNet neural network architecture directly generates a raw audio waveform, showing excellent results in text-to-speech and general audio generation.

Moreover It can be used almost all sequence generation such as text or image generation.

This repository provides some works related to [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/). 

Our works build on Ibab's [Implementation](https://github.com/ibab/tensorflow-wavenet).

## Features
- [x] Local conditioning
- [x] Generalized fast generation algorithm
- [ ] Mixture of discretized logistics loss
- [ ] Parallel Wavenet

### Generalized fast generation algorithm
We generalized [Fast wavenet](https://github.com/tomlepaine/fast-wavenet) to filter width > 1 by using Multi-Queue structured matrix which has size of (Dilation x (filter_width - 1) x batch_size x channel_size).

When you generate a sample, you must feed the number of samples that have generated to the function. This is because the queue has to choose before queueing operation.

You can find easily the modified points and details of the algorithm in <a href="./notebook/eval algorithm.ipynb">here</a>.

**Check the usage of the incremental generator in <a href="./notebook/demo fast generation.ipynb">here</a>.**

## Applications

### Vocoder


## Requirements
Code is tested on TensorFlow version 1.4 for Python 3.6.

## Related projects
- [Wavenet](https://github.com/ibab/tensorflow-wavenet)
- [Wavenet vocoder in Pytorch](https://github.com/r9y9/wavenet_vocoder)
- [Wavenet vocoder in Tensorflow](https://github.com/azraelkuan/tensorflow_wavenet_vocoder/tree/dev)
- [Fast wavenet](https://github.com/tomlepaine/fast-wavenet)

## References
- ["WaveNet: A Generative Model for Raw Audio"](https://arxiv.org/abs/1609.03499)
- ["Fast Wavenet Generation Algorithm"](https://arxiv.org/pdf/1611.09482.pdf)
- ["PixelCNN++"](https://arxiv.org/pdf/1701.05517.pdf)
- ["Parallel WaveNet: Fast High-Fidelity Speech Synthesis"](https://arxiv.org/abs/1711.10433)
