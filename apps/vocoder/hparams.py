# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    name="wavenet_vocoder",

    # Audio:
    # sample_rate=22050, # ljspeech
    sample_rate=16000, # cmu_arctic

    silence_threshold=2,
    num_mels=80,
    fft_size=1024,
    # shift can be specified by either hop_size or frame_shift_ms
    hop_size=256,
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,
    rescaling=True,
    rescaling_max=0.999,
    
    # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db, causing clipping noise.
    # If False, assertion is added to ensure no clipping happens.o0
    allow_clipping_in_normalization=True,

    # Mixture of logistic distributions:
    log_scale_min=float(np.log(1e-14)),

    # global condition if False set global channel to None

    # ljspeech
    # gc_enable=False,
    # global_channel=16,
    # global_cardinality=-1,  # speaker num

    # cmu_arctic
    gc_enable=True,
    global_channel=16,
    global_cardinality=7,  # speaker num

    # Model:
    # This should equal to `quantize_channels` if mu-law quantize enabled
    # otherwise num_mixture * 3 (pi, mean, log_scale)
    out_channels=10 * 3,
    filter_width=3,
    initial_filter_width=1,
    layers=24,
    stacks=4,
    residual_channels=512,
    dilation_channels=512,
    skip_channels=256,
    input_type="raw",
    quantize_channels={
        "raw": 2**16,
        "mu-raw": 2**8
    },
    use_biases=True,
    scalar_input=True,
    upsample_conditional_features=True,
    upsample_factor=[4, 4, 4, 4],
    l2_regularization_strength=None,

    # Training
    batch_size=2,
    sample_size=15000,
    checkpoint_interval=10000,
    train_eval_interval=10000,
    clip_thresh=-1,
    initial_learning_rate=1e-4,
    max_num_step=int(1e6),

    MOVING_AVERAGE_DECAY=0.9999,
    LEARNING_RATE_DECAY_FACTOR=0.5,
    NUM_STEPS_RATIO_PER_DECAY=0.3,

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
