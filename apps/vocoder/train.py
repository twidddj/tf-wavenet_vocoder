# coding: utf-8
import sys
import os
import tensorflow as tf
import argparse
from time import time

module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from apps.vocoder.model import Vocoder, optimizer_factory
from apps.vocoder.hparams import hparams
from apps.vocoder.datasets.data_feeder import DataFeeder
from apps.vocoder.audio import save_wav

PRINT_LOSS_EVERY = 100


def get_arguments():
    parser = argparse.ArgumentParser(description='Training Wavenet Vocoder')
    parser.add_argument('--metadata_path', type=str, default=None, help='')
    parser.add_argument('--data_path', type=str, default=None, help='')
    parser.add_argument('--log_dir', type=str, default='./apps/vocoder/logs',
                        help='Directory in which to store the logging')
    return parser.parse_args()


def train(log_dir, metadata_path, data_path):
    tf.reset_default_graph()

    vocoder = Vocoder(hparams)
    vocoder.init_synthesizer(hparams.batch_size)

    coord = tf.train.Coordinator()
    reader = DataFeeder(
        metadata_filename=metadata_path,
        coord=coord,
        receptive_field=vocoder.net.receptive_field,
        gc_enable=hparams.gc_enable,
        sample_size=hparams.sample_size,
        npy_dataroot=data_path,
        num_mels=hparams.num_mels,
        speaker_id=None
    )

    if hparams.gc_enable:
        audio_batch, lc_batch, gc_batch = reader.dequeue(hparams.batch_size)
    else:
        audio_batch, lc_batch = reader.dequeue(hparams.batch_size)
        gc_batch = None

    loss = vocoder.loss(audio_batch, lc_batch, gc_batch)

    sess = tf.Session()

    all_params = tf.trainable_variables()
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    decay_steps = hparams.NUM_STEPS_RATIO_PER_DECAY * hparams.max_num_step
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(hparams.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    hparams.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # lr = hparams.initial_learning_rate
    optimizer = optimizer_factory['adam'](learning_rate=lr, momentum=None)

    if hparams.clip_thresh > 0:
        grads_and_vars = optimizer.compute_gradients(loss, all_params)
        grads_and_vars = list(filter(lambda t: t[0] is not None, grads_and_vars))
        capped_gvs = [(tf.clip_by_norm(grad, hparams.clip_thresh), var) for grad, var in grads_and_vars]
        optim = optimizer.apply_gradients(capped_gvs)
    else:
        optim = optimizer.minimize(loss, var_list=all_params, global_step=global_step)

    # Track the moving averages of all trainable variables.
    ema = tf.train.ExponentialMovingAverage(hparams.MOVING_AVERAGE_DECAY, global_step)
    maintain_averages_op = tf.group(ema.apply(all_params))
    train_op = tf.group(optim, maintain_averages_op)

    sess.run(tf.global_variables_initializer())

    last_step, _ = vocoder.load(sess, log_dir)
    last_step = last_step or 0

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    try:
        print_loss = 0.
        start_time = time()
        for step in range(last_step + 1, hparams.max_num_step + 1):

            if gc_batch is None:
                fetches = [audio_batch, vocoder.upsampled_lc, loss, train_op]
                _x, _lc, _loss, _ = sess.run(fetches)
                _gc = None
            else:
                fetches = [audio_batch, vocoder.upsampled_lc, gc_batch, loss, train_op]
                _x, _lc, _gc, _loss, _ = sess.run(fetches)

            print_loss += _loss

            if step % PRINT_LOSS_EVERY == 0:
                duration = time() - start_time
                print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(
                    step, print_loss / PRINT_LOSS_EVERY, duration / PRINT_LOSS_EVERY))
                start_time = time()
                print_loss = 0.

            if step % hparams.checkpoint_interval == 0:
                vocoder.save(sess, log_dir, step)

            if step % hparams.train_eval_interval == 0:
                samples = vocoder.synthesize(sess, _x.shape[1], _lc, _gc)
                targets = _x.reshape(hparams.batch_size, -1)

                for j in range(hparams.batch_size):
                    predicted_wav = samples[j, :]
                    target_wav = targets[j, :]
                    predicted_wav_path = os.path.join(log_dir, 'predicted_{}_{}.wav'.format(step, j))
                    target_wav_path = os.path.join(log_dir, 'target_{}_{}.wav'.format(step, j))
                    save_wav(predicted_wav, predicted_wav_path)
                    save_wav(target_wav, target_wav_path)

    except Exception as error:
        print(error)
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == "__main__":
    args = get_arguments()
    train(os.path.expanduser(args.log_dir),
          os.path.expanduser(args.metadata_path),
          os.path.expanduser(args.data_path))
