# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import random
import numpy as np
import os
import sys
import apps.vocoder.audio as audio
from apps.vocoder.hparams import hparams


def get_file_list(metadata_filename, npy_dataroot, speaker_id):
    files = []
    with open(metadata_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            local_condition_path = line[1]
            wav_path = line[0]
            local_condition_path = os.path.join(npy_dataroot, local_condition_path)
            wav_path = os.path.join(npy_dataroot, wav_path)
            if len(line) == 5:
                global_condition = int(line[4])
            else:
                global_condition = None
            if speaker_id is None:
                files.append((wav_path, local_condition_path, global_condition))
            else:
                global_condition = None
                if int(line[4]) == speaker_id:
                    files.append((wav_path, local_condition_path, global_condition))
    return files


def randomize_file(files):
    random_file = random.choice(files)
    yield random_file


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


def load_npy_data(metadata_filename, npy_dataroot, speaker_id):
    # print("Loading data...", end="")
    files = get_file_list(metadata_filename, npy_dataroot, speaker_id)
    # print("Done!")
    # print("File length:{}".format(len(files)))
    random_files = randomize_file(files)
    for each in random_files:
        wav = np.load(each[0])
        local_condition = np.load(each[1])
        global_condition = each[2]

        yield wav, local_condition, global_condition


class DataFeeder(object):
    def __init__(self, metadata_filename, coord, receptive_field, gc_enable=False,
                 sample_size=None, queue_size=128, npy_dataroot=None, num_mels=None, speaker_id=None):
        self.metadata_filename = metadata_filename
        self.coord = coord
        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.queue_size = queue_size
        self.gc_enable = gc_enable
        self.npy_dataroot = npy_dataroot
        self.num_mels = num_mels
        self.speaker_id = speaker_id

        self.threads = []

        self._placeholders = [
            tf.placeholder(tf.float32, shape=None),
            tf.placeholder(tf.float32, shape=None)
        ]

        if self.gc_enable:
            self._placeholders.append(tf.placeholder(tf.int32, shape=None))
            self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                             [tf.float32, tf.float32, tf.int32],
                                             shapes=[(None, 1), (None, self.num_mels), ()],
                                             name='input_queue')
        else:
            self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                             [tf.float32, tf.float32],
                                             shapes=[(None, 1), (None, self.num_mels)],
                                             name='input_queue')

        self.enqueue = self.queue.enqueue(self._placeholders)

    def dequeue(self, batch_size):
        output = self.queue.dequeue_many(batch_size)
        return output

    def thread_main(self, sess):
        stop = False
        while not stop:
            iterator = load_npy_data(self.metadata_filename, self.npy_dataroot, self.speaker_id)
            for wav, local_condition, global_condition in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                # force to align the audio and local_condition
                # if audio.shape[0] > local_condition.shape[0]:
                #     audio = audio[:local_condition.shape[0], :]
                # else:
                #     local_condition = local_condition[:audio.shape[0], :]

                # audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]], mode='constant')
                # local_condition = np.pad(local_condition, [[self.receptive_field, 0], [0, 0]], mode='constant')
                # if self.sample_size:
                #     while len(audio) > self.receptive_field:
                #         audio_piece = audio[:(self.receptive_field + self.sample_size), :]
                #         audio = audio[self.sample_size:, :]
                #
                #         local_condition_piece = local_condition[:(self.receptive_field + self.sample_size), :]
                #         local_condition = local_condition[self.sample_size:, :]
                #
                #         if self.gc_enable:
                #             sess.run(self.enqueue, feed_dict=
                #             dict(zip(self._placeholders, (audio_piece, local_condition_piece, global_condition))))
                #         else:
                #             sess.run(self.enqueue, feed_dict=
                #             dict(zip(self._placeholders, (audio_piece, local_condition_piece))))
                # else:
                #     if self.gc_enable:
                #         sess.run(self.enqueue, feed_dict=dict(zip(
                #             self._placeholders, (audio, local_condition, global_condition))))
                #     else:
                #         sess.run(self.enqueue, feed_dict=dict(zip(self._placeholders, (audio, local_condition))))

                if hparams.upsample_conditional_features:
                    wav = wav.reshape(-1, 1)
                    assert_ready_for_upsampling(wav, local_condition)
                    if self.sample_size is not None:
                        sample_size = ensure_divisible(self.sample_size, audio.get_hop_size(), True)
                        if wav.shape[0] > sample_size:
                            max_frames = sample_size // audio.get_hop_size()
                            s = np.random.randint(0, len(local_condition) - max_frames)
                            ts = s * audio.get_hop_size()
                            wav = wav[ts:ts + audio.get_hop_size() * max_frames, :]
                            local_condition = local_condition[s:s + max_frames, :]
                            if self.gc_enable:
                                sess.run(self.enqueue, feed_dict=dict(zip(
                                    self._placeholders, (wav, local_condition, global_condition)
                                )))
                            else:
                                sess.run(self.enqueue, feed_dict=dict(zip(
                                    self._placeholders, (wav, local_condition)
                                )))
                else:
                    wav, local_condition = audio.adjust_time_resolution(wav, local_condition)
                    wav = wav.reshape(-1, 1)
                    if self.sample_size is not None:
                        while wav.shape[0] > self.sample_size:
                            wav_piece = wav[:(self.receptive_field + self.sample_size), :]
                            local_condition_piece = local_condition[:(self.receptive_field + self.sample_size), :]
                            wav = wav[:self.sample_size, :]
                            local_condition = local_condition[:self.sample_size, :]
                            assert len(wav_piece) == len(local_condition_piece)

                            if self.gc_enable:
                                sess.run(self.enqueue, feed_dict=dict(zip(
                                            self._placeholders, (wav_piece, local_condition_piece, global_condition))))
                            else:
                                sess.run(self.enqueue, feed_dict=dict(zip(
                                    self._placeholders, (wav_piece, local_condition_piece))))

    def start_threads(self, sess, n_threads=8):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
