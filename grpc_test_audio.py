#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import queue
import soundfile
import librosa

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

FLAGS = None

WAV_SCALE_FACTOR = 2**15-1


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=16000, trim=False,
                 trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / ((2 ** (bits - 1)) - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return WAV_SCALE_FACTOR * float32_samples

    @classmethod
    def from_file(cls, filename, target_sr=16000, offset=0, duration=0,
                  min_duration=0, trim=False):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(filename, 'r') as f:
            dtype_options = {'PCM_16': 'int16',
                             'PCM_32': 'int32', 'FLOAT': 'float32'}
            dtype_file = f.subtype
            if dtype_file in dtype_options:
                dtype = dtype_options[dtype_file]
            else:
                dtype = 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        num_zero_pad = int(target_sr * min_duration - samples.shape[0])
        if num_zero_pad > 0:
            samples = np.pad(samples, [0, num_zero_pad], mode='constant')

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

# read audio chunk from a file


def get_audio_chunk_from_soundfile(sf, chunk_size):

    dtype_options = {'PCM_16': 'int16', 'PCM_32': 'int32', 'FLOAT': 'float32'}
    dtype_file = sf.subtype
    if dtype_file in dtype_options:
        dtype = dtype_options[dtype_file]
    else:
        dtype = 'float32'
    audio_signal = sf.read(chunk_size, dtype=dtype)
    end = False
    # pad to chunk size
    if len(audio_signal) < chunk_size:
        end = True
        audio_signal = np.pad(audio_signal, (0, chunk_size-len(
            audio_signal)), mode='constant')
    return audio_signal, end

# generator that returns chunks of audio data from file


def audio_generator_from_file(input_filename, target_sr, chunk_duration):

    sf = soundfile.SoundFile(input_filename, 'rb')
    chunk_size = int(chunk_duration*sf.samplerate)
    start = True
    end = False

    while not end:

        audio_signal, end = get_audio_chunk_from_soundfile(sf, chunk_size)

        audio_segment = AudioSegment(audio_signal, sf.samplerate, target_sr)

        yield audio_segment.samples, target_sr, start, end
        start = False

    sf.close()


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def sync_send(triton_client, result_list, chunk, batch_size, sequence_id,
              model_name, model_version):

    # Create the tensor for INPUT
    audio_data = np.reshape(chunk[0], (1, -1))
    sr_data = np.full(shape=[batch_size, 1],
                      fill_value=chunk[1], dtype=np.uint32)
    # print(audio_data.shape, sr_data.shape,
    #       sr_data, sequence_id, chunk[2], chunk[3])
    inputs = []
    inputs.append(grpcclient.InferInput(
        'AUDIO_SIGNAL', audio_data.shape, "FP32"))
    inputs.append(grpcclient.InferInput(
        'SAMPLE_RATE', sr_data.shape, "UINT32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(audio_data)
    inputs[1].set_data_from_numpy(sr_data)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('AUDIO_FEATURES'))
    outputs.append(grpcclient.InferRequestedOutput('AUDIO_PROCESSED'))
    # Issue the synchronous sequence inference.

    result = triton_client.infer(model_name=model_name,
                                 inputs=inputs,
                                 outputs=outputs,
                                 sequence_id=sequence_id,
                                 sequence_start=chunk[2],
                                 sequence_end=chunk[3])

    print(result.as_numpy("AUDIO_PROCESSED"))
    result_list.append(result.as_numpy('AUDIO_FEATURES'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='100.100.100.156:8001',
        help='Inference server URL and it gRPC port. Default is localhost:8001.'
    )
    parser.add_argument('-d',
                        '--dyna',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Assume dynamic sequence model')
    parser.add_argument('-o',
                        '--offset',
                        type=int,
                        required=False,
                        default=0,
                        help='Add offset to sequence ID used')

    FLAGS = parser.parse_args()

    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # We use the custom "sequence" model which takes 1 input
    # value. The output is the accumulated value of the inputs. See
    # src/custom/sequence.
    FILE_NAME = "de_example.wav"
    model_name = "riva-asr"
    model_version = ""
    batch_size = 1

    sequence_id0 = 1000 + FLAGS.offset * 2
    result0_list = []

    for audio_chunk in audio_generator_from_file(FILE_NAME, 16000, 0.1):
        # print(audio_chunk[0])
        try:
            sync_send(triton_client, result0_list, audio_chunk, batch_size,
                      sequence_id0, model_name, model_version)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    for i in range(len(result0_list)):

        print('frame: ', i)
        print((result0_list[i][0].shape))
        print((result0_list[i]))
    # if i == 0 or i == 1 or i == 2 or i == 3 or i == 4:
    # filename = "/clients/output/np_" + str(i).zfill(3) + '.npy'
    # np.save(filename, result0_list[i][0])

    #print("PASS: Sequence")
