#!/usr/bin/env python

import array
import math
import sys
import wave

import numpy

SAMPLE_RATE = 44100
SAMPLE_BYTES = 2
CHANNELS = 1

def ReadWav(filename):
  wav = None
  try:
    wav = wave.open(filename, 'r')
    channels = wav.getnchannels()
    sample_bytes = wav.getsampwidth()

    wav_data = wav.readframes(wav.getnframes())

    if (sample_bytes == 2):
      wav_array = numpy.fromstring(wav_data, dtype=numpy.int16)
    elif (sample_bytes == 1):
      wav_array = numpy.fromstring(wav_data, dtype=numpy.int8)
    else:
      raise ValueError('Sample depth of %d bytes not supported' % sample_bytes)

    float_array = numpy.zeros(wav_array.shape[0] / channels)
    for i in range(channels):
      float_array += wav_array[i::channels]
    float_array /= max(abs(float_array))
  finally:
    if wav:
      wav.close()

  return float_array

def WriteWav(filename, data, channels=1, sample_rate=44100):
  wav = None
  try:
    wav = wave.open(filename, 'w')
    wav.setnchannels(channels)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.setnframes(len(data) / channels)

    norm_data = (data * 32767 / (max(abs(data)))).astype(numpy.int16)

    wav.writeframes(norm_data.tostring())
  finally:
    if wav:
      wav.close()

def Sweep(length, start, end):
  samples = int(SAMPLE_RATE * length)

  fractions = numpy.arange(samples) / float(samples)
  scale = 2.0 * math.pi / SAMPLE_RATE
  return numpy.sin(scale * numpy.cumsum(start + (end - start)*fractions))

def Noise(length):
  samples = int(SAMPLE_RATE * length)
  return numpy.random.random(samples)

def LinearFadeOut(sound, length=None):
  if length:
    samples = int(min(len(sound), SAMPLE_RATE * length))
  else:
    samples = len(sound)

  gain = numpy.ones(len(sound))
  gain[-samples:] = 1.0 - (numpy.arange(samples).astype(numpy.float64) / samples)
  print gain[::1000]

  return sound * gain


def main(argv):
  WriteWav('bd.wav', LinearFadeOut(Sweep(0.25, 200, 60)))
  WriteWav('noise.wav', Noise(0.05))

if  __name__ == '__main__':
  sys.exit(main(sys.argv))
