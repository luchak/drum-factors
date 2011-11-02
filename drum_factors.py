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
    sample_rate = wave.getframerate()

    wav_data = wav.readframes(wav.getnframes())

    if (sample_bytes == 2):
      wav_array = numpy.fromstring(wav_data, dtype=numpy.sint16)
    else:
      assert False
  finally:
    if wav:
      wav.close()

def main(argv):
  wav_array = array.array('h')
  for i in range(44100):
    wav_array.append(int(math.sin((330.0 + 220*(i/44100.0)) * 2.0 * math.pi * i / 44100.0) * 16000))

  output_filename = 'test.wav'
  wav = wave.open(output_filename, 'w')
  wav.setnchannels(CHANNELS)
  wav.setsampwidth(SAMPLE_BYTES)
  wav.setframerate(SAMPLE_RATE)

  wav.setnframes(44100)
  wav.writeframes(wav_array)
  wav.close()

if  __name__ == '__main__':
  sys.exit(main(sys.argv))
