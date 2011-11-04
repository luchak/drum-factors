#!/usr/bin/env python

import math
import sys
import wave

import cvxopt.solvers
import matplotlib.pyplot
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
  return 2.0*numpy.random.random(samples) - 1.0

def DifferenceFilter(sound):
  return numpy.convolve(numpy.array((1.0, -1.0)), sound)

def SumFilter(sound):
  return numpy.convolve(numpy.array((1.0, 1.0)), sound)

def LinearFadeOut(sound, length=None):
  if length:
    samples = int(min(len(sound), SAMPLE_RATE * length))
  else:
    samples = len(sound)

  gain = numpy.ones(len(sound))
  gain[-samples:] = 1.0 - (numpy.arange(samples).astype(numpy.float64) / samples)

  return sound * gain

def SumWithLength(a, b, length):
  result = numpy.zeros(length)
  a_copy_length = min(len(a), length)
  result[:a_copy_length] += a[:a_copy_length]
  b_copy_length = min(len(b), length)
  result[:b_copy_length] += b[:b_copy_length]
  return result

def Normalize(sound):
  return sound / max(abs(sound))

def SumUnequal(a, b):
  new_length = max(len(a), len(b))
  return SumWithLength(a, b, new_length)

def FFTMagnitudes(data):
  return abs(numpy.fft.rfft(data))

def PlotFFT(data):
  matplotlib.pyplot.plot(FFTMagnitudes(data))
  matplotlib.pyplot.xscale('log')
  matplotlib.pyplot.yscale('log')
  matplotlib.pyplot.show()

def main(argv):
  bd = 2.0*LinearFadeOut(Sweep(0.15, 140, 50))
  hh1 = Noise(0.05)
  hh2 = 2.0*DifferenceFilter(hh1)
  hh3 = 2.0*LinearFadeOut(DifferenceFilter(DifferenceFilter(Noise(0.2))))
  sn = SumUnequal(LinearFadeOut(SumFilter(Noise(0.1))), 3.0*LinearFadeOut(Sweep(0.08, 200, 130)))

  num_beats = 16
  if len(argv) > 2:
    num_beats = int(argv[2])

  loop = ReadWav(argv[1])
  beat_length = len(loop)/num_beats
  loop_beat_ffts = [FFTMagnitudes(loop[i*beat_length:(i+1)*beat_length]) for i in range(num_beats)]
  fft_length = len(loop_beat_ffts[0])

  samples = [bd, hh1, hh2, hh3, sn]
  sample_ffts = []
  for sample in samples:
    padded_sample = numpy.zeros(beat_length)
    copy_length = min(beat_length, len(sample))
    padded_sample[:copy_length] = sample[:copy_length]
    sample_ffts.append(FFTMagnitudes(padded_sample))

  target = numpy.array(loop_beat_ffts).flatten()
  sample_matrix_rows = []
  for fft in sample_ffts:
    for i in range(num_beats):
      padded_row = numpy.zeros(num_beats * fft_length)
      padded_row[i*fft_length:(i+1)*fft_length] = fft
      sample_matrix_rows.append(padded_row)
  sample_matrix = numpy.array(sample_matrix_rows).T

  gamma = 200000.0
  #sequence = numpy.linalg.lstsq(sample_matrix, target)[0]
  sequence_items = sample_matrix.shape[1]
  zero_matrix = cvxopt.spmatrix([], [], [], size=(sequence_items, sequence_items))
  quadratic_objective = cvxopt.sparse([[cvxopt.matrix(numpy.dot(sample_matrix.T, sample_matrix)), zero_matrix], [zero_matrix, zero_matrix]])
  linear_objective = cvxopt.matrix([[cvxopt.matrix(-numpy.dot(sample_matrix.T, target)), cvxopt.matrix(gamma * numpy.ones(sequence_items))]])
  sequence_eye = cvxopt.spmatrix(1.0, range(sequence_items), range(sequence_items))
  l1_norm_matrix = cvxopt.sparse([[sequence_eye, -sequence_eye, -sequence_eye], [-sequence_eye, -sequence_eye, sequence_eye*0]])
  l1_norm_target = cvxopt.matrix(cvxopt.matrix(numpy.zeros(3*sequence_items)))
  sequence = cvxopt.solvers.coneqp(quadratic_objective, linear_objective, l1_norm_matrix, l1_norm_target)['x']
  sequence = numpy.array(sequence)[:sequence_items]
  sequence = sequence.reshape((len(sequence)/num_beats, num_beats))

  print sequence
  binary_sequence = sequence.copy()
  binary_sequence[sequence < 1e-4] = 0.0
  binary_sequence[binary_sequence > 0.0] = 1.0
  print binary_sequence

  loop_out = numpy.zeros(num_beats * beat_length)
  for i, sample in enumerate(samples):
    for j in range(num_beats):
      beat_copy_start = j*beat_length
      beat_copy_length = min(len(sample), len(loop_out) - beat_copy_start)
      beat_copy_end = beat_copy_start + beat_copy_length
      loop_out[beat_copy_start:beat_copy_end] += sequence[i,j] * sample[:beat_copy_length]
  WriteWav('loop_out.wav', loop_out)


if  __name__ == '__main__':
  sys.exit(main(sys.argv))
