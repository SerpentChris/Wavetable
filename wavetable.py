#   wavetable.py is a wavetable written in Python.
#   Copyright (C) 2022  Chris Calderon
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
import wave
import struct
from math import pi, sin, fmod, copysign
from typing import Sequence, Self, Callable
from functools import cache
from itertools import repeat
import random

FloatFunc = Callable[[float], float]
# The number of half-steps away from A, used
# to calculate frequences relative to A4 = 440 Hz
NOTE_TO_STEPS = {
    'B': 2,
    'A': 0,
    'G': -2,
    'F': -4,
    'E': -5,
    'D': -7,
    'C': -9,
}
A4_FREQUENCY = 440.0
TWO_PI = 2*pi


@cache
def note_to_frequency(note: str) -> float:
    '''
    Calculates the frequency of a given note.
    The note is specified in a string like "A4".
    Sharp or flat specified by a # or b are the end, e.g. "B4b"
    '''
    if note.endswith('b'):
        offset = -1
    elif note.endswith('#'):
        offset = +1
    else:
        offset = 0

    base_note, octave = note[0], note[1]
    steps = 12*(int(octave) - 4) + NOTE_TO_STEPS[base_note] + offset
    return A4_FREQUENCY*(2**(steps/12))


class WaveTable:
    def __init__(self, table: Sequence[float]):
        self.table = table

    @classmethod
    def from_func(cls, func: FloatFunc, period: float, size: int) -> Self:
        step = period/size
        table = [func(i*step) for i in range(size)]
        return cls(table)

    def sample(self, count: int, freq: float, freq_sample: float, volume: float) -> list[float]:
        table = self.table
        size = len(table)
        mod_mask = size -1
        i = 0
        result = [0.0]*count
        inc = freq*size/freq_sample
        for sample in range(count):
            x0i = int(i) % size
            x1i = (x0i + 1) % size
            result[sample] = volume*((i-x0i)*table[x1i] + fmod(x1i - i + size, size)*table[x0i])
            i = fmod(i + inc, size)
        return result


def demo():
    tempo = 120  # beats per minute
    track = [ # to do multiple tracks you can sum the samples of each track
        ('C4', 1),
        ('C4', 1),
        ('G4', 1),
        ('G4', 1),
        ('A4', 1),
        ('A4', 1),
        ('G4', 2),
        ('F4', 1),
        ('F4', 1),
        ('E4', 1),
        ('E4', 1),
        ('D4', 1),
        ('D4', 1),
        ('C4', 2),
        ('G4', 1),
        ('G4', 1),
        ('F4', 1),
        ('F4', 1),
        ('E4', 1),
        ('E4', 1),
        ('D4', 2),
        ('G4', 1),
        ('G4', 1),
        ('F4', 1),
        ('F4', 1),
        ('E4', 1),
        ('E4', 1),
        ('D4', 2),
        ('C4', 1),
        ('C4', 1),
        ('G4', 1),
        ('G4', 1),
        ('A4', 1),
        ('A4', 1),
        ('G4', 2),
        ('F4', 1),
        ('F4', 1),
        ('E4', 1),
        ('E4', 1),
        ('D4', 1),
        ('D4', 1),
        ('C4', 2),
    ]
    table_size = 100
    sin_table = WaveTable.from_func(sin, TWO_PI, table_size)
    square_table = WaveTable.from_func(lambda x: copysign(1.0, sin(x)), TWO_PI, table_size)

    sample_rate = 44100.0
    channels = 1
    depth = 2  # 16-bits per sample equals 2 bytes
    volume = 2000.0
    filename = 'twinkle-{}-table.wav'


    audible_beat_percent = 0.9  # add some silence at the end of a beat to hear articulation of each note
    sin_samples = []
    square_samples = []
    samples_per_beat = int(sample_rate * 60 / tempo)
    for note, beats in track:
        audible_samples = int(beats*samples_per_beat*audible_beat_percent)
        silent_samples = samples_per_beat - audible_samples
        f = note_to_frequency(note)

        sin_samples.extend(sin_table.sample(audible_samples, f, sample_rate, volume))
        sin_samples.extend(repeat(0.0, silent_samples))
        square_samples.extend(square_table.sample(audible_samples, f, sample_rate, volume))
        square_samples.extend(repeat(0.0, silent_samples))


    # adding some white noise to the samples
    noise_volume = 100.0
    random.seed("noice")
    for i in range(len(sin_samples)):
        noise = random.gauss(sigma=noise_volume/2)
        sin_samples[i] += noise
        square_samples[i] += noise

    for kind, samples in [('sin', sin_samples), ('square', square_samples)]:
        packed_samples = struct.pack('<%dh' % len(samples), *map(int, samples))
        wav_file = wave.open(filename.format(kind), 'wb')
        wav_file.setparams((channels, depth, sample_rate, len(samples), 'NONE', 'not compressed'))
        wav_file.writeframes(packed_samples)
        wav_file.close()


if __name__ == '__main__':
    demo()
