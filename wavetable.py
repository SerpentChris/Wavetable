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
from __future__ import annotations
import wave
import struct
from math import pi, sin, fmod, copysign
from typing import Sequence, Callable
from functools import cache
from itertools import repeat
import random
import tkinter
from tkinter import ttk
from tkinter.filedialog import asksaveasfile
from tkinter.messagebox import showerror
import scipy
from scipy.interpolate import CubicSpline
import numpy

FloatFunc = Callable[[float], float]
# The number of half-steps away from A, used
# to calculate frequencies relative to A4 = 440 Hz
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
    """
    Calculates the frequency of a given note.
    The note is specified in a string like "A4".
    Sharp or flat specified by a # or b are the end, e.g. "B4b"
    """
    if note.endswith('b'):
        offset = -1
    elif note.endswith('#'):
        offset = +1
    else:
        offset = 0

    base_note, octave = note[0], note[1]
    steps = 12*(int(octave) - 4) + NOTE_TO_STEPS[base_note] + offset
    return A4_FREQUENCY*(2**(steps/12))


def draw_waveform():
    root = tkinter.Tk()
    screen_height = root.winfo_screenheight()
    screen_width = root.winfo_screenwidth()
    width = 520
    height = 313
    root.title("Draw a Waveform")
    root.geometry(
        f"{width}x{height}"
        f"+{(screen_width - width)//2}"
        f"+{(screen_height - height)//2}"
    )

    slider_frame = ttk.Frame(root, padding=5)
    input_frame = ttk.Frame(root, padding=5)
    button_frame = ttk.Frame(root, padding=5)
    scales = [ttk.Scale(slider_frame, from_=-1.0, to=1.0, orient='vertical', length=200) for _ in range(20)]
    table_size_lbl = ttk.Label(input_frame, text="Table Size: ")
    table_size_entry = ttk.Entry(input_frame)
    save_btn = ttk.Button(button_frame, text="Save")
    clear_btn = ttk.Button(button_frame, text="Reset")

    def save_cmd():
        print(root.geometry())
        try:
            table_size = int(table_size_entry.get().strip())
        except:
            showerror(title="Error", message="Please enter a valid table size.")
            return

        output = asksaveasfile('wb', parent=root, defaultextension='.wtbl')
        if output is None:
            return

        xs = list(range(len(scales)))
        ys = [s.get() for s in scales]
        f = CubicSpline(xs, ys)
        new_xs = numpy.linspace(0.0, 1.0, table_size)
        new_ys = f(new_xs)
        new_ys.tofile(output)
        return

    def clear_cmd():
        for s in scales:
            s.set(0.0)
        root.update_idletasks()

    save_btn['command'] = save_cmd
    clear_btn['command'] = clear_cmd

    slider_frame.pack()
    input_frame.pack()
    button_frame.pack()
    for s in scales:
        s.pack(side='left')
    table_size_lbl.pack(side='left')
    table_size_entry.pack(side='left')
    save_btn.pack()
    clear_btn.pack()
    root.mainloop()
    return


class WaveTable:
    def __init__(self, table: Sequence[float]):
        self.table = table

    @classmethod
    def from_func(cls, func: FloatFunc, period: float, size: int) -> WaveTable:
        step = period/size
        table = [func(i*step) for i in range(size)]
        return cls(table)

    @classmethod
    def from_file(cls, filename: str) -> WaveTable:
        vals = numpy.fromfile(filename, dtype=numpy.float64)
        # normalize the wave form so the max and min are 1 and -1
        min_val = min(vals)
        max_val = max(vals)
        vals *= 2.0 / (max_val - min_val)
        new_min = min(vals)
        new_max = max(vals)
        if new_min < -1.0:
            vals += -1.0 - new_min
        elif new_max > 1.0:
            vals -= (max_val - 1.0)
        return cls(vals)

    def sample(self, count: int, freq: float, freq_sample: float, volume: float) -> list[float]:
        table = self.table
        size = len(table)
        i = 0
        result = [0.0]*count
        inc = freq*size/freq_sample
        for sample in range(count):
            x0i = int(i)
            x1i = x0i + 1
            result[sample] = volume*((i-x0i)*table[x1i % size] + (x1i - i)*table[x0i])
            i = fmod(i + inc, size)
        return result


def demo():
    tempo = 120  # beats per minute
    track = [  # to do multiple tracks you can sum the samples of each track
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
    custom_table = WaveTable.from_file('wave1.wtbl')
    sample_rate = 44100.0
    channels = 1
    depth = 2  # 16-bits per sample equals 2 bytes
    volume = 2000.0
    filename = 'twinkle-{}-table.wav'

    audible_beat_percent = 0.9  # add some silence at the end of a beat to hear articulation of each note
    sin_samples = []
    square_samples = []
    custom_samples = []
    samples_per_beat = int(sample_rate * 60 / tempo)
    for note, beats in track:
        audible_samples = int(beats*samples_per_beat*audible_beat_percent)
        silent_samples = samples_per_beat - audible_samples
        f = note_to_frequency(note)

        sin_samples.extend(sin_table.sample(audible_samples, f, sample_rate, volume))
        sin_samples.extend(repeat(0.0, silent_samples))
        square_samples.extend(square_table.sample(audible_samples, f, sample_rate, volume))
        square_samples.extend(repeat(0.0, silent_samples))
        custom_samples.extend(custom_table.sample(audible_samples, f, sample_rate, volume))
        custom_samples.extend(repeat(0.0, silent_samples))

    # adding some white noise to the samples
    noise_volume = 100.0
    random.seed("noice")
    for i in range(len(sin_samples)):
        noise = random.gauss(sigma=noise_volume/2, mu=0.0)
        sin_samples[i] += noise
        square_samples[i] += noise
        custom_samples[i] += noise

    for kind, samples in [('sin', sin_samples), ('square', square_samples), ('custom', custom_samples)]:
        packed_samples = struct.pack('<%dh' % len(samples), *map(int, samples))
        wav_file = wave.open(filename.format(kind), 'wb')
        wav_file.setparams((channels, depth, sample_rate, len(samples), 'NONE', 'not compressed'))
        wav_file.writeframes(packed_samples)
        wav_file.close()


if __name__ == '__main__':
    demo()
    # draw_waveform()
