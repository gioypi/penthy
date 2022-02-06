"""Open and manipulate audio files to use as a dataset.

Functions:
    convert_to_wav(file_path): Convert an audio file to wav using ffmpeg.
    valid_metadata(file_path): Check a file for appropriate sample frequency, bit-depth and audio channels.
    extract_features(file_path): Open the audio file of the given path and extract and return its features.
    extract_features_from_dir(dir_path): Open all audio files of the given directory and return their extracted
    features.
    extract_spectogram(file_path): Open the audio file of the given path and extract the value of some frequencies.
"""

from pyAudioAnalysis import audioBasicIO as aBIO
from pyAudioAnalysis import ShortTermFeatures as sTF
import os
import subprocess
import ffmpeg
import numpy as np


def convert_to_wav(file_path):
    """Create a wav file from a flac file and save it in the same path.

    Parameters:
        file_path (string): The path to the file to be converted.
    Returns:
        new_name (string): The path to the newly created wav file.
    """

    permitted_extensions = ".flac"
    new_name = None
    if file_path.endswith(permitted_extensions):
        file_name, old_ext = os.path.splitext(file_path)
        signal = ffmpeg.input(file_path)
        new_name = file_name + ".wav"
        ffmpeg.output(signal, new_name).run()
    return new_name


def valid_metadata(file_path):
    """Run an initial check of an audio file for acceptable sample frequency, bit-depth and number of channels.

    Parameters:
        file_path (string): The path to the file to be checked.
    Returns:
        (boolean): True means valid metadata, False means not true flac.
    """

    ffprobe_command = 'ffprobe "' + file_path + \
                      '" -show_entries stream=sample_rate -select_streams a -of compact=p=0:nk=1 -v 0'
    sample_rate = int(subprocess.check_output(ffprobe_command, universal_newlines=True).strip())
    ffprobe_command = 'ffprobe "' + file_path + \
                      '" -show_entries stream=bits_per_sample -select_streams a -of compact=p=0:nk=1 -v 0'
    bits_per_sample = int(subprocess.check_output(ffprobe_command, universal_newlines=True).strip())
    ffprobe_command = 'ffprobe "' + file_path + \
                      '" -show_entries stream=channels -select_streams a -of compact=p=0:nk=1 -v 0'
    channels = int(subprocess.check_output(ffprobe_command, universal_newlines=True).strip())
    if sample_rate < 44100:
        return False
    elif bits_per_sample < 16:
        return False
    elif channels < 2:
        return False
    else:
        return True


def extract_features(file_path):
    """Open an audio file and return its audio features.

    Optimized for easy modifications, NOT for speed.

    Parameters:
        file_path (string): The path to the file to be opened.
    Returns:
        feat_list (2D float list): All extracted features for several time fragments.
    """

    [sample_freq, signal] = aBIO.read_audio_file(file_path)
    signal = aBIO.stereo_to_mono(signal)
    features, f_names = sTF.feature_extraction(signal, sample_freq, 0.1*sample_freq, 0.1*sample_freq)
    features = np.delete(features, slice(3), 0)
    features = np.delete(features, slice(2, 65), 0)
    feat_list = features.transpose().tolist()   # features is an ndarray.
    return feat_list


def extract_features_from_dir(dir_path):
    """Open all audio files in a directory and return their audio features.

    Do NOT look recursively into directories, only open files in the current level.
    Expect a very large list as output. Directories with many files will only work with 64bit Python.

    Parameters:
        dir_path (string): The path to the directory containing the audio files to be opened.
    Returns:
        feat_list (2D float list): All extracted features for several time fragments for all files.
    """

    permitted_extensions = ".wav"
    feat_list = list(list())
    for file_name in os.listdir(dir_path):
        if file_name.endswith(permitted_extensions):
            file_path = os.path.join(dir_path, file_name)
            feat_list.extend(extract_features(file_path))
    return feat_list


def extract_spectogram(file_path):
    """Compute the spectogram of the given file.
    TODO: Review doc
    Parameters:
        file_path (string): The path to the file to be opened.
    Returns:
        spect_list (2D float list): Values for frequency ranges for the duration of the audio.
    """

    [sample_freq, signal] = aBIO.read_audio_file(file_path)
    signal = aBIO.stereo_to_mono(signal)
    spect, time_axis, freq_axis = sTF.spectrogram(signal, sample_freq, 0.03*sample_freq, 0.09*sample_freq, plot=False)
    spect = np.delete(spect, slice(570), 1)     # Delete low frequencies (first 19kHz) and leave high frequencies.
    spect_list = spect.tolist()    # spect is an ndarray.
    # print("Shape after cut: ", spect.shape)
    # print(len(spect_list[0]))
    # print(len(spect_list[10]))
    # print(len(spect_list[200]))
    return spect_list
