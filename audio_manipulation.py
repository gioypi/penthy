"""Open and manipulate audio files to use as a dataset.

Functions:
    convert_to_wav(file_path, verbose): Convert an audio file to wav using ffmpeg.
    valid_metadata(file_path): Check a file for appropriate extension, sample frequency and audio channels.
    extract_spectrogram(file_path, verbose): Create spectrogram images for time segments of an audio file.
    extract_spectrogram_from_dir(dir_path, verbose, multiprocess): Create spectrogram images for time segments of
    audio files within a directory.
"""

import ffmpeg
import subprocess
import os
from sys import exit
import numpy as np
from multiprocessing import Pool


# Time duration of each spectrogram segment in seconds.
SPECT_DURATION = 8

# Resolution of each spectrogram segment in pixels.
HEIGHT = 128
WIDTH = HEIGHT


def convert_to_wav(file_path, verbose=False):
    """Create a wav file from a flac file and save it in the same path.

    Parameters:
        file_path (string): The path to the file to be converted.
        verbose (boolean): When True, print the ffmpeg output.
    Returns:
        new_name (string): The path to the newly created wav file. Equals None if no conversion took place.
    """

    permitted_extensions = ".flac"
    new_name = None
    if file_path.endswith(permitted_extensions):
        file_name, old_ext = os.path.splitext(file_path)
        signal = ffmpeg.input(file_path)
        new_name = file_name + ".wav"

        # ffmpeg will print by default, when verbose==False, supress its output.
        if not verbose:
            ffmpeg.output(signal, new_name).global_args('-loglevel', 'quiet').run()
        else:
            ffmpeg.output(signal, new_name).run()
    return new_name


def valid_metadata(file_path):
    """Run an initial check on an audio file for acceptable sample frequency and number of channels.

    Also check if the file extension is one of the supported lossless formats.

    Parameters:
        file_path (string): The path to the file to be checked.
    Returns:
        (boolean): True means valid metadata, False means not true flac.
    """

    permitted_extensions = ".flac", ".wav"
    if not(file_path.endswith(permitted_extensions)):
        return False

    ffprobe_command = ['ffprobe', file_path, '-show_entries', 'stream=sample_rate',
                       '-select_streams', 'a', '-of', 'compact=p=0:nk=1', '-v', '0']
    try:
        sample_rate = int(subprocess.check_output(ffprobe_command, universal_newlines=True).strip())
    except subprocess.CalledProcessError as err:
        print("Error using ffprobe:", err.output, err.returncode)
        exit(-1)

    ffprobe_command_2 = ['ffprobe', file_path, '-show_entries', 'stream=channels',
                         '-select_streams', 'a', '-of', 'compact=p=0:nk=1', '-v', '0']
    try:
        channels = int(subprocess.check_output(ffprobe_command_2, universal_newlines=True).strip())
    except subprocess.CalledProcessError as err:
        print("Error using ffprobe:", err.output, err.returncode)
        exit(-1)

    if sample_rate < 44100:
        return False
    elif channels < 2:
        return False
    else:
        return True


def extract_spectrogram(file_path, verbose=False):
    """Compute spectrograms for overlapping time fragments of an audio file, excluding low frequencies.

    Parameters:
        file_path (string): The path to the file to be opened.
        verbose (boolean): When True, print the ffmpeg output and other information.
    Returns:
        spect_list (list): List of images of spectrogram segments in RGB. Each element is a 3D ndarray.
    """

    signal = ffmpeg.input(file_path)

    # Find the sample rate of the file.
    ffprobe_command = ['ffprobe', file_path, '-show_entries', 'stream=sample_rate', '-select_streams', 'a', '-of',
                       'compact=p=0:nk=1', '-v', '0']
    try:
        sample_rate = int(subprocess.check_output(ffprobe_command, universal_newlines=True).strip())
    except subprocess.CalledProcessError as err:
        print("Error using ffprobe:", err.output, err.returncode)
        exit(-1)

    # print("sample frequency:", sample_rate)

    # Find the duration of the file in seconds.
    ffprobe_command_2 = ['ffprobe', file_path, '-show_entries', 'stream=duration', '-select_streams', 'a', '-of',
                         'compact=p=0:nk=1', '-v', '0']
    try:
        duration = float(subprocess.check_output(ffprobe_command_2, universal_newlines=True).strip())
    except subprocess.CalledProcessError as err:
        print("Error using ffprobe:", err.output, err.returncode)
        exit(-1)
    # print("duration:", duration)

    # Support very small files by looping them until they cover one spectrogram segment.
    if duration < SPECT_DURATION:
        loops = int(SPECT_DURATION / duration)      # aloop will output the audio 1 time more than loops
        signal = ffmpeg.filter(signal, "aloop", loop=loops, size=sample_rate*duration*(loops+1))
        # Enable exporting a file with the new looped audio, for debugging:
        # ffmpeg.output(signal, "temp_audio_files/test_loop.wav").global_args('-loglevel', 'quiet').run()
        duration = duration * (loops + 1)
        # print("new duration:", duration)
        if verbose:
            print("File duration shorter than segment size, audio looped.")

    spect_list = list()         # List of ndarrays, about to contain the spectrogram segments.
    sample_start = 0
    sample_end = sample_rate * SPECT_DURATION
    step = int((sample_rate * SPECT_DURATION) / 2)       # 50% overlap
    sample_total = sample_rate * duration
    while sample_end <= int(sample_total / step) * step:
        signal_seg = ffmpeg.filter(signal, "atrim", start_pts=sample_start, end_pts=sample_end)
        signal_seg = ffmpeg.filter(signal_seg, "showspectrumpic", s=str(WIDTH)+"x"+str(HEIGHT),
                                   legend="disabled", start="16200", stop="22000")

        # ffmpeg will print by default, when verbose==False, supress its output.
        if not verbose:
            # signal_seg = ffmpeg.output(signal_seg, "temp_audio_files/test_spect" + str(sample_end) + ".png")\
            #     .global_args('-loglevel', 'quiet').run()
            signal_seg = ffmpeg.output(signal_seg, "pipe:", format="rawvideo", pix_fmt="rgb24")\
                    .global_args('-loglevel', 'quiet').compile()
        else:
            # signal_seg = ffmpeg.output(signal_seg, "temp_audio_files/test_spect" + str(sample_end) + ".png").run()
            signal_seg = ffmpeg.output(signal_seg, "pipe:", format="rawvideo", pix_fmt="rgb24").compile()

        # Capture the output from ffmpeg and save it to an ndarray.
        pipe = subprocess.run(signal_seg, stdout=subprocess.PIPE)
        spect_seg = np.frombuffer(buffer=pipe.stdout, dtype=np.uint8).reshape([WIDTH, HEIGHT, 3])
        spect_list.append(spect_seg)

        sample_start += step
        sample_end += step

    if verbose:
        print("Spectrogram list extracted from", file_path)
        print("List dimensions: ", len(spect_list), len(spect_list[0]), len(spect_list[0][0]), len(spect_list[0][0][0]))
    return spect_list


def extract_spectrogram_from_dir(dir_path, verbose=False, multiprocess=True):
    """Compute spectrograms for all appropriate files within a directory, using extract_spectrogram().

    Do NOT look recursively into directories, only open files in the current level.
    If multiprocessing is enabled, multiple processes will start and the load of spectrograms will be split among them.
    The number of processes will be calculated automatically, based on available CPU cores.
    An 'if __name__ == "__main__"' guard must be used in the module that calls this function with multiprocess enabled.

    Parameters:
        dir_path (string): The path to the directory containing the audio files to be opened.
        verbose (boolean): When True, print information about the extracted data.
        multiprocess (boolean): When True, create multiple processes to extract spectrograms faster.
    Returns:
        spect_list (list): List of RGB images of spectrogram segments from all files. Each element is a 3D ndarray.
    """

    permitted_extensions = ".wav", ".flac"
    spect_list = list()

    # If possible, create multiple processes.
    if multiprocess and (os.cpu_count() is not None) and (os.cpu_count() >= 4):
        num_workers = os.cpu_count() - 2
        if num_workers > 8:
            num_workers -= 2
        # print("Worker process num:", num_workers)
        pool = Pool(num_workers)
        files = list()      # List of file paths to open.
        for file_name in os.listdir(dir_path):
            if file_name.endswith(permitted_extensions):
                file_path = os.path.join(dir_path, file_name)
                files.append(file_path)
        temp_list = pool.imap_unordered(extract_spectrogram, files)
        pool.close()
        pool.join()

        # Convert list of lists to flattened list.
        for i in temp_list:
            spect_list.extend(i)

    else:
        for file_name in os.listdir(dir_path):
            if file_name.endswith(permitted_extensions):
                file_path = os.path.join(dir_path, file_name)
                spect_list.extend(extract_spectrogram(file_path))

    if verbose:
        print("\nSpectrogram list extracted from directory", dir_path)
        print("List dimensions: ", len(spect_list), len(spect_list[0]), len(spect_list[0][0]), len(spect_list[0][0][0]))
    return spect_list
