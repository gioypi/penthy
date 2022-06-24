"""Create a Convolutional Neural Network from file and activate it to analyze audio files in a directory.

Use an already trained Keras Convolutional Neural Network saved in the tf format to evaluate audio files.
This version is intended for use in directories with multiple files for evaluation at the same time.
Create multiple processes, if possible. Do NOT look recursively into directories, only open files in the current level.
Only possible mp3 transcoding is inspected. Even if a file is verified as truly lossless in terms of mp3 transcoding,
it could still be upsampled, transcoded from a different format or altered in other ways.
"""

from logging import getLogger, FATAL
import os
import audio_manipulation as am
from keras.models import load_model
from sys import exit, argv
from multiprocessing import Pool
from time import monotonic
import colorama


class ANSIColors:
    """Color codes to customize the appearance of stdout in print statements.
    """

    RED = "\033[31m"
    GREEN = "\033[32m"
    END = "\033[m"     # Default setting.
    BOLD = '\033[1m'


def main(arg=None):
    # Evaluation threshold that discriminates the network's output into truly lossless and transcoded audio.
    EVAL_THRESHOLD = 0.6

    # Path to the file or directory with the saved model to load.
    MODEL_PATH = "saved_models/trained_penthy"

    # Path to the directory to scan.
    if arg is None:
        # dir_path = argv[1]        # Enable direct assignment or passing an argument. Supports Unicode.
        dir_path = "dataset_files/output_demo"
    else:
        dir_path = arg

    # Any file other than the permitted formats will be ignored.
    permitted_extensions = ".flac", ".wav"

    # Supress tensorflow messages.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Code number for FATAL
    getLogger('tensorflow').setLevel(FATAL)

    colorama.init()     # Make ANSI colors usable in Windows. Must be deinit-ed at the end of the program.
    print("\n" + ANSIColors.BOLD + "penthy" + ANSIColors.END + " will evaluate lossless audio files in", end=' ')
    print(dir_path, "for possible mp3 transcoding.")
    print("Supported formats:", permitted_extensions)
    start_time = monotonic()
    try:
        net = load_model(MODEL_PATH)
    except IOError:
        print("Failed to read Convolutional Neural Network from file.")
        exit(1)
    else:
        print("Convolutional Neural Network created from file.\n")
    # net.summary()     # Enable to check the network's structure.

    # Create a list of file paths to open.
    files = list()
    for file_name in os.listdir(dir_path):
        if file_name.endswith(permitted_extensions):
            file_path = os.path.join(dir_path, file_name)
            files.append(file_path)
    # print("Number of files to evaluate:", len(files))

    if len(files) == 0:
        print("No applicable files found.")
        exit(0)

    # Run an initial test for audio resolution etc.
    for track in files:
        if not am.valid_metadata(track):
            print(os.path.basename(track), "evaluated as", end=' ')
            print(ANSIColors.RED + "Transcoded" + ANSIColors.END, end=' ')
            print("(invalid specs)")
            files.remove(track)

    # Extract spectrogram segments.
    # If possible, create multiple processes.
    if (os.cpu_count() is not None) and (os.cpu_count() >= 4):
        num_workers = os.cpu_count() - 2
        if num_workers > 8:
            num_workers -= 2
        # print("Worker process num:", num_workers)     # Remember that there is one more process, the parent.
        pool = Pool(num_workers)
        chunk = int(len(files) / num_workers)       # Number of audio files to give to each worker.
        if chunk < 1:
            chunk = 1
        # print("Chunk size: ", chunk)
        spect_list_list = pool.map(am.extract_spectrogram, files, chunksize=chunk)
        pool.close()
        pool.join()
    else:
        spect_list_list = list()
        for track in files:
            spect_list_list.append(am.extract_spectrogram(track))

    # Activate the CNN with the spectrogram segments and print the result of the evaluation.
    for i in range(len(files)):
        spect_list = spect_list_list[i]
        sum_out = 0
        true_votes = 0
        transc_votes = 0
        for spect in spect_list:
            net_out = net(spect.reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0]  # net is defined and assigned.
            sum_out += net_out
            if net_out > EVAL_THRESHOLD:
                true_votes += 1
            else:
                transc_votes += 1
        avg_out = sum_out / len(spect_list)
        if avg_out > EVAL_THRESHOLD:      # Enable either an average of outputs or the voting system.
        # if true_votes > transc_votes:
            print(os.path.basename(files[i]), "evaluated as", end=' ')
            print(ANSIColors.GREEN + "Truly Lossless" + ANSIColors.END, end=' ')
        else:
            print(os.path.basename(files[i]), "evaluated as", end=' ')
            print(ANSIColors.RED + "Transcoded" + ANSIColors.END, end=' ')
        print("(average CNN's output: {:.2f}%)".format(avg_out * 100))
        # print("(votes true/transcoded:", true_votes, "/", transc_votes, ")")

    end_time = monotonic()      # Real time passed, not time in CPU/GPU.
    print("\nEvaluation complete. Duration: %.3f minutes" % ((end_time - start_time) / 60))
    print("Note that even if a file is verified as truly lossless in terms of mp3 transcoding,")
    print("it could still be transcoded from a different format, upsampled, or altered in other ways.")
    colorama.deinit()


if __name__ == "__main__":
    main()
