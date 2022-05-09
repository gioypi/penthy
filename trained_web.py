"""Create a Neural Network from file and activate it to analyze the given audio file and delete it afterwards.

Use an already trained Keras Convolutional Neural Network saved in the tf format to evaluate audio files.
Print code-word trueflac or fakeflac in stdout, to be parsed by PHP.
This module is designed to work on the website and expects the filename as an argument.
The given audio file will be DELETED after evaluation! This is useful for the webserver, where the file is uploaded.
If you want to evaluate files manually inside the IDE, you may prefer trained.py.
Only possible mp3 transcoding is inspected. Even if a file is verified as truly lossless in terms of mp3 transcoding,
it could still be upsampled, transcoded from a different format or altered in other ways.
"""

import os
from logging import getLogger, FATAL
# Supress tensorflow messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Code number for FATAL
getLogger('tensorflow').setLevel(FATAL)

import audio_manipulation as am
from keras.models import load_model
from sys import exit, argv


# The name of the file to evaluate. Expects a name without whitespace.
audio_filename = argv[1]

# Path to the file or directory with the saved model to load.
MODEL_PATH = "trained_penthy"

try:
    net = load_model(MODEL_PATH)
except IOError:
    # print("Failed to read network from file.")
    exit(-1)
else:
    # print("Network created from file.")
    pass

# print("Extracting spectograms...")
if not am.valid_metadata(audio_filename):
    print("fakeflac")
    # Delete the file from the server.
    if os.path.exists(audio_filename):
        os.remove(audio_filename)
    exit(0)
spect_list = am.extract_spectogram(audio_filename)
# print("Spectograms extracted.\n")

# Delete the file from the server.
if os.path.exists(audio_filename):
    os.remove(audio_filename)

sum_out = 0
for i in spect_list:
    sum_out += net(i.reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0]      # net is not undefined.
avg_out = sum_out / len(spect_list)

# print("Average output of file: ", avg_out)
if avg_out > 0.6:
    print("trueflac")       # PHP expects an absolute equality of the possible code-words in stdout.
else:
    print("fakeflac")
