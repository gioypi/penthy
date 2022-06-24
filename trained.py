"""Create a Convolutional Neural Network from file and activate it to analyze the given audio file.

Use an already trained Keras Convolutional Neural Network saved in the tf format to evaluate audio files.
This version will work inside the IDE and can be better for testing and evaluating files manually.
For the version that takes arguments and works on the website, look trained_web.py.
Only possible mp3 transcoding is inspected. Even if a file is verified as truly lossless in terms of mp3 transcoding,
it could still be upsampled, transcoded from a different format or altered in other ways.
"""

import audio_manipulation as am
from keras.models import load_model
from sys import exit


# The name of the file to evaluate.
# Good directory structure is important, this is an example with a relative path.
AUDIO_FILENAME = "temp_audio_files/the return - Anger management sessions.wav"

# Path to the file or directory with the saved model to load.
MODEL_PATH = "saved_models/trained_penthy"

# Evaluation threshold that discriminates the network's output into truly lossless and transcoded audio.
EVAL_THRESHOLD = 0.6


try:
    net = load_model(MODEL_PATH)
except IOError:
    print("Failed to read network from file.")
    exit(1)
else:
    print("Network created from file.")
# net.summary()

print("Extracting spectrograms...")
if not am.valid_metadata(AUDIO_FILENAME):
    print("--Fake flac--")
    exit(0)
spect_list = am.extract_spectrogram(AUDIO_FILENAME)
print("Spectrograms extracted.\n")

sum_out = 0
for i in spect_list:
    sum_out += net(i.reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0]      # net is not undefined.

avg_out = sum_out / len(spect_list)
print("Average output of file: ", avg_out)
if avg_out > EVAL_THRESHOLD:
    print("--True flac--")
else:
    print("--Fake flac--")

# Study the variance between outputs:
# print("\n\nRandom outputs from the file:")
# print(net(spect_list[1].reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0])
# print(net(spect_list[2].reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0])
# print(net(spect_list[3].reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0])
# print(net(spect_list[10].reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0])
# print(net(spect_list[18].reshape(1, am.HEIGHT, am.WIDTH, 3)).numpy()[0][0])
