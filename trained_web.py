"""Create a Neural Network from file and activate it to analyze the given file.

Use an already trained Neural Network saved as an xml file to analyze audio files.
Print code-word trueflac or fakeflac in stdout, to be parsed by PHP.
This module is designed to work on the website and expects the filename as an argument.
If you want to evaluate files manually inside the IDE, you may prefer trained.py.
"""

from pybrain3.tools.xml.networkreader import NetworkReader
import audio_manipulation
import net_manipulation
import sys
import os

# The name of the file with the saved Neural Network.
NET_FILENAME = "trained_penthy.xml"

# The name of the file to evaluate. Expects a name without whitespace.
audio_filename = sys.argv[1]

try:
    n = NetworkReader.readFrom(NET_FILENAME)
except OSError:
    # print("Failed to read network from file.")
    sys.exit(-1)
# else:
#     print("Network created from file.")

# print("Extracting audio features...")
if not audio_manipulation.valid_metadata(audio_filename):
    if os.path.exists(audio_filename):
        os.remove(audio_filename)
    print("fakeflac")
    sys.exit(0)

# Convert flac to wav.
if audio_filename.endswith(".flac"):
    # new_filename = audio_manipulation.convert_to_wav(audio_filename)      # Will fail outside the IDE.
    name, old_ext = os.path.splitext(audio_filename)
    new_filename = name + ".wav"
    os.system("ffmpeg -loglevel quiet -i " + audio_filename + " -ar 44100 " + new_filename)
    if os.path.exists(audio_filename):
        os.remove(audio_filename)
    audio_filename = new_filename

# feat_list = audio_manipulation.extract_features(audio_filename)   # For networks that utilize audio features instead.
feat_list = audio_manipulation.extract_spectogram(audio_filename)
if os.path.exists(audio_filename):
    os.remove(audio_filename)
# print("Features extracted.\n")
out_sum = 0.0
i = 0
while i < len(feat_list):
    j = 0
    while j < len(feat_list[i]):
        # Enable the normalization that is identical to the one during the training of the network.
        feat_list[i][j] = net_manipulation.normalize(feat_list[i][j], 0.0, 0.0194) * 10
        # For networks with more boosted frequency features:
        # feat_list[i][j] = net_manipulation.normalize(feat_list[i][j], 0.0, 0.194) * 10
        j += 1
    out_sum += n.activate(feat_list[i])
    i += 1
out_avg = out_sum / len(feat_list)

# print("Average output: ", out_avg * 100, "%")

if out_avg > 0.55:
    print("trueflac")
else:
    print("fakeflac")
