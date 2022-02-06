"""Create a Neural Network from file and activate it to analyze the given file.
TODO: Review doc
Use an already trained Neural Network saved as an xml file to analyze audio files.
"""

from pybrain3.tools.xml.networkreader import NetworkReader
import audio_manipulation
import net_manipulation
import sys


# The name of the file with the saved Neural Network.
NET_FILENAME = "trained_penthy.xml"

# The name of the file to evaluate.
AUDIO_FILENAME = "dataset_files/validation_wav/fake-Natural - Imagine Dragons.wav"

try:
    n = NetworkReader.readFrom(NET_FILENAME)
except OSError:
    print("Failed to read network from file.")
    sys.exit(1)
else:
    print("Network created from file.")

print("Extracting audio features...")
if not audio_manipulation.valid_metadata(AUDIO_FILENAME):
    print("--Fake flac--")
    exit(0)
# feat_list = audio_manipulation.extract_features(AUDIO_FILENAME)
feat_list = audio_manipulation.extract_spectogram(AUDIO_FILENAME)
print("Features extracted.\n")
out_sum = 0.0
i = 0
while i < len(feat_list):
    j = 0
    while j < len(feat_list[i]):
        feat_list[i][j] = net_manipulation.normalize(feat_list[i][j], 0.0, 0.194) * 10
        # feat_list[i][j] = feat_list[i][j] * 100
        j += 1
    out_sum += n.activate(feat_list[i])
    i += 1
out_avg = out_sum / len(feat_list)
# out_avg = net_manipulation.normalize(out_avg, 0.3786, 0.3788)
print("Average normalized output: ", out_avg * 100, "%")
# print(type(out_avg))
if out_avg > 0.5:
    print("--True flac--")
else:
    print("--Fake flac--")
print("\nRandom raw outputs: ")
print(n.activate(feat_list[1]))
print(n.activate(feat_list[2]))
print(n.activate(feat_list[3]))
print(n.activate(feat_list[50]))
print(n.activate(feat_list[200]))
print("all zeros: ", n.activate([0]*91))
print("all high: ", n.activate([0.01]*91))


# Find bounds for the normalization of the output.
# i = 0
# min_feat = 1.1
# max_feat = -1.1
# print("Searching for output bounds...")
# while i < len(feat_list):
#     current = n.activate(feat_list[i])
#     if current < min_feat:
#         min_feat = current
#     if current > max_feat:
#         max_feat = current
#     i += 1
# print("Activation bounds:")
# print("min: ", min_feat)
# print("max: ", max_feat)
