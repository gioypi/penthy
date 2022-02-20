"""Create a Neural Network from file and activate it to analyze the given file.

Use an already trained Pybrain Neural Network saved as an xml file to analyze audio files.
This version will work inside the IDE and can be better for testing and evaluating files manually.
For the version that takes arguments and works on the website, look trained_web.py.
"""

from pybrain3.tools.xml.networkreader import NetworkReader
import audio_manipulation
import net_manipulation
import sys


# The name of the file with the saved Neural Network. Use a path if not in the same directory.
NET_FILENAME = "trained_penthy.xml"

# The name of the file to evaluate.
# Good directory structure is important, this is an example with a relative path.
AUDIO_FILENAME = "dataset_files/validation_wav/example.wav"

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
    sys.exit(0)
# feat_list = audio_manipulation.extract_features(AUDIO_FILENAME)   # For networks that utilize audio features instead.
feat_list = audio_manipulation.extract_spectogram(AUDIO_FILENAME)
print("Features extracted.\n")

# Activate for all time segments of the file and calculate the average output for increased accuracy.
out_sum = 0.0
i = 0
while i < len(feat_list):
    j = 0
    while j < len(feat_list[i]):
        # Customize the normalization bounds per dataset.
        feat_list[i][j] = net_manipulation.normalize(feat_list[i][j], 0.0, 0.0194) * 10
        # feat_list[i][j] = net_manipulation.normalize(feat_list[i][j], 0.0, 0.194) * 10        # For older models.
        j += 1
    out_sum += n.activate(feat_list[i])
    i += 1
out_avg = out_sum / len(feat_list)

# Example of normalizing output. Use when stuck in small output range.
# out_avg = net_manipulation.normalize(out_avg, 0.34, 0.37)

print("Average output: ", out_avg * 100, "%")
if out_avg > 0.55:
    print("--True flac--")
else:
    print("--Fake flac--")

# Study the variance between outputs:
# print("\nRandom raw outputs: ")
# print(n.activate(feat_list[1]))
# print(n.activate(feat_list[2]))
# print(n.activate(feat_list[3]))
# print(n.activate(feat_list[50]))
# print(n.activate(feat_list[200]))
# print("All zeros: ", n.activate([0]*91))
# print("All high: ", n.activate([0.01]*91))


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
# print("Output min: ", min_feat)
# print("Output max: ", max_feat)
