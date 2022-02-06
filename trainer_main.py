"""Call functions from other modules to extract audio features and experiment.
TODO: Review doc
"""

import audio_manipulation
import net_manipulation
from pybrain3.datasets import SupervisedDataSet


# feat_list = audio_manipulation.extract_features("dataset_files/test_wav/true-Dead Inside - Muse.wav")
# print("~ in main now: ", type(feat_list))
# print("length of list: ", len(feat_list))
# print("length of list[i]: ", len(feat_list[0]))
# print(feat_list)

# large_feat_list = audio_manipulation.extract_features_from_dir("dataset_files/wav_of_transc_flac")
# print("~ in main now: ", type(large_feat_list))
# print("length of list: ", len(large_feat_list))
# print("length of list[i]: ", len(large_feat_list[0]))

# new_name = audio_manipulation.convert_to_wav("temp_audio_files/A Peculiar Passing - Peter Gundry.flac")
# print(new_name)
#
# print(audio_manipulation.valid_metadata("dataset_files/test_wav/true-Dead Inside - Muse.wav"))

# Test shuffle:
# in_ds = SupervisedDataSet(1, 1)
# in_ds.addSample(0.5, 1)
# in_ds.addSample(0.1, 0.2)
# in_ds.addSample(0.02, 0.04)
# in_ds.addSample(0.0001, 0.0002)
# in_ds = net_manipulation.shuffle_ds(in_ds)
# for i, t in in_ds:
#     print(i, " -> ", t)

# Test spectogram:
feat_list = audio_manipulation.extract_spectogram("dataset_files/validation_wav/true-Always - Bon Jovi.wav")
print("~ in main now: ", type(feat_list))
num_zeros = 0
num_elements = 0
num_medium = 0
num_high = 0
num_mid = 0
i = 0
while i < len(feat_list):
    j = 0
    while j < len(feat_list[i]):
        feat_list[i][j] = net_manipulation.normalize(feat_list[i][j], 0.0, 0.194) * 10
        # feat_list[i][j] = feat_list[i][j] * 100
        num_elements += 1
        if feat_list[i][j] == 0:
            num_zeros += 1
        if feat_list[i][j] >= 0.001:
            num_medium += 1
        if feat_list[i][j] >= 0.1:
            num_high += 1
        if (feat_list[i][j] > 0) and (feat_list[i][j] < 0.01):
            num_mid += 1
        j += 1
    i += 1
print("Elements: ", num_elements)
print("Zeros: ", num_zeros)
print("Mediums: ", num_medium)
print("Highs: ", num_high)
print("Between 0-0.01: ", num_mid)

# Test normalization:
# print(net_manipulation.normalize(5, 0.0, 7.5))
