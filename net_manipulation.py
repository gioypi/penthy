"""Prepare and edit datasets and the network's output.

Functions:
    normalize(var, current_lower, current_upper, desired_lower, desired_upper): Normalize a number to fit within the
    desired range.
    fill_ds(in_ds, dir_path, target, verbose): Add samples to a PyBrain dataset, using audio features extracted
    from all audio files in the specified directory.
    shuffle_ds(in_ds): Create a randomly shuffled version of a given SupervisedDataSet.
"""

import audio_manipulation
import os
import random
from pybrain3.datasets import SupervisedDataSet


def normalize(var, current_lower, current_upper, desired_lower=0.0, desired_upper=1.0):
    """Normalize a number to fit in a given range.

    Parameters:
        var: The number to normalize.
        current_lower (float): The lower bound of the range before normalization.
        current_upper (float): The upper bound of the range before normalization.
        desired_lower (float): The lower bound of the normalized range.
        desired_upper (float): The upper bound of the normalized range.
    Returns:
        norm_var: The normalized number.
    """

    norm_var = (var - current_lower) * ((desired_upper - desired_lower) / (current_upper - current_lower))
    norm_var += desired_lower

    # Make sure no case escapes the range.
    if norm_var < desired_lower:
        norm_var = desired_lower
    elif norm_var > desired_upper:
        norm_var = desired_upper
    return norm_var


def fill_ds(in_ds, dir_path, target, verbose=False):
    """Extract audio features from all accepted files in a directory and use them to add samples to a Pybrain dataset.

    Do NOT look recursively into directories, only open files in the current level.
    Expect a very large dataset.
    All added samples will have the value of target as desirable output. Audio files are expected to be categorized into
    directories based on desirable output, in order to call this function for each category.

    Parameters:
        in_ds (Pybrain3 SupervisedDataSet): The existing dataset to be filled with (extra) samples.
        dir_path (string): The path to the directory containing the audio files to be opened.
        target (int or float): The desirable output of the Network for all files in the given path.
        verbose (boolean): When True, print the current file and the size of the lists and dataset.
    Returns:
        in_ds (Pybrain3 SupervisedDataSet): The dataset that was filled with (extra) samples.
    """

    permitted_extensions = ".wav"
    # min_f = 1000
    # max_f = -1000
    for file_name in os.listdir(dir_path):
        if file_name.endswith(permitted_extensions):
            file_path = os.path.join(dir_path, file_name)
            if verbose:
                print("Extracting features of ", file_name)
            # feat_list = audio_manipulation.extract_features(file_path)
            feat_list = audio_manipulation.extract_spectogram(file_path)
            if verbose:
                print("length of feat_list: ", len(feat_list), " time fragments")
                # print("length of feat_list[i]: ", len(feat_list[0]), " audio features\n")
                print("length of feat_list[i]: ", len(feat_list[0]), " frequency segments\n")

            # Find the range of the features to normalize them later.
            # i = 0
            # while i < len(feat_list):
            #     j = 0
            #     while j < len(feat_list[i]):
            #         if feat_list[i][j] < min_f:
            #             min_f = feat_list[i][j]
            #         elif feat_list[i][j] > max_f:
            #             max_f = feat_list[i][j]
            #         j += 1
            #     i += 1

            # Normalize features before adding them to the dataset.
            i = 0
            while i < len(feat_list):
                j = 0
                while j < len(feat_list[i]):
                    # Customize the normalization bounds per dataset.
                    feat_list[i][j] = normalize(feat_list[i][j], 0.0, 0.0194) * 10   # Use for 91 input frequencies.
                    # feat_list[i][j] = normalize(feat_list[i][j], 0.0, 0.194) * 10  # For older models.
                    j += 1
                i += 1
            for inp in feat_list:
                in_ds.addSample(inp, target)
    if verbose:
        print("length of in_ds: ", in_ds.getLength())
    # print("Whole DS bounds:")
    # print("min: ", min_f, " max: ", max_f)
    return in_ds


def shuffle_ds(in_ds):
    """Shuffle all samples of a Pybrain SupervisedDataSet.

    Shuffle the given dataset randomly, while retaining the link between input and target of the Network.
    Create and return a new shuffled dataset. Slow and memory-hungry method.

    Parameters:
        in_ds (Pybrain3 SupervisedDataSet): The dataset to be shuffled.
    Returns:
        new_ds (Pybrain3 SupervisedDataSet): The newly created shuffled dataset.
    """

    in_list = list(list())
    target_list = list()
    for inp, target in in_ds:
        in_list.extend(inp)
        target_list.extend(target)
    merged_list = list(zip(in_list, target_list))
    random.shuffle(merged_list)
    random.shuffle(merged_list)     # Shuffle twice for better results. Disable if unnecessary.
    in_list, target_list = zip(*merged_list)
    new_ds = SupervisedDataSet(91, 1)
    i = 0
    while i < len(target_list):
        new_ds.addSample(in_list[i], target_list[i])
        i += 1
    return new_ds
