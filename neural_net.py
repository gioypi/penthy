"""Create a Convolutional Neural Network with Keras and train it to tell truly lossless and transcoded audio apart.

Build a Keras CNN to evaluate audio compression based on spectograms. Use spectogram images extracted with
the audio_manipulation module as a dataset. Save the trained Network, but not the dataset.
A network output of '1' corresponds to a spectogram derived from a truly lossless source.
An output of '0' corresponds to a spectogram derived from audio transcoded to mp3 and back to a lossless format.
Only possible mp3 transcoding is inspected. Even if a file is verified as truly lossless in terms of mp3 transcoding,
it could still be upsampled, transcoded from a different format or altered in other ways.
"""

import audio_manipulation as am
import time
import wakepy
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout


def main():
    # Number of samples from the input dataset used in each update within an epoch.
    BATCH_SIZE = 16

    # Number of epochs to train the network.
    EPOCH_NUM = 50

    # Proportion of the input dataset used for validation, instead of training.
    # Float between 0 and 1.
    PROP_VALID = 0.06

    # Path to the file or directory where the model will be saved after training.
    MODEL_PATH = "saved_models/trained_penthy"

    # Check that tensorflow recognizes the GPU.
    print("Tensorflow GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    wakepy.set_keepawake(keep_screen_awake=False)   # Prevent OS from sleeping during long training sessions.
    print("Extracting spectograms...")
    start_time = time.monotonic()
    spect_true = np.array(am.extract_spectogram_from_dir("dataset_files/true_flac_44-16"))
    in_ds = tf.data.Dataset.from_tensor_slices((spect_true, np.array([1] * spect_true.shape[0])))

    # [spect stands for spectogram, trans and transc stand for transcoded]
    # For the transcoded samples, use various bitrates and compression qualities.
    trans_list = am.extract_spectogram_from_dir("dataset_files/flac_44-16_transcoded_from_mp3_320")
    trans_list.extend(am.extract_spectogram_from_dir("dataset_files/flac_44-16_transcoded_from_mp3_128"))
    trans_list.extend(am.extract_spectogram_from_dir("dataset_files/flac_44-16_transc_from_mp3_320_prime"))
    spect_trans = np.array(trans_list)
    temp_ds = tf.data.Dataset.from_tensor_slices((spect_trans, np.array([0] * spect_trans.shape[0])))
    in_ds = in_ds.concatenate(temp_ds)
    end_time = time.monotonic()
    # Real time passed, not time in CPU/GPU.
    print("Spectograms ready. Creation duration: %.4f minutes" % ((end_time - start_time) / 60))
    # print("Input dataset:", in_ds)
    # print(list(in_ds.as_numpy_iterator()))

    print("Preparing dataset...")
    start_time = time.monotonic()
    num_elements = spect_true.shape[0] + spect_trans.shape[0]
    in_ds = in_ds.shuffle(num_elements, reshuffle_each_iteration=False)
    num_valid = int(num_elements * PROP_VALID)
    num_train = num_elements - num_valid
    valid_ds = in_ds.take(num_valid)
    train_ds = in_ds.skip(num_valid)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(num_train, reshuffle_each_iteration=True)   # Reshuffle after each epoch.
    end_time = time.monotonic()
    print("Dataset ready. Preparation duration: %.4f minutes" % ((end_time - start_time) / 60))
    print("Dataset size:", num_elements, "samples")
    print("of which", num_train, "used for training and", num_valid, "used for validation.")

    print("Creating neural network...")
    net = Sequential()
    net.add(Conv2D(10, (3, 3), padding="valid", data_format="channels_last", activation="relu", use_bias=True,
                   input_shape=(am.HEIGHT, am.WIDTH, 3), kernel_initializer="random_normal"))
    net.add(Conv2D(10, (3, 3), padding="valid", data_format="channels_last", activation="relu", use_bias=True,
                   kernel_initializer="random_normal"))
    net.add(MaxPooling2D((2, 2), padding="valid", data_format="channels_last"))
    net.add(Conv2D(8, (3, 3), padding="valid", data_format="channels_last", activation="relu", use_bias=True,
                   kernel_initializer="random_normal"))
    net.add(Conv2D(8, (3, 3), padding="valid", data_format="channels_last", activation="relu", use_bias=True,
                   kernel_initializer="random_normal"))
    net.add(MaxPooling2D((2, 2), padding="valid", data_format="channels_last"))
    net.add(Flatten(data_format="channels_last"))
    net.add(Dropout(0.2))
    net.add(Dense(32, activation="relu", use_bias=True, kernel_initializer="random_normal"))
    net.add(Dropout(0.2))
    net.add(Dense(16, activation="relu", use_bias=True, kernel_initializer="random_normal"))
    net.add(Dense(1, activation="sigmoid", use_bias=True, kernel_initializer="random_normal"))
    print("Neural network created.")
    net.summary()

    print("Training neural network...")
    start_time = time.monotonic()
    net.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.005, rho=0.95),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])
    net.fit(train_ds, epochs=EPOCH_NUM, verbose=1, validation_data=valid_ds)
    end_time = time.monotonic()
    print("Training complete. Training duration: %.4f minutes" % ((end_time - start_time) / 60))

    print("Saving neural network to file...")
    net.save(MODEL_PATH, save_format=tf)
    print("Network saved.")
    wakepy.unset_keepawake()


if __name__ == "__main__":
    main()
