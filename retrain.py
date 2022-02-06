"""Create a Neural Network from file and train it again to improve accuracy.
TODO: review doc
Create a Neural Network from a saved trained model and train it with Backpropagation. Use audio features extracted with
the audio_manipulation module as a dataset. Save the trained Network, but not the dataset.
Use the net_manipulation module for Network related actions.
"""

from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader
import time
import net_manipulation
import wakepy
import sys


# Proportion for the division of the input sample, in [0.0, 1.0].
# Define the percentage of the sample used for training. The rest is used for testing.
# When set to 1.0 the saved trained Network is supposed to activate somewhere else for testing.
PROP = 0.95

# Number of epochs to train the network, if this training method is enabled.
EPOCH_NUM = 500

# The name of the file to read from and save the Neural Network.
NET_FILENAME = "trained_penthy.xml"

# Create a Neural Network from file.
try:
    n = NetworkReader.readFrom(NET_FILENAME)
except OSError:
    print("Failed to read network from file.")
    sys.exit(1)
else:
    print("Network created from file.")

# Create the DataSet.
print("Extracting audio features...")
in_ds = SupervisedDataSet(91, 1)  # The arguments are the dimensions of the input and output layers.
target = 1  # The desired output of the Network for the specified input.
in_ds = net_manipulation.fill_ds(in_ds, "dataset_files/wav_of_true_flac", target)
target = 0
in_ds = net_manipulation.fill_ds(in_ds, "dataset_files/wav_of_transc_flac_128", target)     # Changed to lower quality.
# print(in_ds.batches(1), "length: ", len(in_ds.batches(1)))
print("Features extracted.")
# print("in_ds: ", in_ds)
# print("length of in_ds: ", len(in_ds))

print("Preparing dataset...")
in_ds = net_manipulation.shuffle_ds(in_ds)
train_ds, test_ds = in_ds.splitWithProportion(PROP)
# train_ds = in_ds        # Skip slow split for testing.
print("Dataset ready.")

# Train the Network and save it.
print("Training neural network...")
start_time = time.process_time()
wakepy.set_keepawake(keep_screen_awake=False)   # Prevent OS from sleeping during long training sessions.
trainer = BackpropTrainer(n, train_ds, learningrate=0.02)  # Training through backpropagation.
# trainer.trainEpochs(EPOCH_NUM)

# Alternatively, shuffle the dataset between Epochs.
i = 0
while i < EPOCH_NUM:
    print("Current iteration: ", (i + 1), "/", EPOCH_NUM)
    trainer.train()
    train_ds = net_manipulation.shuffle_ds(train_ds)
    i += 1

wakepy.unset_keepawake()
end_time = time.process_time()
print("Training complete. Training duration: %.4f minutes" % ((end_time - start_time) / 60))
try:
    NetworkWriter.writeToFile(n, NET_FILENAME)
except OSError:
    print("Failed to save network.")
else:
    print("Network saved to xml file.")

# Use the trained network and test for errors.
# print(test_ds)
# start_time = time.process_time()
# trainer.testOnData(test_ds, verbose=True)
# end_time = time.process_time()
# print("Activation duration: %.4f seconds" % (end_time - start_time))
