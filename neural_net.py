"""Build and train a Neural Network to evaluate flac compression quality.
TODO: review doc
Create a Neural Network with multiple Layers and train it with Backpropagation. Use audio features extracted with
the audio_manipulation module as a dataset. Save the trained Network, but not the dataset.
Use the net_manipulation module for Network related actions.
"""

from pybrain3.structure import RecurrentNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer, TanhLayer
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.tools.xml.networkwriter import NetworkWriter
import time
import net_manipulation
import wakepy


# Proportion for the division of the input sample, in [0.0, 1.0].
# Define the percentage of the sample used for training. The rest is used for testing.
# When set to 1.0 the saved trained Network is supposed to activate somewhere else for testing.
PROP = 0.95

# Number of epochs to train the network, if this training method is enabled.
EPOCH_NUM = 500

# The name of the file to save the Neural Network.
NET_FILENAME = "trained_penthy.xml"


def pause():
    """Pauses the program, until there is user input. Auxiliary function for testing and debugging."""
    input("Paused: Press <ENTER> to continue...")


# Build a Neural Network.
n = buildNetwork(91, 60, 1, hiddenclass=LinearLayer, bias=True, outputbias=True, recurrent=True)
n.randomize()

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

# Custom mini dataset for debugging
# train_ds = SupervisedDataSet(91, 1)
# train_ds.addSample([0.0]*91, 0.0)
# train_ds.addSample([0.01]*91, 1.0)

# for inp, targ in train_ds:
#     print(targ)
# print("train_ds: ", train_ds)
# print("length of train_ds: ", len(train_ds))
# print("test_ds: ", test_ds)
# print("length of test_ds: ", len(test_ds))
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
