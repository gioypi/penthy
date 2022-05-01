![logo_full](https://user-images.githubusercontent.com/52460732/152804013-f6ff06ca-5968-402d-be61-cce3c2cb7683.png)
# penthy
Neural network for classification of flac compression quality.

## About penthy
penthy is an audiophile tool :notes: to check whether a flac file contains truly lossless music, as it is supposed to, or comes from a lossy source, like an mp3 file.

The flac file format compresses music in a way that it remains unaltered, in contrast to lossy compression formats (like mp3) that sacrifice sound quality to limit file size. Confirming that your flac discography is truly lossless is an open-ended problem. One approach to tell the difference is machine learning. Although, penthy cannot evaluate every aspect of digital audio to guarantee that your file is exactly what came out of the studio, she tries to identify transcoding from mp3 sources. A song that passes penthy's challenge is not necessarily genuine, but it is unlikely to be an mp3 wearing a flac trenchcoat.

_penthy_ is short for _Penthesilea_, a skilled queen of the Amazons who fought in the Trojan War, according to Greek mythology.

## Working principle
A **Convolutional Neural Network** was trained with the highest frequencies of several songs in the form of spectogram images, in order to recognize flac files that were once mp3s.

The songs are split into small segments to produce the spectograms which are given to the network as inputs. The training dataset contained the truly lossless versions of the songs and their fake counterparts – flac files transcoded from mp3 files generated from the originals. Various music genres and mp3 qualities were included. Each spectogram is a 128x128 px RGB image depicting only the 16200-22000 Hz frequency range for 8 seconds of audio, saved as a numpy array. The trained model accepts flac or wav tracks as input and outputs a float number from 0 to 1. An output of '0' corresponds to audio transcoded to mp3 and back to a lossless format. An output of '1' classifies the song as not transcoded from an mp3 source, but it could still be transcoded from a different format, subjected to upsampling or altered in other ways.

The CNN is structured as follows:
![cnn arch white](https://user-images.githubusercontent.com/52460732/166119728-477f7357-be9d-4d65-b316-5c4ce7ab2cd1.png)


The current trained model performs generally well, with an approximate accuracy of 90%.
False negatives (genuine files classified as transcoded) are more common than false positives (transcoded files classified as truly lossless), especially for songs that lack higher frequencies.

## Used technologies
- [Python 3](https://www.python.org/)
- [TensorFlow 2](https://www.tensorflow.org/) with [Keras](https://keras.io/)
- [FFmpeg](https://ffmpeg.org/)

## Usage
You may use this code to evaluate your flac files with the pretrained model or train your own if you have access to truly lossless discography.
- *neural_net.py* builds a dataset with flexible multiprocessing and trains a new model.
- *trained.py* evaluates a single file.
- *trained_dir.py* evaluates all applicable files in a directory with multiprocessing (**recommended** if you use penthy to scan your collection).
- *audio_manipulation.py* is used by all modules to generate the spectograms.
- No dataset included.

For instance, running *trained_dir.py* for a directory that contains both genuine and transcoded files will output something like this:
![output_demo](https://user-images.githubusercontent.com/52460732/166128288-8a5b9744-98de-4ec3-991d-4d93069a96d2.png)  
You do ***not*** need both versions of a file to get an accurate evaluation, as in this example. Each file is classified separately.

## Installation Requirements

- [Python](https://www.python.org/downloads/) (3.7 64-bit has been tested) (in Windows, make sure to add Python to the PATH environment variable)
- [FFmpeg](https://ffmpeg.org/download.html) (including ffprobe) (in Windows, make sure to add FFmpeg to the PATH environment variable)

and the following python packages (or see *requirements.txt*, generated by PyCharm):
- [tensorflow](https://www.tensorflow.org/install) and optionally its demanding [requirements](https://www.tensorflow.org/install/gpu) for GPU support (**recommended**)
- [keras](https://keras.io/getting_started/)
- [numpy](https://numpy.org/)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- [colorama](https://github.com/tartley/colorama)
- [wakepy](https://github.com/np-8/wakepy) (only if you train new models)

plus the dependencies of these packages that will come up during installation (e.g. pandas, scipy, matplotlib, scikit-learn).

## Credits and license
The license of this repository refers to code written by the author and not the libraries and functions used. For those, look at the respective licenses of the original projects.
Music in the example of usage is courtesy of Dean Washburn (nvlachost@gmail.com).
Proper attribution of penthy requires mentioning all parties of the following crediting.

Achilleas Papastamatiou developed penthy as part of his undergraduate thesis in the Department Of Computer Science And Telecommunications in the University of Thessaly in Greece. The project was supervised by professor George Fourlas and supported by professor Vaggelis Spyrou.

## Auxiliary technologies used while developing
(Not required to run or fork penthy)
- [Audacity](https://www.audacityteam.org/)
- [Spek](http://spek.cc/)
- [VLC media player](https://www.videolan.org/vlc/)
- Audiochecker (by Dester)
