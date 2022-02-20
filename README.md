![logo_full](https://user-images.githubusercontent.com/52460732/152804013-f6ff06ca-5968-402d-be61-cce3c2cb7683.png)
# penthy
Neural network for classification of flac compression quality.

## About penthy
penthy is an audiophile tool :notes: to check whether a flac file contains truly lossless music, as it is supposed to, or comes from a lossy source, like an mp3 file.

The flac file format compresses music in a way that it remains unaltered, in contrast to lossy compression formats (like mp3) that sacrifice sound quality to limit file size. Confirming that your flac discography is truly lossless is an open-ended problem. One approach to tell the difference is machine learning. Although, penthy cannot evaluate every aspect of digital audio to guarantee that your file is exactly what came out of the studio, she tries to identify transcoding from mp3 sources.

_penthy_ is short for _Penthesilea_, a skilled queen of the Amazons who fought in the Trojan War, according to Greek mythology.

## Working principle
A **recurrent neural network** with one hidden layer was trained with the highest frequencies of several songs from different genres. The frequencies were divided into 91 time segments and given to the network as inputs. The dataset contained the truly lossless versions of the songs and their fake counterparts – flac files transcoded from mp3 files generated from the originals.

## Used technologies
- Python 3.7
- [Pybrain](http://www.pybrain.org/), with the Python 3 wrapper, [Pybrain3](https://github.com/AlexProgramm/pybrain3)
- [PyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
- [FFmpeg](https://ffmpeg.org/)

## Installation
You may use this code to evaluate your flac files with the pretrained model or train your own if you have access to truly lossless discography. If you do not plan to train, you can follow the minimum requirements, otherwise you need to follow the complete requirements. The _neural_net.py_ and _retrain.py_ files are not needed for evaluations.

#### Minimum requirements (evaluation only):
- Python (3.7 64-bit has been tested) (in Windows, make sure to add it to the PATH environment variable)
- FFmpeg (in Windows, make sure to add it to the PATH environment variable)

and the following python packages:
- numpy
- scipy
- matplotlib (can be omitted if no error occurs)
- Pybrain3 (and its [multiple dependencies](http://pybrain.org/docs/quickstart/installation.html))
- PyAudioAnalysis

plus the dependencies of these packages that will come up during installation.

#### Complete requirements:
All the minimum requirements and the following python packages:
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- [wakepy](https://github.com/np-8/wakepy)
 
plus the dependencies of these packages that will come up during installation.

## Credits and license
The license of this repository refers to code written by the author and not the libraries and functions used. For those, look at the respective licenses of the original projects.
Proper attribution requires mentioning all parties of the following crediting.

Achilleas Papastamatiou developed penthy as his undergraduate thesis in the Department Of Computer Science And Telecommunications in the University of Thessaly in Greece. The project was supervised by professor George Fourlas and supported by professor Vaggelis Spyrou.

## Auxiliary technologies used while developing
(Not required to run or fork penthy)
- [Audacity](https://www.audacityteam.org/)
- [Spek](http://spek.cc/)
- [VLC media player](https://www.videolan.org/vlc/)
- Audiochecker (by Dester)
