:: Script to change the file format, using FFMPEG.
:: To use in anything different change the extension in the for loop.
:: FFMPEG must be installed AND included in the Windows PATH.

@ECHO OFF

:: The output folder must be created manually beforehand.
for %%a in (*.flac) DO (
	ffmpeg -i "%%a" -ar 44100 "wav_of_flac\%%~na.wav"
)

:: To use in a single file, just run:
:: ffmpeg -i in.mp3 out.wav


:: For debugging purposes
::PAUSE