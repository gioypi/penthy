:: Script to change the sample frequency and bit-depth of audio files, using FFMPEG.
:: To use in anything different than flac, change the extension in the for loop.
:: FFMPEG must be installed AND included in the Windows PATH.

@ECHO OFF

:: The output folder must be created manually beforehand.
for %%a in (*.flac) DO (
	ffmpeg -i "%%a" -ar 44100 -sample_fmt s16 "flac_44-16\%%~na.flac"
)



:: For debugging purposes
::PAUSE