:: Script to change the file format, using FFMPEG.
:: To use in anything different change the extension in the for loop.
:: FFMPEG must be installed AND included in the Windows PATH.

@ECHO OFF

:: The output folder must be created manually beforehand.
for %%a in (*.mp3) DO (
	ffmpeg -i "%%a" "flac_transcoded\%%~na.flac"
)


:: For debugging purposes
::PAUSE