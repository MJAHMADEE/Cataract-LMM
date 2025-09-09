@echo off
setlocal enabledelayedexpansion

set INPUT_DIR=.
set OUTPUT_DIR=.

REM Check if FFmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo FFmpeg is not installed. Please install it to continue.
    pause
    exit /b 1
)

REM Loop through all .mp4 files
for %%f in ("%INPUT_DIR%\*.mp4") do (
    set "input_file=%%f"
    set "filename=%%~nf"
    set "output_file=%OUTPUT_DIR%\processed_!filename!.mp4"
    
    echo Processing: !input_file!
    ffmpeg -i "!input_file!" -vcodec libx265 -crf 23 -movflags +faststart "!output_file!"
    
    if !errorlevel! equ 0 (
        echo Processed: !input_file! -^> !output_file!
    ) else (
        echo Error processing: !input_file!
    )
    echo.
)

echo All files processed!
pause
