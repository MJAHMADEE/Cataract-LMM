#!/bin/bash
# Directory containing input files (use current directory if not specified)
INPUT_DIR="."
OUTPUT_DIR="."
# Check if FFmpeg is installed
command -v ffmpeg >/dev/null 2>&1 || { echo >&2 "FFmpeg is not installed. Please install it to continue."; exit 1; }
# Loop through all .mp4 files in the input directory
for input_file in "$INPUT_DIR"/*.mp4; do
    # Skip if no files are found
    [[ -e "$input_file" ]] || continue
    # Extract the main filename without extension
    main_filename=$(basename -- "${input_file%.*}")
    # Define the output file name
    output_file="${OUTPUT_DIR}/processed_${main_filename}.mp4"
    # Apply the FFmpeg command
    ffmpeg -i "$input_file" \
        -filter_complex "[0:v]crop=268:58:6:422,avgblur=10[fg];[0:v][fg]overlay=6:422[v]" \
        -map "[v]" -map 0:a \
        -c:v libx265 -crf 23 -c:a copy -movflags +faststart "$output_file"
    # Check if FFmpeg succeeded
    if [ $? -eq 0 ]; then
        echo "Processed: $input_file -> $output_file"
    else
        echo "Error processing: $input_file"
    fi
done
echo "Batch processing complete."
