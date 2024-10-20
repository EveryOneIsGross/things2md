# Video Frame Extractor and Subtitle Processor

This Python script extracts frames from a video file, processes accompanying subtitles, and generates a JSON file containing frame data and corresponding subtitle chunks.

## Features

- Extracts a specified number of frames from a video file
- Processes SRT subtitle files
- Generates a JSON file with frame data and relevant subtitle text
- Creates frame images in PNG format

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- srt

## Usage

1. Run the script:
   ```
   python movie2FOLDER.py
   ```

2. Follow the prompts to input:
   - Path to the video file
   - Path to the SRT subtitle file
   - Output folder path
   - Number of frames to extract
   - Context window size for subtitle entries (optional, default is 60)

3. The script will create an 'images' subfolder in the specified output folder, containing:
   - Extracted frame images (PNG format)
   - A JSON file with frame data and subtitle chunks

## Output

- Frame images: `{safe_title}_frame_{timestamp}.png`
- JSON file: `{safe_title}_frame_data.json`

## Notes

- The script uses a safe filename function to ensure compatibility across different operating systems.
- Frame extraction is distributed evenly across the video duration.
- Subtitle chunks are non-overlapping and include context before and after each frame's timestamp.