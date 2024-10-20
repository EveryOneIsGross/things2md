# Podcast RSS to Whisper Transcription

This Python script processes podcast RSS feeds, downloads episodes, and transcribes them using OpenAI's Whisper model.

## Features

- Parses podcast RSS XML feeds
- Extracts episode metadata
- Downloads audio files
- Transcribes audio using Whisper large-v3 model
- Generates markdown files with episode info and transcripts

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- librosa
- pydub
- tqdm

## Usage

1. Update the input and output paths in the script:
   - `xml_file`: Path to the RSS XML file
   - `json_file`: Path for the output JSON file
   - `media_folder`: Directory to store downloaded audio files
   - `markdown_folder`: Directory to store output markdown files

2. Run the script:
   ```
   python rssxml2whisper.py
   ```

## Notes

- Requires sufficient disk space for audio downloads and transcripts
- Transcription speed depends on your hardware (GPU recommended)
- Adjust `max_chunk_size_mb` in `segment_audio()` if memory issues occur