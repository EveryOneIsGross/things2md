# YouTube Video Processor

A Python tool for downloading, transcribing, and analyzing YouTube videos.

## Features

- Download YouTube videos
- Transcribe audio using local models or API services
- Extract and analyze video frames
- Generate markdown summaries with transcriptions and frame descriptions

## Requirements

- Python 3.7+
- FFmpeg
- Various Python libraries (see requirements.txt)

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install FFmpeg if not already present on your system

## Usage

```
python yt_turbo.py <input_file> <output_folder> [options]
```

- `<input_file>`: Text file containing YouTube URLs (one per line)
- `<output_folder>`: Directory to store processed files

### Options

- `--mode`: Transcription mode (`api` or `local`, default: local)
- `--language`: Audio language (default: en)
- `--num_frames`: Number of frames to extract (default: 32)
- `--vision_api`: Image processing API (`ollama`, `google`, or `openai`, default: ollama)

See `--help` for full list of options.

## Output

For each video, the tool generates:
- Transcription (JSON)
- Frame descriptions (JSON)
- Summary markdown file
