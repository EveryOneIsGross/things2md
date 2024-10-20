# CHUNKsum

CHUNKsum is a hierarchical Markdown summarizer that processes large documents efficiently.

## Features

- Breaks down large Markdown files into manageable chunks
- Summarizes text using the Ollama API with the gemma2:2b model
- Employs a hierarchical summarization approach
- Utilizes concurrent processing for improved performance
- Saves intermediate summaries at each layer

## Requirements

- Python 3.7+
- Dependencies: ollama, tiktoken, tqdm, colorama

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## Options
- `--output`: Specify output file (default: input_file_summary.md)
- `--chunk-size`: Set chunk size in tokens (default: 1000)
- `--model`: Choose AI model (default: gemma2:2b)

## Usage

python CHUNKsum.py <input_file> [options]

## Output

- Generates intermediate summary files for each layer
- Produces a final summary file