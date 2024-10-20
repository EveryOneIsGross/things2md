# EPUB to Markdown Converter

A Python script that converts EPUB files to Markdown format, preserving images and basic structure.

## Features

- Converts EPUB files to Markdown
- Extracts and resizes images to PNG format
- Preserves cover images
- Maintains valid web links
- Optionally preserves folder structure

## Requirements

- Python 3.6+
- Required libraries: `ebooklib`, `beautifulsoup4`, `html2text`, `Pillow`

## Installation

1. Clone this repository
2. Install required libraries:
   ```
   pip install ebooklib beautifulsoup4 html2text Pillow
   ```

## Usage

python epubs2md.py <epub_folder> <output_folder> [--preserve-structure]

- `<epub_folder>`: Path to the folder containing EPUB files
- `<output_folder>`: Path to the output folder for Markdown files
- `--preserve-structure`: Optional flag to maintain the original folder structure
