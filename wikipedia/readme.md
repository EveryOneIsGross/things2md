# Wikipedia to Markdown Converter

A Python script that converts Wikipedia pages to Markdown format.

## Features

- Converts single Wikipedia URLs or processes multiple URLs from a file
- Removes unnecessary elements like table of contents and sidebars
- Extracts external links into a separate file
- Handles references and creates a references section
- Detects and skips placeholder or non-existent pages

## Requirements

- Python 3.6+
- Required libraries: requests, beautifulsoup4

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install requests beautifulsoup4
   ```

## Usage

Convert a single Wikipedia page:
```
python wikipedia2md.py https://en.wikipedia.org/wiki/Python_(programming_language) output_directory
```

Process multiple URLs from a file:
```
python wikipedia2md.py input_file.txt output_directory
```

## Output

- Markdown files (.md) for each processed Wikipedia page
- External links files (_linkz.md) if external links are present

## Limitations

- Will not perfectly handle all Wikipedia page layouts due to aggressive cleaning
- Does not process images or complex tables