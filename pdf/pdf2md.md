# PDF Processing Tools

This repository contains two Python scripts for processing PDF files and enhancing markdown files with image descriptions.

## 1. PDFolder2MD Marker

A simple Python script to process PDF files in a folder (and its subfolders) using the `marker_single` command.

### Description

This script walks through a specified directory and its subdirectories, identifies PDF files, and processes them using the `marker_single` command. It creates an 'output' folder in each directory containing PDFs and runs the `marker_single` command on each PDF file.

marker project: [https://github.com/VikParuchuri/marker](https://github.com/VikParuchuri/marker)

### Usage

Run the script from the command line, providing the path to the folder containing PDF files:

```
python pdfolder2md_marker.py /path/to/pdf/folder
```

### Features

- Recursively processes all PDF files in the specified folder and its subfolders
- Creates an 'output' folder in each directory containing PDFs
- Uses `marker_single` command with a batch multiplier of 2

## 2. PDF MD Folder IMAGE2TEXT

A Python script that processes markdown files in a folder and its subfolders, updating image descriptions using AI-generated content.

### Description

This script walks through a specified directory and its subdirectories, identifies markdown files, and updates the descriptions of images within these files using the Ollama AI model.

### Usage

Run the script from the command line, providing the path to the folder containing markdown files:

```
python pdfmdfolderIMAGE2TEXT.py /path/to/markdown/folder [options]
```

Options:
- `--yaml_file`: Path to the YAML file containing prompt templates (default: prompt_template.yaml)
- `--use_context`: Use surrounding text context in the prompt
- `--timeout`: Timeout for API calls in minutes (default: 2)
- `--max_retries`: Maximum number of retries for API calls (default: 3)

### Features

- Processes markdown files recursively in the specified folder and subfolders
- Updates image descriptions using AI-generated content
- Supports custom prompt templates via YAML file
- Option to use surrounding text context for better descriptions
- Configurable timeout and retry settings for API calls
- Handles small images and various error scenarios

## Requirements

- Python 3.x
- `marker_single` command-line tool (for PDFolder2MD Marker)
- Required Python packages: ollama, yaml, tiktoken, Pillow, colorama, nltk (for PDF MD Folder IMAGE2TEXT)

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install ollama pyyaml tiktoken Pillow colorama nltk
   ```
3. Ensure `marker_single` is installed and accessible in your system's PATH (for PDFolder2MD Marker)

## Note

Ensure that all required tools and dependencies are properly installed and configured on your system before running these scripts.
```
