
# Gemini Conversation Analyzer

This Python script uses the Gemini API to analyze and reconstruct conversations from raw, unpunctuated text. It provides a detailed linguistic analysis and structures the conversation for better readability.

## Features

- Analyzes raw text input from a Markdown file
- Reconstructs conversations with proper formatting and punctuation
- Identifies distinct speakers and separates utterances
- Provides detailed linguistic analysis, including tone, register, and cultural references
- Handles both dialogues and monologues
- Generates a summary of the conversation

## Requirements

- Python 3.6+
- `google-generativeai` library
- Gemini API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gemini-conversation-analyzer.git
   cd gemini-conversation-analyzer
   ```

2. Install the required packages:
   ```
   pip install google-generativeai
   ```

3. Set up your Gemini API key as an environment variable:
   ```
   export GEMINI_API_KEY='your-api-key-here'
   ```

## Usage

Run the script with a Markdown file containing the raw conversation text:

```
python transcripts/geminiSUMCONVERSATION.py input_file.md
```

The script will generate an output file named `input_file_flashtranscribed.md` with the analyzed and reconstructed conversation.

## How it works

1. The script reads the input Markdown file.
2. It sends the content to the Gemini API for analysis and reconstruction.
3. If the initial response is incomplete, it continues the conversation with the API.
4. The reconstructed conversation and analysis are saved to a new Markdown file.
