import json
import os
import argparse
from dotenv import load_dotenv
import tiktoken
from groq import Groq
from openai import OpenAI
import time

load_dotenv()

def save_response(input_filename, content):
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f"{base_name}_resynthed.md"
    with open(output_filename, "w") as f:
        f.write(content)
    print(f"Saved response as {output_filename}")

def chunk_text(text, max_tokens=32000, overlap=512):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        start = end - overlap
    return chunks

def initial_analysis(client, model, text, llm_choice):
    system_message = """You are an expert in conversation analysis. Your task is to provide an initial comprehensive analysis of the entire conversation. Focus on:

    1. Identifying distinct speakers and their characteristics.
    2. Analyzing the overall conversation structure and flow.
    3. Identifying key themes, topics, and any notable linguistic features.
    4. Hypothesizing about the conversation's context, setting, and medium.

    Provide your analysis in a structured format using XML tags."""

    user_message = f"""Analyze the following conversation:

{text[:10000]}... (truncated for brevity)

Provide a comprehensive initial analysis, including speaker identification, conversation structure, and overall context."""

    max_tokens = 8000 if llm_choice == "groq" else 4000

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content

def process_chunk(client, model, system_message, chunk, chunk_index, total_chunks, initial_analysis, llm_choice, previous_context=""):
    user_message = f"""You are processing chunk {chunk_index} of {total_chunks}.

Previous context (if any):
{previous_context}

Initial analysis:
{initial_analysis}

New chunk to process:
{chunk}

Instructions:
- Focus on reconstructing the fully expressed dialogue based on the initial analysis.
- Maintain consistency with previous chunks and the identified speakers.
- Use strict speaker ID formatting: "Speaker {{number}}: " or "{{identified_name}}: "
- Wrap your response in XML tags as follows:
  <transcription>
    [The reconstructed conversation with proper formatting and punctuation]
  </transcription>
  <notes>
    [Any additional notes or observations specific to this chunk]
  </notes>
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    max_tokens = 8000 if llm_choice == "groq" else 4000

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content

def process_input(input_file, llm_choice):
    with open(input_file, 'r') as f:
        user_input = f.read()

    if llm_choice == "groq":
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = "llama-3.1-70b-versatile"
    elif llm_choice == "ollama":
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )
        model = "llama2"
    else:
        raise ValueError("Invalid LLM choice")

    print("Performing initial analysis...")
    initial_analysis_result = initial_analysis(client, model, user_input, llm_choice)
    print("Initial analysis completed.")

    system_message = """You are an expert in conversation reconstruction. Your task is to accurately transcribe and format the conversation based on the provided initial analysis. Focus on:

    1. Using the correct speaker labels consistently.
    2. Reconstructing the dialogue with appropriate punctuation and formatting.
    3. Maintaining the conversation flow and context across chunks.

    Provide your transcription in a structured format using XML tags."""

    chunks = chunk_text(user_input)
    full_response = f"<initial_analysis>\n{initial_analysis_result}\n</initial_analysis>\n\n"
    previous_context = ""

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        response = process_chunk(client, model, system_message, chunk, i+1, len(chunks), initial_analysis_result, llm_choice, previous_context)
        print(f"Chunk {i+1} processed.")
        full_response += response + "\n\n"
        
        # Update previous_context for the next iteration
        previous_context = chunk[-512:]  # Keep last 512 characters as context

    save_response(input_file, full_response)

    return full_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a conversation using an LLM API")
    parser.add_argument("input_file", help="Path to the input Markdown file")
    parser.add_argument("--llm", choices=["groq", "ollama"], default="groq", help="Choose the LLM API to use (default: groq)")
    args = parser.parse_args()

    try:
        response = process_input(args.input_file, args.llm)
        print("Response saved successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")