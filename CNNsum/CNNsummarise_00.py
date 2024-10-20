import argparse
import tiktoken
from openai import OpenAI

def tokenize(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)

def chunk_text(text, max_tokens, overlap_tokens):
    tokens = tokenize(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        if end > len(tokens):
            end = len(tokens)
        
        chunk = tokens[start:end]
        chunks.append(tiktoken.get_encoding("cl100k_base").decode(chunk))
        
        start = end - overlap_tokens  # Move start by subtracting overlap

    return chunks

def summarize_chunk(client, chunk, context=""):
    prompt = f"{context}\n\nSummarize the following text:\n\n{chunk}"
    response = client.chat.completions.create(
        model="llama3.1:latest",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an abstracted summary agent. Without commentary you summarise what you are presented with as a detailed paragraph, ensuring all details have been mentioned."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def hierarchical_summarize(client, text, chunk_size, overlap_size):
    chunks = chunk_text(text, chunk_size, overlap_size)
    summaries = []
    context = ""

    # First level of summarization
    for i in range(0, len(chunks), 2):
        pair = chunks[i:i+2]
        summary = summarize_chunk(client, "\n\n".join(pair), context)
        summaries.append(summary)
        context = summary  # Use the latest summary as context for the next pair

    # Continue summarizing until we have a single summary
    layer = 1
    while len(summaries) > 1:
        print(f"Layer {layer} summaries: {len(summaries)}")
        new_summaries = []
        for i in range(0, len(summaries), 2):
            pair = summaries[i:i+2]
            summary = summarize_chunk(client, "\n\n".join(pair))
            new_summaries.append(summary)
        summaries = new_summaries
        layer += 1

    print(f"Final layer: 1 summary")
    return summaries[0]

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Markdown Summarizer")
    parser.add_argument("file", help="Path to the Markdown file to summarize")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Maximum number of tokens per chunk")
    parser.add_argument("--overlap-size", type=int, default=200, help="Number of overlapping tokens between chunks")
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        text = f.read()

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    final_summary = hierarchical_summarize(client, text, args.chunk_size, args.overlap_size)
    print("Final Summary:")
    print(final_summary)

if __name__ == "__main__":
    main()