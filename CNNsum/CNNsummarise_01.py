import argparse
import tiktoken
from openai import OpenAI
from tqdm import tqdm
import os
import concurrent.futures
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

def tokenize(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)

def chunk_generator(file_path, max_tokens, overlap_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    buffer = []
    buffer_tokens = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            line_tokens = enc.encode(line)
            buffer.extend(line_tokens)
            buffer_tokens += len(line_tokens)
            
            while buffer_tokens >= max_tokens:
                chunk_tokens = buffer[:max_tokens]
                yield enc.decode(chunk_tokens)
                buffer = buffer[max_tokens - overlap_tokens:]
                buffer_tokens = len(buffer)
        
        # Yield any remaining content
        if buffer:
            yield enc.decode(buffer)

def summarize_chunk(client, chunk, context=""):
    system_instruction = """You are an AI assistant designed to create concise, factual summaries. Your task is to represent the key information from the given text without adding any commentary, context, or personal interpretation. Focus solely on the main points and essential details presented in the original text. Your summary should be:

1. Purely informational
2. Free of any added context or background information
3. Devoid of personal opinions or commentary
4. Concise and to the point
5. Faithful to the original content without introducing new ideas

Summarize the text in a way that someone reading your summary would get an accurate, condensed version of the original content."""

    user_prompt = f"Summarize the following text, focusing only on the key information presented:\n\n{chunk}"

    response = client.chat.completions.create(
        model="gemma2:2b",
        temperature=0,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# The rest of the script remains the same

def save_layer_summaries(summaries, output_file, layer):
    with open(f"{output_file}_layer_{layer:02d}.md", 'w') as f:
        for i, summary in enumerate(summaries, 1):
            f.write(f"## Summary {i}\n\n{summary}\n\n")

def summarize_pair(args):
    client, pair, context = args
    return summarize_chunk(client, "\n\n".join(pair), context)

def hierarchical_summarize(client, file_path, chunk_size, overlap_size, output_file):
    chunks = list(chunk_generator(file_path, chunk_size, overlap_size))
    summaries = []
    context = ""

    print(f"\n{Fore.CYAN}Chunking process:")
    print(f"{Fore.CYAN}Number of chunks created: {len(chunks)}")
    print(f"{Fore.CYAN}Number of chunk pairs to summarize: {(len(chunks) + 1) // 2}")

    print(f"\n{Fore.GREEN}Generation 1 (Initial Chunking):")
    for i, chunk in enumerate(chunks):
        print(f"{Fore.YELLOW}Chunk {i+1}:{Style.RESET_ALL} {chunk[:50]}...")

    print(f"\n{Fore.GREEN}Generation 2 (First Layer Summarization):")
    for i in tqdm(range(0, len(chunks), 2), desc="Summarizing chunks"):
        pair = chunks[i:i+2]
        summary = summarize_chunk(client, "\n\n".join(pair), context)
        summaries.append(summary)
        context = summary
        print(f"{Fore.YELLOW}Summary {i//2 + 1}:{Style.RESET_ALL} {summary[:100]}...")

    save_layer_summaries(summaries, output_file, 1)

    layer = 3
    while len(summaries) > 1:
        print(f"\n{Fore.GREEN}Generation {layer} (Layer {layer-1} Summarization):")
        print(f"{Fore.CYAN}Number of summaries to process: {len(summaries)}")
        
        new_summaries = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(summaries), 2):
                pair = summaries[i:i+2]
                futures.append(executor.submit(summarize_pair, (client, pair, "")))
            
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Summarizing layer {layer-1}")):
                summary = future.result()
                new_summaries.append(summary)
                print(f"{Fore.YELLOW}Summary {i+1}:{Style.RESET_ALL} {summary[:100]}...")
        
        save_layer_summaries(new_summaries, output_file, layer-1)
        summaries = new_summaries
        layer += 1

    return summaries[0]

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Markdown Summarizer")
    parser.add_argument("file", help="Path to the Markdown file to summarize")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Maximum number of tokens per chunk")
    parser.add_argument("--overlap-size", type=int, default=200, help="Number of overlapping tokens between chunks")
    args = parser.parse_args()

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    output_file = os.path.splitext(args.file)[0]
    final_summary = hierarchical_summarize(client, args.file, args.chunk_size, args.overlap_size, output_file)
    
    print(f"\n{Fore.GREEN}Final Summary:")
    print(f"{Fore.YELLOW}{final_summary}")

    with open(f"{output_file}_final_summary.md", 'w') as f:
        f.write(final_summary)

if __name__ == "__main__":
    main()