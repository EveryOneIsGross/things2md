import os
import re
import argparse
import ollama
import yaml
import tiktoken
from PIL import Image
import numpy as np
import time
import asyncio
from colorama import init, Fore, Style
import nltk
from nltk.tokenize import sent_tokenize

# Initialize colorama
init(autoreset=True)

# Download the punkt tokenizer for sentence splitting
nltk.download('punkt', quiet=True)

DEFAULT_PROMPT_WITH_CONTEXT = """
Describe the image in detail, noting its key features, any text present, and its overall composition. 
Consider the context provided before and after the image:

Context before the image:
{before_context}

Context after the image:
{after_context}

Please provide a concise yet comprehensive description of the image based on this context and the image itself.
"""

DEFAULT_PROMPT_WITHOUT_CONTEXT = """
Describe the image in detail, noting its key features, any text present, and its overall composition. 
Please provide a concise yet comprehensive description of the image based solely on its visual content.
"""

def load_prompt_templates(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            templates = yaml.safe_load(file)
            return templates.get('prompt_with_context', DEFAULT_PROMPT_WITH_CONTEXT), \
                   templates.get('prompt_without_context', DEFAULT_PROMPT_WITHOUT_CONTEXT)
    except (FileNotFoundError, KeyError):
        print(f"{Fore.YELLOW}Warning: Could not load prompt templates from {yaml_file}. Using default prompts.{Style.RESET_ALL}")
        return DEFAULT_PROMPT_WITH_CONTEXT, DEFAULT_PROMPT_WITHOUT_CONTEXT

def get_surrounding_text(content, match_start, match_end, num_tokens=64, threshold=0.1):
    encoding = tiktoken.get_encoding("cl100k_base")
    
    before_text = content[:match_start]
    after_text = content[match_end:]
    
    before_sentences = sent_tokenize(before_text)
    after_sentences = sent_tokenize(after_text)
    
    before_context = ""
    after_context = ""
    
    threshold_tokens = int(num_tokens * threshold)
    
    # Get context before the image
    if before_sentences:
        for sentence in reversed(before_sentences):
            temp_context = sentence + " " + before_context
            tokens_count = len(encoding.encode(temp_context))
            if tokens_count > num_tokens:
                if tokens_count <= num_tokens + threshold_tokens:
                    before_context = temp_context  # Include this sentence if it's within the threshold
                break
            before_context = temp_context
    
    # Get context after the image
    if after_sentences:
        for sentence in after_sentences:
            temp_context = after_context + " " + sentence
            tokens_count = len(encoding.encode(temp_context))
            if tokens_count > num_tokens:
                if tokens_count <= num_tokens + threshold_tokens:
                    after_context = temp_context  # Include this sentence if it's witshin the threshold
                break
            after_context = temp_context
    
    return before_context.strip(), after_context.strip()

async def ollama_chat_with_timeout(model, messages, timeout_minutes):
    try:
        print(f"{Fore.CYAN}Sending prompt to model:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{messages[0]['content']}{Style.RESET_ALL}")
        return await asyncio.wait_for(
            asyncio.to_thread(ollama.chat, model=model, messages=messages),
            timeout=timeout_minutes * 60
        )
    except asyncio.TimeoutError:
        raise TimeoutError("API call timed out")

def process_markdown_file(file_path, image_folder, prompt_with_context, prompt_without_context, use_context, size_threshold=64, timeout_minutes=2, max_retries=3):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    original_content = content  # Store the original content for context extraction

    # Find all image links
    image_links = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', content))
    
    # Process each image link
    for i, match in enumerate(image_links):
        is_first = (i == 0)
        is_last = (i == len(image_links) - 1)
        
        full_match = match.group(0)
        current_description = match.group(1)
        image_filename = match.group(2)
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"{Fore.YELLOW}Image not found: {image_filename}. Skipping.{Style.RESET_ALL}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size
        
        if width < size_threshold or height < size_threshold:
            print(f"{Fore.RED}Removing small image: {image_filename} ({width}x{height} pixels){Style.RESET_ALL}")
            content = content.replace(full_match, "", 1)
            continue

        if use_context:
            before_context, after_context = get_surrounding_text(original_content, match.start(), match.end(), num_tokens=128, threshold=0.1)
            
            if is_first:
                before_context = ""
            if is_last:
                after_context = ""
            
            prompt = prompt_with_context.format(
                before_context=before_context,
                after_context=after_context
            )
        else:
            prompt = prompt_without_context

        new_description = None
        for attempt in range(max_retries):
            try:
                res = asyncio.run(ollama_chat_with_timeout(
                    model="minicpm-v:latest",
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [image_path]
                        }
                    ],
                    timeout_minutes=timeout_minutes
                ))
                new_description = res['message']['content'].strip()
                new_description = ' '.join(new_description.split())

                if new_description and not re.match(r'^\[[\d\., ]+\]$', new_description):
                    print(f"{Fore.GREEN}Generated description for {image_filename}: {new_description}{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.YELLOW}Invalid response for {image_filename}, attempt {attempt + 1} of {max_retries}{Style.RESET_ALL}")
                    new_description = None
            except TimeoutError:
                print(f"{Fore.RED}API call timed out for {image_filename}, attempt {attempt + 1} of {max_retries}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error processing {image_filename}: {str(e)}, attempt {attempt + 1} of {max_retries}{Style.RESET_ALL}")
            
            time.sleep(1)  # Short delay between retries

        if new_description is None:
            while True:
                user_choice = input(f"{Fore.YELLOW}Failed to process {image_filename}. Do you want to (r)etry, (s)kip, or (t)erminate? {Style.RESET_ALL}").lower()
                if user_choice == 'r':
                    break  # This will continue the outer loop, retrying the current image
                elif user_choice == 's':
                    print(f"{Fore.YELLOW}Skipping {image_filename}{Style.RESET_ALL}")
                    new_description = current_description  # Keep the current description
                    break
                elif user_choice == 't':
                    print(f"{Fore.RED}Terminating script.{Style.RESET_ALL}")
                    exit(0)
                else:
                    print(f"{Fore.RED}Invalid choice. Please enter 'r', 's', or 't'.{Style.RESET_ALL}")

        if new_description is not None:
            new_match = f"![{new_description}]({image_filename})"
            content = content.replace(full_match, new_match, 1)

    updated_content = content.strip()

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

def process_folder(folder_path, prompt_with_context, prompt_without_context, use_context, timeout_minutes=2, max_retries=3):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                process_markdown_file(file_path, root, prompt_with_context, prompt_without_context, use_context, timeout_minutes=timeout_minutes, max_retries=max_retries)

def main():
    parser = argparse.ArgumentParser(description="Process markdown files in a folder and its subfolders to update image descriptions.")
    parser.add_argument("folder_path", help="Path to the folder containing markdown files")
    parser.add_argument("--yaml_file", default="prompt_template.yaml", help="Path to the YAML file containing the prompt templates")
    parser.add_argument("--use_context", action="store_true", help="Use surrounding text context in the prompt")
    parser.add_argument("--timeout", type=int, default=2, help="Timeout for API calls in minutes")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for API calls")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"{Fore.RED}Error: {args.folder_path} is not a valid directory.{Style.RESET_ALL}")
        return

    prompt_with_context, prompt_without_context = load_prompt_templates(args.yaml_file)
    process_folder(args.folder_path, prompt_with_context, prompt_without_context, args.use_context, args.timeout, args.max_retries)
    print(f"{Fore.GREEN}Processing complete.{Style.RESET_ALL}")

if __name__ == '__main__':
    main()