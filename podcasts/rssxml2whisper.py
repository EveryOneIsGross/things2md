import re
import json
import requests
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
from pydub import AudioSegment
from tqdm import tqdm
import tempfile

def extract_fields(content):
    items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
    
    json_data = []
    for item in items:
        episode_data = {
            'title': extract_tag(item, 'title'),
            'enclosure': {
                'url': extract_attribute(item, 'enclosure', 'url'),
                'length': extract_attribute(item, 'enclosure', 'length'),
                'type': extract_attribute(item, 'enclosure', 'type')
            },
            'description': extract_tag(item, 'description'),
            'keywords': extract_tag(item, 'itunes:keywords'),
            'summary': extract_tag(item, 'itunes:summary'),
            'people': extract_people(item)
        }
        json_data.append(episode_data)
    
    return json_data

def extract_tag(text, tag):
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else ''

def extract_attribute(text, tag, attr):
    match = re.search(f'<{tag}[^>]*{attr}="([^"]*)"', text)
    return match.group(1) if match else ''

def extract_people(text):
    people = []
    matches = re.findall(r'<podcast:person([^>]*)>(.*?)</podcast:person>', text)
    for attrs, name in matches:
        role = re.search(r'role="([^"]*)"', attrs)
        href = re.search(r'href="([^"]*)"', attrs)
        people.append({
            'name': name.strip(),
            'role': role.group(1) if role else '',
            'href': href.group(1) if href else ''
        })
    return people

def parse_xml_to_json(xml_file):
    with open(xml_file, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    return extract_fields(content)

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

def download_media(json_data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for item in json_data:
        title = sanitize_filename(item['title'])
        url = item['enclosure']['url']
        file_extension = os.path.splitext(url.split('?')[0])[1]  # Remove query parameters
        
        file_path = os.path.join(output_folder, f"{title}{file_extension}")
        
        print(f"Downloading: {title}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Saved: {file_path}")
        
        # Add the file path to the item data
        item['file_path'] = file_path

def setup_local_model(cache_dir=None, language='en'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device.upper()}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_id = "openai/whisper-large-v3"

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")
    else:
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        print(f"Using default cache directory: {cache_dir}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        cache_dir=cache_dir
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=25,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe, processor, device, torch_dtype

def transcribe_segment_local(pipe, processor, device, torch_dtype, segment_file):
    audio, sr = librosa.load(segment_file, sr=16000)
    result = pipe(audio)
    transcription = result["text"]
    segments = [{
        "start": 0,
        "end": len(audio) / sr,
        "text": transcription
    }]
    return {"text": transcription, "segments": segments}

def segment_audio(file_path, max_chunk_size_mb=24, overlap_duration_ms=1000):
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"Error: Could not decode {file_path}. Please ensure it's a valid audio file. Error: {e}")
        return []

    audio = audio.set_channels(1).set_frame_rate(16000)
    duration_ms = len(audio)
    bytes_per_ms = audio.frame_width * audio.frame_rate / 1000
    max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024

    temp_segment_files = []
    start_ms = 0
    
    with tqdm(total=duration_ms, desc="Segmenting audio") as pbar:
        while start_ms < duration_ms:
            chunk_duration_ms = max_chunk_size_bytes / bytes_per_ms
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            
            segment = audio[start_ms:end_ms]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                segment.export(temp_file.name, format="wav")
                temp_segment_files.append(temp_file.name)
            
            if end_ms >= duration_ms:
                break
            
            start_ms = max(start_ms + chunk_duration_ms - overlap_duration_ms, 0)
            pbar.update(end_ms - start_ms)

    return temp_segment_files

def transcribe_audio(pipe, processor, device, torch_dtype, audio_file):
    segments = segment_audio(audio_file)
    full_transcription = ""
    
    for segment in tqdm(segments, desc="Transcribing segments"):
        result = transcribe_segment_local(pipe, processor, device, torch_dtype, segment)
        full_transcription += result["text"] + " "
        os.unlink(segment)  # Delete temporary segment file
    
    return full_transcription.strip()

def create_markdown_files(json_data, output_folder, pipe, processor, device, torch_dtype):
    os.makedirs(output_folder, exist_ok=True)

    for episode in json_data:
        title = episode['title']
        safe_title = sanitize_filename(title)
        
        # Transcribe the audio
        print(f"Transcribing: {title}")
        transcript = transcribe_audio(pipe, processor, device, torch_dtype, episode['file_path'])
        
        # Create markdown content
        markdown_content = f"""# {title}

## Description
{episode['description']}

## Keywords
{episode['keywords']}

## People
"""
        for person in episode['people']:
            markdown_content += f"- **{person['name']}** (Role: {person['role']}, Link: {person['href']})\n"

        markdown_content += f"\n## Audio File\n[{safe_title}.mp3](./{safe_title}.mp3)\n"
        
        markdown_content += f"\n## Transcript\n{transcript}\n"

        # Write markdown file
        md_filename = os.path.join(output_folder, f"{safe_title}.md")
        with open(md_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)

        print(f"Created: {md_filename}")

# Usage
xml_file = 'podcast_rss\podcasts.xml'  # Replace with your input XML file name
json_file = 'podcast_rss\podcasts.json'
media_folder = "PODCASTS"
markdown_folder = "PODCASTS"

try:
    # Parse XML and save to JSON
    parsed_data = parse_xml_to_json(xml_file)
    save_to_json(parsed_data, json_file)
    print(f"JSON database has been created: {json_file}")
    print(f"Processed {len(parsed_data)} items")

    # Download media files
    download_media(parsed_data, media_folder)
    print("Media download completed successfully.")

    # Set up the transcription model
    pipe, processor, device, torch_dtype = setup_local_model()

    # Create markdown files with transcripts
    create_markdown_files(parsed_data, markdown_folder, pipe, processor, device, torch_dtype)
    print("Markdown files with transcripts created successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")