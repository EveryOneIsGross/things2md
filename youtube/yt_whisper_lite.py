import os
import re
import json
import argparse
import time
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import yaml
import logging
import traceback
import base64
from openai import OpenAI as OpenAIClient
from colorama import Fore, Style, init

# YouTube and video processing libraries
from pytube import YouTube
from yt_dlp import YoutubeDL
import cv2

# Audio processing libraries
from pydub import AudioSegment
import librosa

# Transcription and AI libraries
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, EncoderDecoderCache

from datasets import Dataset
import multiprocessing

import ollama

# Google Gemini API
import google.generativeai as genai
import PIL.Image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize colorama
init(autoreset=True)

def load_prompt_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def safe_filename(name: str) -> str:
    # Remove or replace problematic characters
    safe_name = re.sub(r'[^\w\-_\. ]', '', name)
    # Replace periods with underscores, except for the last one (file extension)
    safe_name = re.sub(r'\.(?=.*\.)', '_', safe_name)
    # Remove leading/trailing spaces and periods
    safe_name = safe_name.strip('. ')
    # Ensure the filename is not empty
    return safe_name if safe_name else 'untitled'

def setup_local_model(cache_dir: str = None, language: str = 'en'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logging.info(f"Using device: {device.upper()}")
    logging.info(f"Torch dtype: {torch_dtype}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    whisper = pipeline(
        "automatic-speech-recognition",
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"cache_dir": cache_dir} if cache_dir else {},
        generate_kwargs={"task": "translate", "language": language}
    )

    return whisper

def download_video(video_url: str, output_folder: str) -> tuple:
    with YoutubeDL() as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_title = info['title']
        safe_title = safe_filename(video_title)
        video_description = info.get('description', 'No description available')
        duration = info['duration']
        channel = info.get('channel', 'Unknown')

    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        'outtmpl': f'{output_folder}/{safe_title}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': True,
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    
    logging.info(f"Downloaded and processed: {safe_title}")
    
    return duration, safe_title, video_description, channel

def extract_transcript(video_url: str, output_folder: str, safe_title: str):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        video_id = YouTube(video_url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get the English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English is not available, try to get the auto-generated transcript
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except:
                # If neither is available, get the first available transcript
                transcript = transcript_list.find_transcript(['en'])
        
        srt_captions = ""
        for i, entry in enumerate(transcript.fetch(), start=1):
            start = format_time(entry['start'])
            end = format_time(entry['start'] + entry['duration'])
            text = entry['text'].replace('\n', ' ')
            srt_captions += f"{i}\n{start} --> {end}\n{text}\n\n"
        
        transcript_path = os.path.join(output_folder, f"{safe_title}_transcript.srt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(srt_captions)
        logging.info(f"Transcript saved for {safe_title}")
        return transcript_path
    except Exception as e:
        logging.error(f"Error extracting transcript for {safe_title}: {str(e)}")
        return None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def extract_frames_and_create_json(video_folder: str, safe_title: str, output_folder: str, num_frames: int):
    video_extensions = ['.mp4', '.webm', '.avi', '.mov']  # Add more if needed
    video_path = None
    
    for ext in video_extensions:
        temp_path = os.path.join(video_folder, f"{safe_title}{ext}")
        if os.path.exists(temp_path):
            video_path = temp_path
            break
    
    if not video_path:
        raise FileNotFoundError(f"No video file found for {safe_title}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_data = []
    
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # Create a separate folder for images
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    for frame_index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        
        if ret:
            timestamp = frame_index / fps
            frame_filename = f"{safe_title}_frame_{timestamp:.2f}.jpg"
            frame_path = os.path.join(images_folder, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            logging.info(f"Saved frame: {frame_path}")

            frame_data.append({
                "filename": frame_filename,
                "timestamp": timestamp
            })
    
    json_filename = f"{safe_title}_frame_data.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, 'w') as f:
        json.dump(frame_data, f, indent=4)
    
    video.release()
    logging.info(f"Extracted {len(frame_data)} frames. Frame data JSON created: {json_filename}")
    return json_path, frame_data

def segment_audio(file_path: str, max_chunk_size_mb: float = 24, overlap_duration_ms: int = 1000) -> List[str]:
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        logging.error(f"Error: Could not decode {file_path}. Please ensure it's a valid audio file. Error: {e}")
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

    logging.info(f"Number of segments: {len(temp_segment_files)}")
    return temp_segment_files

def transcribe_segment_api(segment_file: str, prompt: str, language: str, client) -> Dict[str, Any]:
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            with open(segment_file, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(segment_file, file.read()),  # Pass file as a tuple
                    model="whisper-1",
                    prompt=prompt,
                    language=language,
                    response_format="verbose_json",
                )
            
            print(f"{Fore.GREEN}API Transcription:{Style.RESET_ALL} {transcription.get('text', '')[:100]}...")
            
            # Log the raw API response for debugging
            logging.info(f"Raw API transcription output: {transcription}")

            # Check if the response is already a dictionary
            if isinstance(transcription, dict):
                return {
                    "text": transcription.get("text", ""),
                    "segments": transcription.get("segments", [])
                }
            # If it's not a dictionary, assume it's an object with attributes
            else:
                return {
                    "text": getattr(transcription, "text", ""),
                    "segments": getattr(transcription, "segments", [])
                }

        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"An error occurred: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed after {max_retries} attempts. Error: {e}")
                raise

def transcribe_segment_local(segment_file: str, whisper) -> Dict[str, Any]:
    result = whisper(segment_file, return_timestamps=True)
    
    transcription = result["text"]
    
    print(f"{Fore.CYAN}Local Transcription:{Style.RESET_ALL} {transcription[:100]}...")
    
    # Use the timestamps if available, otherwise use the full audio length
    if "chunks" in result:
        segments = [
            {
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1],
                "text": chunk["text"]
            } for chunk in result["chunks"]
        ]
    else:
        segments = [{
            "start": 0,
            "end": None,  # We don't know the end time without processing the audio file
            "text": transcription
        }]

    return {"text": transcription, "segments": segments}

def generate_markdown(transcription: Dict[str, Any], frame_descriptions: List[Dict[str, str]], output_file: str, metadata: Dict[str, Any]):
    with open(output_file, "w", encoding="utf-8") as md_file:
        md_file.write(f"# {metadata['title']}\n")
        md_file.write(f"## Channel: {metadata['channel']}\n")
        md_file.write(f"[Video URL]({metadata['url']})\n\n")
        
        md_file.write("## Description\n")
        md_file.write(f"{metadata.get('description', 'No description available')}\n\n")
        
        md_file.write("## Transcription\n")
        # Check if the transcription has a "text" key (API mode might not have it)
        if "text" in transcription:
            md_file.write(transcription["text"] + "\n\n")
        else:
            # Handle the case where "text" key is missing (e.g., API mode with different structure)
            md_file.write(transcription.get("transcript", "Transcription not available") + "\n\n")
        
        md_file.write("## Frame Descriptions\n")
        for frame in frame_descriptions:
            md_file.write(f"### Frame at {frame['timestamp']:.2f} seconds\n")
            md_file.write(f"{frame['description']}\n\n")

def process_image(image_path: str, frame_info: Dict[str, Any], video_title: str, previous_description: str, args, vision_client=None, openai_client=None) -> str:
    user_prompt = f"{args.content_string}\n\n" \
                  f"Video Title: {video_title}\n\n" \
                  f"Timestamp: {frame_info['timestamp']:.2f} seconds\n\n" \
                  f"Previous Frame Description: {previous_description if previous_description else 'Not available for first frame.'}\n\n" \
                  f"Describe the current image in detail, considering the context from the previous frame's description:"

    logging.info(f"Using prompt for {os.path.basename(image_path)}:\n{user_prompt}\n")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if args.vision_api == "google":
                image_description = process_image_google(image_path, user_prompt, vision_client)
            elif args.vision_api == "ollama":
                image_description = process_image_ollama(image_path, user_prompt)
            elif args.vision_api == "openai":
                image_description = process_image_openai(image_path, user_prompt, openai_client)
            else:
                raise ValueError(f"Unsupported vision API: {args.vision_api}")

            print(f"{Fore.YELLOW}Image Description ({args.vision_api}):{Style.RESET_ALL} {image_description[:100]}...")

            image_description = ' '.join(image_description.split())
            if image_description and not re.match(r'^\[[\d\., ]+\]$', image_description):
                logging.info(f"Generated description for {os.path.basename(image_path)}: {image_description}")
                return image_description
            else:
                logging.warning(f"Invalid response for {os.path.basename(image_path)}, attempt {attempt + 1} of {max_retries}")
        except Exception as e:
            logging.error(f"Error processing image {os.path.basename(image_path)}, attempt {attempt + 1} of {max_retries}: {str(e)}")
            logging.error(traceback.format_exc())
    
    logging.error(f"Failed to generate valid description for {os.path.basename(image_path)} after {max_retries} attempts.")
    return "Unable to process this image"

def process_image_google(image_path: str, user_prompt: str, vision_client) -> str:
    try:
        image = PIL.Image.open(image_path)
        response = vision_client.generate_content([user_prompt, image])
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error processing image with Gemini API: {str(e)}")
        return f"Error processing image: {str(e)}"

def process_image_ollama(image_path: str, user_prompt: str) -> str:
    try:
        res = ollama.chat(
            model="minicpm-v:latest",
            messages=[
                {
                    'role': 'system',
                    'content': "You are an AI assistant analyzing video frames."
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                    'images': [image_path]
                }
            ]
        )
        return res['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error processing image with Ollama: {str(e)}")
        return f"Error processing image: {str(e)}"

def process_image_openai(image_path: str, user_prompt: str, openai_client) -> str:
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = openai_client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error processing image with OpenAI GPT-4 Vision API: {str(e)}")
        return f"Error processing image: {str(e)}"


def process_video(video_url: str, base_output_folder: str, args):
    try:
        print(f"\n{Fore.BLUE}Processing video:{Style.RESET_ALL} {video_url}")
        
        # Create a folder for this video
        yt = YouTube(video_url)
        video_title = yt.title
        safe_title = safe_filename(video_title)
        video_folder = os.path.join(base_output_folder, safe_title)
        os.makedirs(video_folder, exist_ok=True)
        
        # Initialize audio_segments as an empty list
        audio_segments = []

        # Download video
        duration, safe_title, video_description, channel = download_video(video_url, video_folder)
        
        # Extract transcript
        transcript_path = extract_transcript(video_url, video_folder, safe_title)
        
        # Segment audio
        audio_path = os.path.join(video_folder, f"{safe_title}.mp3")
        if os.path.exists(audio_path):
            audio_segments = segment_audio(audio_path, args.max_chunk_size_mb, args.overlap_duration)
        else:
            logging.warning(f"Audio file not found: {audio_path}")

        # Transcribe audio segments
        full_transcription = []
        raw_transcriptions = []
        
        # Initialize clients based on the chosen mode and API
        if args.mode == "api":
            openai_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            whisper = setup_local_model(args.cache_dir, args.language)

        if args.vision_api == "google":
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            vision_client = genai.GenerativeModel('gemini-1.5-flash')
        elif args.vision_api == "openai":
            vision_client = openai_client if 'openai_client' in locals() else OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            vision_client = None

        if args.mode == "api":
            for i, segment_file in enumerate(audio_segments):
                print(f"{Fore.MAGENTA}Transcribing segment {i+1}/{len(audio_segments)} (API){Style.RESET_ALL}")
                transcription = transcribe_segment_api(segment_file, args.prompt, args.language, openai_client)
                raw_transcriptions.append(transcription)
                full_transcription.append(transcription)
        else:
            for i, segment_file in enumerate(audio_segments):
                print(f"{Fore.MAGENTA}Transcribing segment {i+1}/{len(audio_segments)} (Local){Style.RESET_ALL}")
                transcription = transcribe_segment_local(segment_file, whisper)
                full_transcription.append(transcription)
        
        # Extract frames
        frame_data_path, frame_data = extract_frames_and_create_json(video_folder, safe_title, video_folder, args.num_frames)

        # Combine transcriptions
        combined_transcription = {
            "text": " ".join([t["text"] for t in full_transcription]),
            "segments": [s for t in full_transcription for s in t["segments"]]
        }

        # Process frames
        frame_descriptions = []
        previous_description = None
        for frame in frame_data:
            image_path = os.path.join(video_folder, "images", frame['filename'])
            description = process_image(
                image_path, 
                frame, 
                video_title, 
                previous_description,
                args,
                vision_client,
                vision_client if args.vision_api == "openai" else None
            )
            frame_descriptions.append({
                "timestamp": frame['timestamp'],
                "description": description
            })
            previous_description = description

        # Generate final markdown
        metadata = {
            "title": video_title,
            "channel": channel,
            "url": video_url,
            "description": video_description
        }
        markdown_path = os.path.join(video_folder, f"{safe_title}_final_output.md")
        generate_markdown(combined_transcription, frame_descriptions, markdown_path, metadata)

        # Save all output files
        transcription_json_path = os.path.join(video_folder, f"{safe_title}_transcription.json")
        with open(transcription_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_transcription, f, ensure_ascii=False, indent=2)

        frame_descriptions_json_path = os.path.join(video_folder, f"{safe_title}_frame_descriptions.json")
        with open(frame_descriptions_json_path, "w", encoding="utf-8") as f:
            json.dump(frame_descriptions, f, ensure_ascii=False, indent=2)

        # Save raw transcriptions (API mode only)
        if args.mode == "api":
            raw_transcription_json_path = os.path.join(video_folder, f"{safe_title}_raw_transcription.json")
            with open(raw_transcription_json_path, "w", encoding="utf-8") as f:
                json.dump(raw_transcriptions, f, ensure_ascii=False, indent=2)
            logging.info(f"Raw transcription saved to: {raw_transcription_json_path}")

        logging.info(f"Processing completed for {video_title}")
        logging.info(f"Output files saved in: {video_folder}")

    except Exception as e:
        print(f"{Fore.RED}Error processing video {video_url}: {str(e)}{Style.RESET_ALL}")
        logging.error(traceback.format_exc())

    finally:
        # Clean up temporary files
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)

def main():
    parser = argparse.ArgumentParser(description="Process YouTube videos: download, transcribe, and analyze frames.")
    parser.add_argument("input_file", help="Path to the text file containing YouTube URLs")
    parser.add_argument("output_folder", help="Path to the base output folder")
    parser.add_argument("--mode", choices=["api", "local"], default="local", help="Transcription mode: 'api' or 'local' (default: local)")
    parser.add_argument("--prompt", default="The following is a YouTube video transcript.", help="Prompt for the transcription (API mode only)")
    parser.add_argument("--language", default="en", help="Language of the audio (default: en)")
    parser.add_argument("--max_chunk_size_mb", type=float, default=24, help="Maximum chunk size in MB (default: 24)")
    parser.add_argument("--cache_dir", help="Directory to cache the model (local mode only)")
    parser.add_argument("--overlap_duration", type=int, default=1000, help="Overlap duration between segments in milliseconds (default: 1000)")
    parser.add_argument("--num_frames", type=int, default=64, help="Number of frames to extract from each video (default: 64)")
    parser.add_argument("--content_string", default="The following image is a frame from a YouTube video. Describe the image in detail, noting relevant visual elements and how it relates to the video context:", help="Custom content string for image description prompt")
    parser.add_argument("--system_prompt", default="You are an AI assistant specialized in analyzing and describing images from video frames. Your task is to provide detailed, accurate, and context-aware descriptions of each frame, considering the video title and previous frame descriptions when available. Focus on visual elements and pay attention to how the image relates to the provided context. Be concise yet comprehensive in your descriptions.", help="System prompt for the image description model")
    parser.add_argument("--prompt_config", help="Path to YAML file containing prompt configurations")
    parser.add_argument("--vision_api", choices=["ollama", "google", "openai"], default="ollama", help="Vision API to use for image processing (default: ollama)")
    
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise ValueError(f"Input file not found: {args.input_file}")

    os.makedirs(args.output_folder, exist_ok=True)

    with open(args.input_file, 'r') as file:
        video_urls = file.read().splitlines()

    for url in video_urls:
        process_video(url.strip(), args.output_folder, args)

    logging.info("All videos processed successfully!")

if __name__ == "__main__":
    main()