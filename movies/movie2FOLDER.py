import os
import json
import cv2
import numpy as np
from srt import parse as srt_parse

def safe_filename(name):
    return ''.join(c for c in name if c.isalnum() or c in (' ', '_', '-', '.')).strip()

def extract_frames_and_create_json(video_path, srt_path, output_folder, safe_title, num_frames, context_window=60):
    # Create 'images' subfolder
    images_folder = os.path.join(output_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    subtitle_list = list(srt_parse(srt_content))
    
    frame_data = []
    
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    last_end_index = -1  # Keep track of the last subtitle index we've used
    
    for frame_index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        
        if ret:
            timestamp = frame_index / fps
            frame_filename = f"{safe_title}_frame_{timestamp:.2f}.png"
            cv2.imwrite(os.path.join(images_folder, frame_filename), frame)
            
            # Get non-overlapping text chunk
            text_chunk, last_end_index = get_non_overlapping_text_chunk(subtitle_list, timestamp, context_window, last_end_index)
            
            frame_data.append({
                "filename": frame_filename,  # Changed to just filename, as it's now in the same folder
                "timestamp": timestamp,
                "transcript_chunk": text_chunk
            })
    
    # Save the frame data JSON in the images folder
    json_filename = f"{safe_title}_frame_data.json"
    with open(os.path.join(images_folder, json_filename), 'w', encoding='utf-8') as f:
        json.dump(frame_data, f, indent=4, ensure_ascii=False)
    
    video.release()
    print(f"Extracted {num_frames} frames and created JSON: {json_filename}")

    # Create Markdown in the images folder
    #srt_to_markdown(frame_data, images_folder, safe_title)

def get_non_overlapping_text_chunk(subtitle_list, timestamp, context_window, last_end_index):
    nearest_index = find_nearest_timestamp(subtitle_list, timestamp)
    
    # Start from where we left off last time, or from the nearest index minus the context window
    start_index = max(last_end_index + 1, nearest_index - context_window)
    end_index = min(len(subtitle_list), nearest_index + context_window + 1)
    
    text_chunk = []
    for sub in subtitle_list[start_index:end_index]:
        text_chunk.append(f"[{format_time(sub.start.total_seconds())} - {format_time(sub.end.total_seconds())}] {sub.content}")
    
    return "\n".join(text_chunk), end_index - 1  # Return the text chunk and the new last_end_index

def find_nearest_timestamp(subtitle_list, target_time):
    return min(range(len(subtitle_list)), key=lambda i: abs(subtitle_list[i].start.total_seconds() - target_time))

def srt_to_markdown(frame_data, output_folder, safe_title):
    markdown_content = f"# {safe_title} Transcript\n\n"
    
    for frame in frame_data:
        timestamp = frame['timestamp']
        subtitle_chunk = frame['transcript_chunk']
        markdown_content += f"## Frame at [{format_time(timestamp)}]\n\n{subtitle_chunk}\n\n"
    
    md_filename = f"{safe_title}_transcript.md"
    with open(os.path.join(output_folder, md_filename), 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Created Markdown transcript: {md_filename}")

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

def main():
    video_path = input("Enter the path to the video file: ")
    srt_path = input("Enter the path to the SRT file: ")
    output_folder = input("Enter the output folder path: ")
    num_frames = int(input("Enter the number of frames to extract: "))
    context_window = int(input("Enter the number of subtitle entries to include before and after each frame (default is 60): ") or "60")
    
    safe_title = safe_filename(os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_folder, exist_ok=True)
    
    extract_frames_and_create_json(video_path, srt_path, output_folder, safe_title, num_frames, context_window)

if __name__ == "__main__":
    main()