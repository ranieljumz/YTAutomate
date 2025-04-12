import os
from typing import List
from pydantic import BaseModel
from moviepy import *
from moviepy.video.tools.subtitles import SubtitlesClip
import whisper
from whisper.utils import get_writer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
import math

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# Font settings - using Arial Bold as it's widely available and readable
FONT_PATH = "C:/WINDOWS/FONTS/lsans.ttf"  # You may need to specify full path or use another font
FONT_SIZE = 80
TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "yellow"
BG_COLOR = (0, 0, 0, 150)  # Semi-transparent black
TEXT_PADDING = 0.05  # 10% padding
MAX_ZOOM = 1.1  # 120% zoom
WORD_SPACING = 0.3  # 30% of font size as spacing between words
LINE_SPACING = 1.2  # 120% of font size as line height

def generate_subtitles(audio_path: str) -> str:
    """Generate SRT subtitles using Whisper"""
    model = whisper.load_model("medium")  # Use 'small' or 'medium' for better quality
    result = model.transcribe(audio_path)
    
    # Save as SRT
    srt_path = os.path.splitext(audio_path)[0] + ".srt"
    srt_writer = get_writer("srt", os.path.dirname(audio_path))
    srt_writer(result, srt_path)
    
    return srt_path

def create_composite_video(image_paths: list[str], audio_path: str, output_path: str):
    """Create the final video with all effects"""
    
    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration
    
    subtitle_path = generate_subtitles(audio_path)
    # Calculate image display durations
    num_images = len(image_paths)
    image_duration = audio_duration / num_images
    
    # Create zooming image clips
    image_clips = []
    for i, img_path in enumerate(image_paths):
        # Create zoom effect
        img = ImageClip(img_path).with_duration(image_duration)
        image_clip = img.with_effects([vfx.Resize((1080,1920))])
        
        # Add zoom-in effect
        zoom_factor = lambda t: 1 + (MAX_ZOOM - 1) * min(t/image_duration, 1)
        zoomed_clip = image_clip.with_effects([vfx.Resize((zoom_factor))])
        
        # Set position in timeline
        zoomed_clip = zoomed_clip.with_start(i * image_duration)
        image_clips.append(zoomed_clip)
    
    # Create composite of all image clips
    video = CompositeVideoClip(image_clips, size=(1080,1920))
    
    # Add subtitles with karaoke effect
    subtitle_clip = create_karaoke_subtitles(subtitle_path, video.size)
    video = CompositeVideoClip([video,subtitle_clip])
    
    # Set audio
    video = video.with_audio(audio)
    
    # Write output
    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        threads=10,
        preset="ultrafast",
    )

def create_karaoke_subtitles(srt_path: str, video_size: tuple) -> CompositeVideoClip:
    """Create subtitles with karaoke-style word highlighting"""
    # Read SRT file
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Parse SRT
    subtitles = parse_srt(srt_content)
    
    # Create subtitle clips
    subtitle_clips = []
    
    for sub in subtitles:
        # Split text into words with timings
        words = split_words_with_timing(sub['text'], sub['start'], sub['end'])
        
        # Create text clips for each word group that fits on screen
        word_groups = group_words_to_fit(words, video_size)
        
        for group in word_groups:
            # Create text clip with highlighting
            txt_clip = create_highlighted_text_clip(
                group['words'],
                group['start'],
                group['end'],
                video_size
            )
            subtitle_clips.append(txt_clip)
    
    return concatenate_videoclips(subtitle_clips)

def parse_srt(srt_content: str) -> list[dict]:
    """Parse SRT content into list of subtitle entries"""
    entries = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            idx = lines[0]
            times = lines[1]
            text = '\n'.join(lines[2:])
            
            # Parse time
            start_end = times.split(' --> ')
            if len(start_end) == 2:
                start = srt_time_to_seconds(start_end[0])
                end = srt_time_to_seconds(start_end[1])
                
                entries.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
    
    return entries

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def split_words_with_timing(text: str, start: float, end: float) -> list[dict]:
    """Split text into words with estimated timing"""
    words = text.split()
    if not words:
        return []
    
    duration = end - start
    word_duration = duration / len(words)
    
    word_timings = []
    for i, word in enumerate(words):
        word_start = start + i * word_duration
        word_end = word_start + word_duration
        word_timings.append({
            'text': word,
            'start': word_start,
            'end': word_end
        })
    
    return word_timings

def group_words_to_fit(words: List[dict], video_size: tuple) -> List[dict]:
    """Group words into lines that fit on screen with proper spacing"""
    max_width = video_size[0] * (1 - 2 * TEXT_PADDING)
    groups = []
    current_line = []
    current_width = 0
    
    for word in words:
        # Estimate word width (we'll calculate exactly later)
        estimated_width = len(word['text']) * FONT_SIZE * 0.6
        
        if current_line and (current_width + estimated_width + (FONT_SIZE * WORD_SPACING)) > max_width:
            # Finish current line
            groups.append({
                'words': current_line,
                'start': current_line[0]['start'],
                'end': current_line[-1]['end']
            })
            current_line = []
            current_width = 0
            
        # Add word to line
        if current_line:
            current_width += FONT_SIZE * WORD_SPACING  # Add spacing
            
        current_line.append(word)
        current_width += estimated_width
    
    # Add last line
    if current_line:
        groups.append({
            'words': current_line,
            'start': current_line[0]['start'],
            'end': current_line[-1]['end']
        })
    
    return groups

def create_highlighted_text_clip(words: List[dict], start: float, end: float, 
                               video_size: tuple) -> TextClip:
    """Create a text clip with proper word spacing and karaoke-style highlighting"""
    # Calculate position with padding
    padding_x = video_size[0] * TEXT_PADDING
    padding_y = video_size[1] * TEXT_PADDING
    position_y = video_size[1] - padding_y - FONT_SIZE * 2  # Bottom of screen
    
    # Create a function that will generate each frame
    def make_frame(t):
        # Convert relative time to absolute time
        abs_time = start + t
        
        # Create blank image with transparent background
        img = Image.new('RGBA', video_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except:
            font = ImageFont.load_default()
        
        # Calculate total text width and prepare word positions
        x_offset = position_x = padding_x
        word_objects = []
        
        for word in words:
            # Get word bounding box
            left, top, right, bottom = draw.textbbox((0, 0), word['text'], font=font)
            word_width = right - left
            word_height = bottom - top
            
            word_objects.append({
                'text': word['text'],
                'width': word_width,
                'start': word['start'],
                'end': word['end'],
                'x': x_offset,
                'y': position_y
            })
            
            # Add space after each word (except last one)
            if word != words[-1]:
                x_offset += word_width + FONT_SIZE * 0.3  # 30% of font size as spacing
        
        # Calculate total width for background
        #total_width = x_offset - position_x + padding_x
        #max_height = FONT_SIZE * 1.2  # Slightly more than font height
        
        # Draw background rectangle (semi-transparent)
        #bg_position = (
        #    position_x - padding_x/2,
        #    position_y - padding_y/2,
        #   position_x + total_width,
        #    position_y + max_height
        #)
        #draw.rectangle(bg_position, fill=BG_COLOR)
        
        # Draw each word with proper spacing and highlighting
        for word in word_objects:
            # Check if we should highlight this word
            highlight = word['start'] <= abs_time < word['end']
            color = HIGHLIGHT_COLOR if highlight else TEXT_COLOR
            
            # Draw word at calculated position
            draw.text((word['x'], word['y']), word['text'], fill=color, font=font)
        
        return np.array(img)
    
    # Create clip
    duration = end - start
    clip = VideoClip(make_frame, duration=duration)
    clip = clip.with_start(start)
    
    return clip
    
image_paths = ['Z:/AutoCut/examples/images/1.png','Z:/AutoCut/examples/images/2.png','Z:/AutoCut/examples/images/3.png','Z:/AutoCut/examples/images/4.png','Z:/AutoCut/examples/images/5.png','Z:/AutoCut/examples/images/6.png', 'Z:/AutoCut/examples/images/7.png']
audio_path = "Z:/AutoCut/examples/audio/1743274933.4666355.wav"
output_path = "Z:/AutoCut/outputs/output.mp4"

def main():
    global app
    print("Starting app...")
    create_composite_video(image_paths,audio_path,output_path)


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()