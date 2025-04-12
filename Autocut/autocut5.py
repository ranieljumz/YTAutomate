import json
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
import re

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# Font settings - using Arial Bold as it's widely available and readable
FONT_PATH = "C:/Users/ranie/AppData/Local/Microsoft/Windows/Fonts/AsapCondensed-Medium.ttf"  # You may need to specify full path or use another font
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
    result = model.transcribe(audio_path, word_timestamps=True)

    # Save as SRT
    wordlevel_info = []

    for each in result["segments"]:
        words = each["words"]
        for word in words:
            # print (word['word'], "  ",word['start']," - ",word['end'])
            wordlevel_info.append(
                {
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                }
            )

    with open("data.json", "w") as f:
        json.dump(wordlevel_info, f, indent=4)
    with open("data.json", "r") as f:
        wordlevel_info_modified = json.load(f)
    return wordlevel_info_modified


def split_text_into_lines(data):

    MaxChars = 80
    # maxduration in seconds
    MaxDuration = 3.0
    # Split if nothing is spoken (gap) for these many seconds
    MaxGap = 1.5

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0

    for idx, word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)

        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx > 0:
            gap = word_data["start"] - data[idx - 1]["end"]
            # print (word,start,end,gap)
            maxgap_exceeded = gap > MaxGap
        else:
            maxgap_exceeded = False

        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line,
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0

    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line,
        }
        subtitles.append(subtitle_line)

    return subtitles


def create_caption(
    textJSON,
    framesize,
    font=FONT_PATH,
    fontsize=FONT_SIZE,
    color="white",
    bgcolor="blue",
):
    wordcount = len(textJSON["textcontents"])
    full_duration = textJSON["end"] - textJSON["start"]

    word_clips = []
    xy_textclips_positions = []

    x_pos = 0
    y_pos = framesize[1] / 2
    # max_height = 0
    frame_width = framesize[0]
    frame_height = framesize[1]
    x_buffer = frame_width * 1 / 10
    y_buffer = frame_height * 1 / 5

    space_width = ""
    space_height = ""

    for index, wordJSON in enumerate(textJSON["textcontents"]):
        duration = wordJSON["end"] - wordJSON["start"]
        word_clip = (
            TextClip(text=wordJSON["word"], font=font, font_size=fontsize, color=color)
            .with_start(textJSON["start"])
            .with_duration(full_duration)
        )
        word_clip_space = (
            TextClip(text=" ", font=font, font_size=fontsize, color=color)
            .with_start(textJSON["start"])
            .with_duration(full_duration)
        )
        word_width, word_height = word_clip.size
        space_width, space_height = word_clip_space.size
        if x_pos + word_width + space_width > frame_width - 2 * x_buffer:
            # Move to the next line
            x_pos = 0
            y_pos = y_pos + word_height + 40

            # Store info of each word_clip created
            xy_textclips_positions.append(
                {
                    "x_pos": x_pos + x_buffer,
                    "y_pos": y_pos + y_buffer,
                    "width": word_width,
                    "height": word_height,
                    "word": wordJSON["word"],
                    "start": wordJSON["start"],
                    "end": wordJSON["end"],
                    "duration": duration,
                }
            )

            word_clip = word_clip.with_position((x_pos + x_buffer, y_pos + y_buffer))
            word_clip_space = word_clip_space.with_position(
                (x_pos + word_width + x_buffer, y_pos + y_buffer)
            )
            x_pos = word_width + space_width
        else:
            # Store info of each word_clip created
            xy_textclips_positions.append(
                {
                    "x_pos": x_pos + x_buffer,
                    "y_pos": y_pos + y_buffer,
                    "width": word_width,
                    "height": word_height,
                    "word": wordJSON["word"],
                    "start": wordJSON["start"],
                    "end": wordJSON["end"],
                    "duration": duration,
                }
            )

            word_clip = word_clip.with_position((x_pos + x_buffer, y_pos + y_buffer))
            word_clip_space = word_clip_space.with_position(
                (x_pos + word_width + x_buffer, y_pos + y_buffer)
            )

            x_pos = x_pos + word_width + space_width

        word_clips.append(word_clip)
        word_clips.append(word_clip_space)

    for highlight_word in xy_textclips_positions:

        word_clip_highlight = (
            TextClip(
                text=highlight_word["word"],
                font=font,
                font_size=fontsize,
                color="yellow",
                stroke_color="black",
                stroke_width=1,
            )
            .with_start(highlight_word["start"])
            .with_duration(highlight_word["duration"])
        )
        word_clip_highlight = word_clip_highlight.with_position(
            (highlight_word["x_pos"], highlight_word["y_pos"])
        )
        word_clips.append(word_clip_highlight)

    return word_clips


def create_composite_video(image_paths: list[str], audio_path: str, output_path: str):
    """Create the final video with all effects"""

    audio = AudioFileClip(audio_path)
    audio_duration = audio.duration

    wordlevel_info_modified = generate_subtitles(audio_path)

    linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)

    all_linelevel_splits = []
    for line in linelevel_subtitles:
        out = create_caption(line, (1080, 1920))
        all_linelevel_splits.extend(out)

    # Calculate image display durations
    num_images = len(image_paths)
    image_duration = audio_duration / num_images

    # Create zooming image clips
    image_clips = []
    for i, img_path in enumerate(image_paths):
        # Create zoom effect
        img = ImageClip(img_path).with_duration(image_duration)
        image_clip = img.with_effects([vfx.Resize((1080, 1920))])

        # Add zoom-in effect
        zoom_factor = lambda t: 1 + (MAX_ZOOM - 1) * min(t / image_duration, 1)
        zoomed_clip = image_clip.with_effects([vfx.Resize((zoom_factor))])

        # Set position in timeline
        zoomed_clip = zoomed_clip.with_start(i * image_duration)
        image_clips.append(zoomed_clip)

    # Create composite of all image clips
    video = CompositeVideoClip(image_clips, size=(1080, 1920))

    # Add subtitles with karaoke effect
    # subtitle_clip = create_karaoke_subtitles(subtitle_path, video.size)
    video = CompositeVideoClip([video] + all_linelevel_splits)

    # Set audio
    video = video.with_audio(audio)

    # Write output
    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        threads=12,
        preset="ultrafast",
    )


image_paths = [
    "Z:/AutoCut/examples/images/1.png",
    "Z:/AutoCut/examples/images/2.png",
    "Z:/AutoCut/examples/images/3.png",
    "Z:/AutoCut/examples/images/4.png",
    "Z:/AutoCut/examples/images/5.png",
    "Z:/AutoCut/examples/images/6.png",
    "Z:/AutoCut/examples/images/7.png",
]
audio_path = "Z:/AutoCut/examples/audio/1743274933.4666355.wav"
output_path = "Z:/AutoCut/outputs/output.mp4"


def main():
    global app
    print("Starting app...")
    create_composite_video(image_paths, audio_path, output_path)


if __name__ == "__main__":
    main()
