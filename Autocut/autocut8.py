# --- START OF FILE autocut6.py ---

import json
import os
import subprocess
import math
import textwrap
import shlex
import platform
import tempfile
import shutil
from typing import List, Dict, Any, Tuple, Optional

# --- Prerequisites ---
# Ensure the following are installed:
# 1. Python 3.x
# 2. FFmpeg and ffprobe:
#    - Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg
#    - Fedora: sudo dnf install ffmpeg
#    - Arch: sudo pacman -S ffmpeg
#    (Ensure they are in your system's PATH)
# 3. OpenAI Whisper:
#    - pip install -U openai-whisper
#    - May require rust: https://www.rust-lang.org/tools/install
# 4. Font file specified in FONT_PATH must exist.

# Assuming whisper is installed and available
try:
    import whisper
except ImportError:
    print("Error: whisper library not found.")
    print("Please install it: pip install -U openai-whisper")
    print("You might also need to install rust: https://www.rust-lang.org/tools/install")
    exit(1)

# --- Configuration ---

# <<< IMPORTANT: CHANGE THIS PATH >>>
# Find a suitable TTF font file on your Linux system. Common locations:
# /usr/share/fonts/truetype/
# /usr/local/share/fonts/
# ~/.local/share/fonts/
# Example using a common DejaVu font:
FONT_PATH = "/usr/share/fonts/truetype/Asap_Condensed/AsapCondensed-Medium.ttf"
# Example using Liberation Sans (often installed):
# FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
# Make absolutely sure this file exists! You can check with: ls -l /path/to/your/font.ttf

# Check if the specified font path exists
if not os.path.exists(FONT_PATH):
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"!!! ERROR: Font file not found at '{FONT_PATH}'")
    print(f"!!! Please edit the FONT_PATH variable in this script to point")
    print(f"!!! to a valid .ttf font file on your Linux system.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit(1)

# INCREASED FONT SIZE for better visibility
FONT_SIZE = 110 # Significantly larger
TEXT_COLOR = "white"       # FFmpeg color name or 0xRRGGBB
HIGHLIGHT_COLOR = "yellow" # FFmpeg color name or 0xRRGGBB
# BG_COLOR = "0x000000@0.6" # Background color for drawtext box (Hex ABGR)
BOX_PADDING = 20           # Increased padding slightly for larger font
TEXT_V_ALIGN = 0.80        # Vertical alignment (0.0=top, 0.5=center, 1.0=bottom)
MAX_ZOOM = 1.15            # Slightly increased max zoom
VIDEO_W = 1080
VIDEO_H = 1920
VIDEO_FPS = 24
FFMPEG_PRESET = "ultrafast" # Faster encoding for testing/initial runs
FINAL_PRESET = "medium"     # Preset for the final encoding stage
FFMPEG_THREADS = os.cpu_count() or 4 # Use available cores or default to 4

# --- Helper Functions ---

def run_command(cmd: list[str], description: str = "command"):
    """Runs a subprocess command, printing errors."""
    # Use shlex.join for modern Python if available, otherwise manual join
    try:
        cmd_str = shlex.join(cmd)
    except AttributeError: # Fallback for older Python versions
        cmd_str = ' '.join(shlex.quote(c) for c in cmd)
    print(f"Running {description}: {cmd_str}")

    try:
        # Use stderr=subprocess.PIPE to capture FFmpeg progress/errors better
        # Set universal_newlines=True is equivalent to text=True, but more explicit for older compatibility
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        # Uncomment below lines for verbose FFmpeg output during debugging
        # print(f"STDOUT [{description}]:\n{process.stdout}\n--------------------")
        # print(f"STDERR [{description}]:\n{process.stderr}\n--------------------")
        # Only print stderr if it contains something potentially relevant (heuristics)
        if process.stderr and not ("frame=" in process.stderr and "fps=" in process.stderr and "time=" in process.stderr):
             print(f"STDERR [{description}]:\n{process.stderr}\n--------------------")

        print(f"{description.capitalize()} completed successfully.")
        return process
    except subprocess.CalledProcessError as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! Error running {description}:")
        print(f"!!! Command: {cmd_str}")
        print(f"!!! Return code: {e.returncode}")
        print(f"!!! Output (stdout): {e.stdout}")
        print(f"!!! Output (stderr): {e.stderr}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise
    except FileNotFoundError:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! Error: '{cmd[0]}' command not found.")
        print(f"!!! Is FFmpeg (specifically '{cmd[0]}') installed and in your system's PATH?")
        print(f"!!! Check with: which {cmd[0]}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise

def get_media_duration(media_path: str) -> float:
    """Gets the duration of an audio or video file using ffprobe."""
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Media file not found for duration check: {media_path}")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    result = run_command(cmd, f"ffprobe duration check for {os.path.basename(media_path)}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        print(f"Error: Could not parse duration from ffprobe output: {result.stdout}")
        raise

def escape_ffmpeg_path(path: str) -> str:
    """Correctly escapes paths for FFmpeg filters (subtitles filename)."""
    # On Linux/macOS, need to escape special filter chars like ' : \
    # Double escaping backslash because it's processed twice (shell then ffmpeg filter).
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    # Windows logic removed as we target Linux now
    # if platform.system() == "Windows":
    #     return path.replace("\\", "/").replace(":", "\\:")

def escape_ffmpeg_path_simple(path: str) -> str:
     """Escapes paths for general FFmpeg arguments (less aggressive - primarily for concat)."""
     # Just ensure forward slashes, generally safer for concat lists etc. across platforms
     # No real escaping needed on Linux for typical filenames in concat lists.
     return path.replace("\\", "/")


# --- Core Logic Functions (Whisper, Line Split - Unchanged from previous revision) ---

def generate_word_timestamps(audio_path: str) -> List[Dict[str, Any]]:
    """Generate word timestamps using Whisper."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")
    print("Loading Whisper model (medium)...")
    # Consider using a smaller model (base, small, tiny) if medium is too slow or resource-intensive
    # model = whisper.load_model("base.en") # Example for faster English-only
    model = whisper.load_model("base")
    print("Transcribing audio (this may take a while)...")
    # Add language detection/setting if needed: result = model.transcribe(audio_path, language='en', word_timestamps=True)
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Transcription complete.")

    wordlevel_info = []
    if result and "segments" in result:
        for segment in result["segments"]:
            # Check if 'words' exists and is a list
            if isinstance(segment.get("words"), list):
                for word_info in segment["words"]:
                    # Check if word_info is a dictionary and contains keys
                    if isinstance(word_info, dict):
                        word = word_info.get("word", "").strip()
                        start = word_info.get("start")
                        end = word_info.get("end")
                        if word and start is not None and end is not None:
                            try:
                                wordlevel_info.append(
                                    {
                                        "word": word,
                                        "start": float(start),
                                        "end": float(end),
                                    }
                                )
                            except ValueError:
                                print(f"Warning: Skipping word with invalid timestamp: {word_info}")
                    else:
                         print(f"Warning: Unexpected word_info format in segment: {word_info}")
            else:
                 # This can happen with Whisper if a segment has no words (e.g., silence)
                 # print(f"Debug: Segment has no 'words' list or it's not a list: {segment}")
                 pass # Just ignore segments without word lists
    else:
        print("Warning: Whisper transcription did not return expected segments/words.")
    return wordlevel_info

def split_text_into_lines(
    data: List[Dict[str, Any]],
    max_chars: int = 30,
    max_duration: float = 3.5, # Slightly longer duration allowed
    max_gap: float = 0.75,
) -> List[Dict[str, Any]]:
    """Splits word timestamps into lines suitable for subtitles."""
    if not data: return []
    subtitles, current_line_words = [], []
    line_start_time = data[0]["start"] if data else 0

    for i, word_data in enumerate(data):
        start, end = word_data["start"], word_data["end"]
        potential_line = current_line_words + [word_data]
        potential_line_text = " ".join(item["word"] for item in potential_line)
        potential_line_chars = len(potential_line_text)
        potential_line_duration = end - line_start_time

        gap = 0
        if current_line_words and i > 0:
             last_word_end_time = current_line_words[-1]["end"]
             gap = start - last_word_end_time

        char_exceeded = potential_line_chars > max_chars
        duration_exceeded = potential_line_duration > max_duration
        gap_exceeded = gap > max_gap and len(current_line_words) > 0
        is_last_word = (i == len(data) - 1)

        # Decide to break line *before* adding the current word if limits exceeded
        if current_line_words and (char_exceeded or duration_exceeded or gap_exceeded):
            line_text = " ".join(item["word"] for item in current_line_words)
            line_end_time = current_line_words[-1]["end"]
            subtitles.append({
                "text": line_text, "start": line_start_time, "end": line_end_time,
                "words": current_line_words
            })
            current_line_words = [word_data] # Start new line with current word
            line_start_time = start
        else:
            # Add current word if no break needed yet, or if line is empty
            if not current_line_words: line_start_time = start # Set start time for new line
            current_line_words.append(word_data)

    # Add the last remaining line
    if current_line_words:
        line_text = " ".join(item["word"] for item in current_line_words)
        line_end_time = current_line_words[-1]["end"]
        subtitles.append({
            "text": line_text, "start": line_start_time, "end": line_end_time,
            "words": current_line_words
        })
    return subtitles


# --- ASS Subtitle Generation (Alternative: Reverse Colors + \k Fill) ---
def generate_ass_subtitle_file(
    lines: List[Dict[str, Any]],
    output_ass_path: str,
    font_name: str = "Arial", # Default, overridden by font file name extraction
    font_size: int = FONT_SIZE,
    # ASS Colors are &HAABBGGRR& (Alpha, Blue, Green, Red)
    text_color_ass: str = "FFFFFF",      # White = FFFFFF
    highlight_color_ass: str = "00FFFF", # Yellow = 00FFFF (Cyan=FFFF00, Green=00FF00, Red=0000FF)
    outline_color_ass: str = "000000",   # Black = 000000
    shadow_color_ass: str = "000000",    # Black = 000000
    text_alpha: str = "00",              # Opaque = 00, 50% = 80, Invisible = FF
    shadow_alpha: str = "80",            # Shadow Alpha (~50% Opaque = 80)
    video_width: int = VIDEO_W,
    video_height: int = VIDEO_H,
    text_v_align: float = TEXT_V_ALIGN,
    box_style: int = 1, # 1 = Outline + Opaque box, 3 = Opaque box only
    outline_width: float = 2.5,
    shadow_depth: float = 2.0
    ):
    """
    Generates an ASS subtitle file using the standard {\\k} tag but with
    reversed Primary/Secondary colors to simulate a highlight fill effect.
    Primary = Highlight (Yellow), Secondary = Base (White).
    """
    print(f"Generating ASS subtitle file using reverse color/fill karaoke: {output_ass_path}")

    vertical_margin = max(15, int(video_height * (1.0 - text_v_align)))

    # Format ASS colors &HAABBGGRR&
    # *** NOTE THE SWAP HERE ***
    primary_color = f"&H{text_alpha}{highlight_color_ass}&" # Primary = Highlight (Yellow/Cyan/etc.)
    secondary_color = f"&H{text_alpha}{text_color_ass}&"   # Secondary = Base (White)
    # *************************
    outline_color = f"&H{text_alpha}{outline_color_ass}&"
    shadow_color = f"&H{shadow_alpha}{shadow_color_ass}&" # BackColour uses shadow settings

    # Extract font name from the FONT_PATH for the ASS style
    # Basic extraction, might need adjustment for complex font names
    base_font_name = os.path.splitext(os.path.basename(FONT_PATH))[0]# --- START MODIFICATION ---
    # Specific override for Asap Condensed Medium based on fc-list output
    if "AsapCondensed-Medium" in base_font_name: # Check if it's the specific file
        clean_font_name = "Asap Condensed Medium"
        print(f"Specific font override applied for Asap Condensed Medium.")
    else:
        # Fallback to original logic for other fonts
        clean_font_name = base_font_name.replace('-', ' ').replace('_', ' ')
    # --- END MODIFICATION ---
    print(f"Using font name in ASS style: '{clean_font_name}' (derived from {os.path.basename(FONT_PATH)})")

    # ASS Header - Note Primary/Secondary are swapped relative to visual intent
    # Alignment: 2 = Bottom Center. Adjust MarginV accordingly.
    # Use clean_font_name derived from the actual font file path
    header = f"""[Script Info]
Title: Generated Fill Karaoke Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {video_width}
PlayResY: {video_height}
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{clean_font_name},{font_size},{primary_color},{secondary_color},{outline_color},{shadow_color},-1,0,0,0,100,100,0,0,{box_style},{outline_width},{shadow_depth},2,25,25,{vertical_margin},1
"""
    # Encoding 1 = Default (usually UTF-8 on Linux)

    dialogue_lines = []
    for line_data in lines:
        line_start_abs = line_data["start"]
        # Use precise end time of the last word for the dialogue line duration
        line_end_abs = line_data["words"][-1]['end'] if line_data["words"] else line_data["end"]
        line_words = line_data["words"]

        # --- Build the ASS line with simple {\k<duration>} tags ---
        ass_line_content = ""
        for i, word_data in enumerate(line_words):
            word_text = word_data["word"].strip()
            if not word_text: continue

            word_start_abs = word_data["start"]
            word_end_abs = word_data["end"]

            # Calculate word duration in CENTISECONDS for \k tag
            # Ensure duration is at least 1 centisecond (10ms)
            duration_cs = max(1, int(round((word_end_abs - word_start_abs) * 100))) # Use round for precision

            # Append the standard karaoke tag and the word
            ass_line_content += f"{{\\k{duration_cs}}}{word_text}"

            # Add a simple space after the word (ASS handles spacing better)
            if i < len(line_words) - 1:
                ass_line_content += " "
        # --- End \k construction ---

        # Format dialogue start/end times (H:MM:SS.cc)
        start_time_str = format_time_ass(line_start_abs)
        end_time_str = format_time_ass(line_end_abs)

        # Add the complete dialogue line
        # Layer=0, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        final_ass_text = ass_line_content.strip()
        if final_ass_text:
            dialogue_lines.append(
                f"Dialogue: 0,{start_time_str},{end_time_str},Default,,0,0,0,,"
                f"{final_ass_text}"
            )

    # Write the ASS file
    try:
        with open(output_ass_path, "w", encoding='utf-8') as f:
            f.write(header)
            f.write("\n".join(dialogue_lines))
        print(f"ASS file generated successfully: {output_ass_path}")
    except Exception as e:
        print(f"Error writing ASS file {output_ass_path}: {e}")
        raise

# Format time H:MM:SS.cc for ASS
def format_time_ass(time_sec: float) -> str:
    """Formats seconds into H:MM:SS.cc"""
    if time_sec < 0: time_sec = 0 # Ensure non-negative time
    hours = int(time_sec // 3600)
    minutes = int((time_sec % 3600) // 60)
    seconds = int(time_sec % 60)
    centiseconds = int(round((time_sec * 100)) % 100) # Use round for precision
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"

# --- Main Video Creation Function (Multi-Stage FFmpeg) ---

def create_composite_video_ffmpeg(
    image_paths: List[str],
    audio_path: str,
    output_path: str,
    font_path: str = FONT_PATH, # Pass font path
    video_width: int = VIDEO_W,
    video_height: int = VIDEO_H,
    video_fps: int = VIDEO_FPS,
    max_zoom: float = MAX_ZOOM,
    ffmpeg_threads: int = FFMPEG_THREADS,
    stage1_preset: str = FFMPEG_PRESET,
    stage3_preset: str = FINAL_PRESET
):
    """Creates the final video using FFmpeg with per-image zoom and subtitles."""
    print("Starting multi-stage FFmpeg video creation process...")
    temp_dir = None # Initialize temporary directory variable
    ass_file_path = None # Initialize ASS file path

    # --- Input Validation ---
    if not image_paths: raise ValueError("No image paths provided.")
    if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio file not found: {audio_path}")
    #for img_path in image_paths:
    #    if not os.path.exists(img_path): raise FileNotFoundError(f"Image file not found: {img_path}")
    if not os.path.exists(font_path): raise FileNotFoundError(f"Font file not found: {font_path}") # Also check font here

    # --- Preparation ---
    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    try:
        # 1. Get audio duration
        audio_duration = get_media_duration(audio_path)
        print(f"Audio duration: {audio_duration:.2f} seconds")
        if audio_duration <= 0: raise ValueError("Audio duration is zero or negative.")

        # 2. Generate word timestamps & subtitle lines
        lines = []
        word_timestamps = generate_word_timestamps(audio_path)
        if word_timestamps:
            lines = split_text_into_lines(word_timestamps)
            if lines:
                # Create ASS file in a temporary location or alongside output
                # Using temp dir is cleaner
                temp_dir = tempfile.mkdtemp(prefix="autocut_ffmpeg_")
                print(f"Created temporary directory: {temp_dir}")
                ass_file_path = os.path.join(temp_dir, "subtitles.ass")
                # Font name for ASS generation is derived inside the function from font_path
                generate_ass_subtitle_file(lines, ass_file_path, font_name="Placeholder") # Name is ignored now
            else: print("Warning: No subtitle lines generated from timestamps.")
        else: print("Warning: Failed to generate word timestamps. Proceeding without subtitles.")

        import requests

        temp_image_paths = []
        temp_dir = tempfile.mkdtemp(prefix="autocut_images_")
        print(f"Created temporary directory for images: {temp_dir}")
        for i, image_url in enumerate(image_paths): # Assuming image_paths is the URL list
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Get filename from URL or generate a temporary one
                filename = os.path.join(temp_dir, f"downloaded_image_{i}.png") # Or infer from URL

                with open(filename, 'wb') as img_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        img_file.write(chunk)
                temp_image_paths.append(filename)
                print(f"Downloaded: {image_url} to {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {image_url}: {e}")
                # Handle the error appropriately (e.g., skip this image)


        # === STAGE 1: Create zoomed video clips for each image ===
        print("\n--- Stage 1: Creating zoomed video clips per image ---")
        num_images = len(temp_image_paths)
        image_duration = audio_duration / num_images
        print(f"Processing {num_images} images, each clip duration: ~{image_duration:.2f} seconds.")

        # Create temp dir if not already created for ASS file
        if not temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="autocut_ffmpeg_")
            print(f"Created temporary directory: {temp_dir}")

        temp_video_paths = []
        temp_concat_list_path = os.path.join(temp_dir, "ffmpeg_concat_list.txt")

        for i, img_path in enumerate(temp_image_paths):
            print(f"Processing image {i+1}/{num_images}: {os.path.basename(img_path)}")
            temp_img_output_path = os.path.join(temp_dir, f"temp_img_{i:03d}.mp4")
            # Use simple escaping for input image path
            escaped_img_path = escape_ffmpeg_path_simple(img_path)

            # Calculate total frames for this clip
            total_frames_in_clip = max(1, int(round(image_duration * video_fps))) # Ensure at least 1 frame

            # --- REVERSED Zoom Expression (Zoom Out: max_zoom -> 1.0) ---
            zoom_range = max_zoom - 1.0
            if total_frames_in_clip <= 1:
                zoom_expr = f"{max_zoom}" # Constant zoom if 1 frame
            else:
                # Formula: zoom = max_zoom - (progress) * (zoom_range) where progress = (frame_num - 1) / (total_frames - 1)
                # FFmpeg 'on' is output frame number (starts at 1). Use max(1,...) to avoid division by zero.
                denominator = max(1, total_frames_in_clip - 1)
                # Needs careful quoting for the shell AND ffmpeg filtergraph
                # Single quotes around the expression for the shell, protect internal quotes if needed
                zoom_expr = f'max(1.0,{max_zoom}-(on-1)/{denominator}*{zoom_range})'

            # Construct filtergraph for scaling, padding, and zoompan
            # Scale first, then pad, then zoompan
            filter_complex_img = (
                f"[0:v]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease:flags=bicubic," # Scale down nicely
                f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color=black," # Pad to target size
                f"setsar=1," # Set Sample Aspect Ratio to 1:1
                f"zoompan=z='{zoom_expr}':" # Apply zoom expression
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':" # Center the zoom
                f"d={total_frames_in_clip}:" # Duration in frames for zoompan
                f"s={video_width}x{video_height}:" # Output size
                f"fps={video_fps}" # Output frame rate
                f"[vout]"
            )

            cmd_img = [
                "ffmpeg", "-y", # Overwrite output without asking
                "-loop", "1", "-i", escaped_img_path, # Input image looped
                "-t", str(image_duration), # Duration for this segment
                "-filter_complex", filter_complex_img, # The complex filtergraph
                "-map", "[vout]", # Map the output of the filtergraph
                "-c:v", "libx264", # Video codec
                "-preset", stage1_preset, # Encoding speed/quality preset
                "-crf", "20", # Quality setting (lower is better, 18-24 often good)
                "-threads", str(ffmpeg_threads // 2 or 1), # Use fewer threads for parallel stages
                "-an", # No audio in intermediate files
                temp_img_output_path # Output path for this segment
            ]

            run_command(cmd_img, f"Create clip for image {i+1}")
            temp_video_paths.append(temp_img_output_path)

        # === STAGE 2: Concatenate video clips ===
        print("\n--- Stage 2: Concatenating video clips ---")
        with open(temp_concat_list_path, "w", encoding='utf-8') as f:
            for vid_path in temp_video_paths:
                # Use simple path escaping for the concat file itself
                f.write(f"file '{escape_ffmpeg_path_simple(vid_path)}'\n")
        print(f"Generated concat list: {temp_concat_list_path}")

        # === STAGE 3: Combine concatenated video, audio, and subtitles ===
        print("\n--- Stage 3: Combining video, audio, and subtitles ---")
        ffmpeg_cmd_final = ["ffmpeg", "-y"] # Overwrite final output

        # Inputs: Concatenated video (via demuxer), Original Audio
        ffmpeg_cmd_final.extend(["-f", "concat", "-safe", "0", "-i", temp_concat_list_path])
        ffmpeg_cmd_final.extend(["-i", escape_ffmpeg_path_simple(audio_path)])

        # Filter complex for subtitles
        filter_complex_final = []
        video_map = "[0:v]" # Default map if no filter
        audio_map = "[1:a]"

        if ass_file_path and os.path.exists(ass_file_path):
            # Escape the ASS path for the subtitles filter
            escaped_ass_path_for_filter = escape_ffmpeg_path(ass_file_path)
            # Escape the font directory path for the subtitles filter
            font_dir = escape_ffmpeg_path(os.path.dirname(font_path))

            # Subtitles filter string
            # We need to specify the font directory for libass to find the font
            # Using fontsdir= parameter. Ensure path escaping is correct for filter.
            # Also ensure output pixel format is compatible with most players (yuv420p)
            subtitle_filter = (
                f"[0:v]subtitles=filename='{escaped_ass_path_for_filter}':fontsdir='{font_dir}'"
                f",format=pix_fmts=yuv420p[vout]"
            )
            filter_complex_final.append(subtitle_filter)
            video_map = "[vout]" # Map the output of the filter
        else:
            # If no subtitles, still ensure pixel format
             filter_complex_final.append(f"[0:v]format=pix_fmts=yuv420p[vout]")
             video_map = "[vout]" # Map the output of the format filter


        if filter_complex_final:
            ffmpeg_cmd_final.extend(["-filter_complex", ";".join(filter_complex_final)])

        # Mapping: Map final video and original audio
        ffmpeg_cmd_final.extend(["-map", video_map]) # Map video (potentially filtered)
        ffmpeg_cmd_final.extend(["-map", "1:a"])     # Directly map audio stream from the second input (index 1)

        # Output settings for the final video
        ffmpeg_cmd_final.extend(["-c:v", "libx264"])        # Video codec
        ffmpeg_cmd_final.extend(["-preset", stage3_preset]) # Use a better preset for final output
        ffmpeg_cmd_final.extend(["-crf", "23"])             # Constant Rate Factor (quality)
        ffmpeg_cmd_final.extend(["-c:a", "aac"])            # Audio codec
        ffmpeg_cmd_final.extend(["-b:a", "192k"])           # Audio bitrate
        ffmpeg_cmd_final.extend(["-r", str(video_fps)])     # Video framerate
        ffmpeg_cmd_final.extend(["-threads", str(ffmpeg_threads)]) # Use configured threads
        ffmpeg_cmd_final.extend(["-movflags", "+faststart"]) # Good for web video streaming
        # Explicitly set duration again, good practice to avoid slight variations
        ffmpeg_cmd_final.extend(["-to", str(audio_duration)]) # Use -to for duration limit
        ffmpeg_cmd_final.append(output_path)          # Final output file path

        # Execute final command
        run_command(ffmpeg_cmd_final, "Final FFmpeg combination")
        print(f"Video created successfully: {output_path}")

    except Exception as e:
         # Print detailed traceback on any error
         print(f"\n--- An error occurred during video creation ---")
         import traceback
         traceback.print_exc()
         print("-------------------------------------------------")

    finally:
        #=== STAGE 4: Cleanup ===
        print("\n--- Stage 4: Cleaning up temporary files ---")
        if temp_dir and os.path.exists(temp_dir):
            try:
                import torch, gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
        # ASS file was inside temp_dir, so no need to delete separately
        # if ass_file_path and os.path.exists(ass_file_path):
        #      try:
        #          os.remove(ass_file_path)
        #          print(f"Removed temporary file: {ass_file_path}")
        #      except Exception as e:
        #          print(f"Warning: Could not remove temporary ASS file {ass_file_path}: {e}")


# --- Main Execution ---

# <<< IMPORTANT: CHANGE THESE PATHS to match your Linux file structure >>>
# Example paths for Linux. Replace with your actual file locations.
# Use absolute paths for clarity or paths relative to where you run the script.
base_dir = "/home/raniel/YTAutomate/Autocut" # Example base directory

image_paths = [
    f"{base_dir}/examples/images/1.png", f"{base_dir}/examples/images/2.png",
    f"{base_dir}/examples/images/3.png", f"{base_dir}/examples/images/4.png",
    f"{base_dir}/examples/images/5.png", f"{base_dir}/examples/images/6.png",
    f"{base_dir}/examples/images/7.png",
]
audio_path = f"{base_dir}/examples/audio/morgan.wav" # CHANGE FILENAME
output_path = f"{base_dir}/outputs/output_ffmpeg_linux.mp4"

def main():
    print("Starting FFmpeg-based video generation (Linux Compatible)...")
    print(f"Using Font: {FONT_PATH}")
    print(f"Using FFmpeg Threads: {FFMPEG_THREADS}")

    # Basic check for example paths before starting
    #if not all(os.path.exists(p) for p in image_paths):
    #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #    print("!!! WARNING: One or more example image paths")
    #    print(f"!!! (e.g., '{image_paths[0]}') do not exist.")
    #    print(f"!!! Please update the 'image_paths' list in the")
    #    print(f"!!! script with your actual image locations.")
    #    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Decide whether to exit or proceed
        # return # Uncomment to stop if example paths are wrong

    if not os.path.exists(audio_path):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! WARNING: Example audio path '{audio_path}'")
        print(f"!!! does not exist. Please update the 'audio_path'")
        print(f"!!! variable with your actual audio file location.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # return # Uncomment to stop if example paths are wrong

    create_composite_video_ffmpeg(image_paths, audio_path, output_path)


if __name__ == "__main__":
    main()

# --- END OF FILE autocut6.py ---