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

# Assuming whisper is installed and available
import whisper

# --- Configuration ---
FONT_PATH = "C:/Users/ranie/AppData/Local/Microsoft/Windows/Fonts/AsapCondensed-Medium.ttf"
# INCREASED FONT SIZE for better visibility
FONT_SIZE = 110 # Significantly larger
TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "yellow"
BG_COLOR = "0x000000@0.6"
BOX_PADDING = 20 # Increased padding slightly for larger font
TEXT_V_ALIGN = 0.80 # Slightly higher placement
MAX_ZOOM = 1.15  # Slightly increased max zoom
VIDEO_W = 1080
VIDEO_H = 1920
VIDEO_FPS = 24
FFMPEG_PRESET = "ultrafast" # Faster encoding for testing
FFMPEG_THREADS = 12

# --- Helper Functions ---

def run_command(cmd: list[str], description: str = "command"):
    """Runs a subprocess command, printing errors."""
    print(f"Running {description}: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        # Use stderr=subprocess.PIPE to capture FFmpeg progress/errors better
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        # Uncomment below lines for verbose FFmpeg output during debugging
        # print(f"STDOUT [{description}]:\n{process.stdout}\n--------------------")
        # print(f"STDERR [{description}]:\n{process.stderr}\n--------------------")
        print(f"{description.capitalize()} completed successfully.")
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Command: {' '.join(shlex.quote(c) for c in cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Output (stdout): {e.stdout}")
        print(f"Output (stderr): {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: '{cmd[0]}' command not found. Is FFmpeg (or ffprobe) installed and in your PATH?")
        raise

def get_media_duration(media_path: str) -> float:
    """Gets the duration of an audio or video file using ffprobe."""
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
    """Correctly escapes paths for FFmpeg filters, especially on Windows."""
    if platform.system() == "Windows":
        return path.replace("\\", "/").replace(":", "\\:")
    else:
        # Escape characters potentially problematic in filters: ' : \
        return path.replace("'", "\\'").replace(":", "\\:").replace("\\", "\\\\")

def escape_ffmpeg_path_simple(path: str) -> str:
     """Escapes paths for general FFmpeg arguments (less aggressive)."""
     return path.replace("\\", "/") # Forward slashes usually work best

# --- Core Logic Functions (Whisper, Line Split - Unchanged from previous revision) ---

def generate_word_timestamps(audio_path: str) -> List[Dict[str, Any]]:
    """Generate word timestamps using Whisper."""
    print("Loading Whisper model...")
    model = whisper.load_model("medium")
    print("Transcribing audio (this may take a while)...")
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Transcription complete.")

    wordlevel_info = []
    if result and "segments" in result:
        for segment in result["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start")
                    end = word_info.get("end")
                    if word and start is not None and end is not None:
                        wordlevel_info.append(
                            {
                                "word": word,
                                "start": float(start),
                                "end": float(end),
                            }
                        )
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
    font_name: str = "Arial",
    font_size: int = FONT_SIZE,
    text_color: str = "FFFFFF",      # Base color (White) - WILL BE SECONDARY
    highlight_color: str = "00FFFF", # Highlight color (Yellow) - WILL BE PRIMARY
    outline_color: str = "000000",   # Outline color (Black)
    shadow_color: str = "000000",    # Shadow color (Black)
    text_alpha: str = "00",          # Text Alpha (Opaque)
    shadow_alpha: str = "60",        # Shadow Alpha (~60% Opaque)
    video_width: int = VIDEO_W,
    video_height: int = VIDEO_H,
    text_v_align: float = TEXT_V_ALIGN,
    box_style: int = 1,
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
    primary_color_ass = f"&H{text_alpha}{highlight_color}&" # Primary = Highlight (Yellow)
    secondary_color_ass = f"&H{text_alpha}{text_color}&"   # Secondary = Base (White)
    # *************************
    outline_color_ass = f"&H{text_alpha}{outline_color}&"
    shadow_color_ass_field = f"&H{shadow_alpha}{shadow_color}&" # BackColour for shadow

    # ASS Header - Note Primary/Secondary are swapped relative to visual intent
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
Style: Default,{font_name},{font_size},{primary_color_ass},{secondary_color_ass},{outline_color_ass},{shadow_color_ass_field},-1,0,0,0,100,100,0,0,{box_style},{outline_width},{shadow_depth},2,25,25,{vertical_margin},1
"""
    # Alignment 2=BottomCenter.

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
            duration_cs = max(1, int((word_end_abs - word_start_abs) * 100))

            # Append the standard karaoke tag and the word
            ass_line_content += f"{{\\k{duration_cs}}}{word_text}"

            # Add a simple space after the word
            if i < len(line_words) - 1:
                ass_line_content += " "
        # --- End \k construction ---

        # Format dialogue start/end times
        start_time_str = format_time_ass(line_start_abs)
        end_time_str = format_time_ass(line_end_abs)

        # Add the complete dialogue line
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
        print("ASS file generated successfully using reverse color/fill karaoke.")
    except Exception as e:
        print(f"Error writing ASS file {output_ass_path}: {e}")
        raise

# Remember to keep format_time_ass function
def format_time_ass(time_sec: float) -> str:
    hours = int(time_sec // 3600); minutes = int((time_sec % 3600) // 60)
    seconds = int(time_sec % 60); centiseconds = int((time_sec * 100) % 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"
# --- Main Video Creation Function (Multi-Stage FFmpeg) ---

def create_composite_video_ffmpeg(
    image_paths: List[str],
    audio_path: str,
    output_path: str,
    video_width: int = VIDEO_W,
    video_height: int = VIDEO_H,
    video_fps: int = VIDEO_FPS,
    max_zoom: float = MAX_ZOOM
):
    """Creates the final video using FFmpeg with per-image zoom."""
    print("Starting multi-stage FFmpeg video creation process...")
    temp_dir = None # Initialize temporary directory variable

    if not image_paths: raise ValueError("No image paths provided.")
    if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio file not found: {audio_path}")
    for img_path in image_paths:
        if not os.path.exists(img_path): raise FileNotFoundError(f"Image file not found: {img_path}")

    # 1. Get audio duration
    audio_duration = get_media_duration(audio_path)
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # 2. Generate word timestamps & subtitle lines (if audio > 0)
    lines = []
    ass_file_path = None
    if audio_duration > 0.1: # Only process subs if audio is long enough
        word_timestamps = generate_word_timestamps(audio_path)
        if word_timestamps:
            lines = split_text_into_lines(word_timestamps)
            if lines:
                ass_file_path = "subtitles.ass"
                font_name_for_ass = os.path.splitext(os.path.basename(FONT_PATH))[0].replace('-', ' ')
                generate_ass_subtitle_file(lines, ass_file_path, font_name=font_name_for_ass)
            else: print("Warning: No subtitle lines generated from timestamps.")
        else: print("Warning: Failed to generate word timestamps.")
    else: print("Audio duration too short, skipping transcription and subtitles.")


    # === STAGE 1: Create zoomed video clips for each image ===
    print("\n--- Stage 1: Creating zoomed video clips per image ---")
    num_images = len(image_paths)
    if num_images == 0: raise ValueError("No images to process.")
    image_duration = audio_duration / num_images
    print(f"Processing {num_images} images, each clip duration: ~{image_duration:.2f} seconds.")

    temp_dir = tempfile.mkdtemp(prefix="autocut_ffmpeg_")
    print(f"Created temporary directory: {temp_dir}")
    temp_video_paths = []
    temp_concat_list_path = os.path.join(temp_dir, "ffmpeg_concat_list.txt")

    try:
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{num_images}: {os.path.basename(img_path)}")
            temp_img_output_path = os.path.join(temp_dir, f"temp_img_{i:03d}.mp4")
            escaped_img_path = escape_ffmpeg_path_simple(img_path)

            # --- Calculate total frames for this clip ---
            total_frames_in_clip = int(image_duration * video_fps)
            if total_frames_in_clip <= 0: total_frames_in_clip = 1 # Need at least one frame

            # --- REVERSED Zoom Expression (Zoom Out: max_zoom -> 1.0) ---
            # Formula: zoom = max_zoom - (progress) * (zoom_range)
            # Clamped using max(1.0, ...)
            zoom_range = max_zoom - 1.0
            # Use max(1, ...) in denominator to avoid division by zero if only 1 frame
            denominator = max(1, total_frames_in_clip - 1)
            zoom_expr = f"'max(1.0, {max_zoom} - (on-1)/{denominator} * {zoom_range})'"
            # Handle the 1-frame case explicitly
            if total_frames_in_clip <= 1:
                zoom_expr = f"'{max_zoom}'" # Start (and stay) at max_zoom if only one frame

            # --- End Zoom Expression Change ---

            cmd_img = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", escaped_img_path, "-t", str(image_duration),
                "-filter_complex",
                f"[0:v]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,"
                f"pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,"
                f"zoompan=z={zoom_expr}:" # Use the new zoom_expr
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=1:"
                f"s={video_width}x{video_height}:"
                f"fps={video_fps}"
                f"[vout]",
                "-map", "[vout]",
                "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", "20", "-an",
                temp_img_output_path
            ]

            print("Stage 1 Command:")
            print(' '.join(shlex.quote(c) for c in cmd_img))

            run_command(cmd_img, f"Create clip for image {i+1}")
            temp_video_paths.append(temp_img_output_path)

        # === STAGE 2: Concatenate video clips ===
        print("\n--- Stage 2: Concatenating video clips ---")
        # Create the concat list file
        with open(temp_concat_list_path, "w") as f:
            for vid_path in temp_video_paths:
                # Use simple path escaping for the concat file itself
                f.write(f"file '{escape_ffmpeg_path_simple(vid_path)}'\n")
        print(f"Generated concat list: {temp_concat_list_path}")
        # You can add a print here to show the content of the concat list if needed
        #with open(temp_concat_list_path, "r") as f:
        #    print("Concat List Content:\n", f.read())

        # === STAGE 3: Combine concatenated video, audio, and subtitles ===
        print("\n--- Stage 3: Combining video, audio, and subtitles ---")
        ffmpeg_cmd_final = ["ffmpeg", "-y"]

        # Inputs: Concatenated video, Original Audio
        ffmpeg_cmd_final.extend(["-f", "concat", "-safe", "0", "-i", temp_concat_list_path])
        ffmpeg_cmd_final.extend(["-i", audio_path])

        # Filters: Apply subtitles if ASS file exists
        filter_complex_final = []
        video_input_stream = "[0:v]" # Video comes from the first input (concat)
        if ass_file_path and os.path.exists(ass_file_path):
            escaped_ass_path = escape_ffmpeg_path(ass_file_path) # Filter path needs filter escaping
            # Ensure font name is properly quoted if it contains spaces
            quoted_font_name = f"'{font_name_for_ass}'" if ' ' in font_name_for_ass else font_name_for_ass
            filter_complex_final.append(
                # Apply subtitles filter. force_style might be needed sometimes.
                f"{video_input_stream}subtitles=filename='{escaped_ass_path}'"
                # Example of force_style (use if ASS style isn't respected):
                # f":force_style='FontName={quoted_font_name},FontSize={FONT_SIZE}'"
                f",format=pix_fmts=yuv420p[vout]" # Ensure compatible pixel format
            )
            final_video_map = "[vout]"
        else:
            # No subtitles, just ensure pixel format
            filter_complex_final.append(f"{video_input_stream}format=pix_fmts=yuv420p[vout]")
            final_video_map = "[vout]"

        if filter_complex_final:
             ffmpeg_cmd_final.extend(["-filter_complex", ";".join(filter_complex_final)])

        # Mapping: Map final video and original audio
        ffmpeg_cmd_final.extend(["-map", final_video_map if filter_complex_final else "0:v"]) # Map video
        ffmpeg_cmd_final.extend(["-map", "1:a"])                   # Map audio from second input

        # Output settings for the final video
        ffmpeg_cmd_final.extend(["-c:v", "libx264"])
        ffmpeg_cmd_final.extend(["-preset", "medium"]) # Use a better preset for final output
        ffmpeg_cmd_final.extend(["-crf", "23"])
        ffmpeg_cmd_final.extend(["-c:a", "aac"])
        ffmpeg_cmd_final.extend(["-b:a", "192k"])
        ffmpeg_cmd_final.extend(["-r", str(video_fps)])
        ffmpeg_cmd_final.extend(["-threads", str(FFMPEG_THREADS)])
        ffmpeg_cmd_final.extend(["-movflags", "+faststart"]) # Good for web video
        # Explicitly set duration again, good practice
        ffmpeg_cmd_final.extend(["-t", str(audio_duration)])
        ffmpeg_cmd_final.append(output_path)

        # Execute final command
        run_command(ffmpeg_cmd_final, "Final FFmpeg combination")
        print(f"Video created successfully: {output_path}")

    finally:
        # === STAGE 4: Cleanup ===
        print("\n--- Stage 4: Cleaning up temporary files ---")
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
        if ass_file_path and os.path.exists(ass_file_path):
             try:
                 os.remove(ass_file_path)
                 print(f"Removed temporary file: {ass_file_path}")
             except Exception as e:
                 print(f"Warning: Could not remove temporary ASS file {ass_file_path}: {e}")


# --- Main Execution ---

# Example Usage (matches original script's setup)
image_paths = [
    "Z:/AutoCut/examples/images/1.png", "Z:/AutoCut/examples/images/2.png",
    "Z:/AutoCut/examples/images/3.png", "Z:/AutoCut/examples/images/4.png",
    "Z:/AutoCut/examples/images/5.png", "Z:/AutoCut/examples/images/6.png",
    "Z:/AutoCut/examples/images/7.png",
]
audio_path = "Z:/AutoCut/examples/audio/1743274933.4666355.wav"
output_path = "Z:/AutoCut/outputs/output_ffmpeg_v2.mp4" # Changed output name

def main():
    print("Starting FFmpeg-based video generation (v2 - Per-image Zoom)...")
    try:
        create_composite_video_ffmpeg(image_paths, audio_path, output_path)
    except Exception as e:
        print(f"\n--- An error occurred during video creation ---")
        # print detailed traceback
        import traceback
        traceback.print_exc()
        print("-------------------------------------------------")

if __name__ == "__main__":
    main()