# --- START OF FILE server.py ---

import os
import time
import torch
import json
import random
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse # Keep FileResponse for potential future use, but current code returns path
from typing import Optional
import torchaudio
import soundfile as sf
# from pydub import AudioSegment, silence # pydub seems unused directly here, maybe used inside F5-TTS?
# import re # re seems unused
from importlib.resources import files
from cached_path import cached_path
import sys
import logging
import io
import gc
# import magic # magic seems unused
from pydantic import BaseModel


# ---> ADD THIS SECTION <---
# Get the absolute path of the directory where the script resides
script_dir = os.path.dirname(os.path.abspath(__file__))
# --- END OF ADDED SECTION ---


# Add F5-TTS root directory to path so we can import modules
# Make sure this path is correct for your Linux environment where F5-TTS is located
F5_TTS_ROOT = os.getenv("F5_TTS_ROOT", "/workspace/F5-TTS") # Use env var or default
if os.path.exists(F5_TTS_ROOT) and F5_TTS_ROOT not in sys.path:
    sys.path.append(F5_TTS_ROOT)
else:
    logging.warning(f"F5-TTS root directory '{F5_TTS_ROOT}' not found or already in sys.path. Ensure it's correctly set.")

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        # Ensure you have the spaces library installed if this is needed
        # pip install huggingface_hub gradio
        # Note: spaces.GPU might require specific Gradio/Hugging Face Spaces environment
        try:
            return spaces.GPU(func)
        except Exception as e:
            logging.warning(f"Could not apply spaces.GPU decorator: {e}. Running without GPU enforcement.")
            return func
    else:
        return func

# Moved imports here after sys.path modification
try:
    from f5_tts.model import DiT, UNetT
    from f5_tts.model.utils import seed_everything
    from f5_tts.infer.utils_infer import (
        load_vocoder,
        load_model,
        preprocess_ref_audio_text,
        infer_process,
        # remove_silence_for_generated_wav, # Seems unused
        # save_spectrogram, # Seems unused
    )
except ImportError as e:
    logging.error(f"Failed to import F5-TTS modules. Ensure F5_TTS_ROOT ('{F5_TTS_ROOT}') is correct and contains the library. Error: {e}")
    sys.exit(1)


DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# load models
logging.info("Loading vocoder...")
#vocoder = load_vocoder()
logging.info("Vocoder loaded.")

def load_f5tts():
    logging.info("Loading F5-TTS model...")
    try:
        ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
        F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
        model = load_model(DiT, F5TTS_model_cfg, ckpt_path)
        logging.info("F5-TTS model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading F5-TTS model: {e}")
        raise

#F5TTS_ema_model = load_f5tts()

logging.basicConfig(level=logging.INFO) # Configure logging early

# ---> LOAD MODELS ONCE HERE <---
logging.info("Loading models at startup...")
try:
    F5TTS_EMA_MODEL_GLOBAL = load_f5tts()
    VOCODER_GLOBAL = load_vocoder()
    logging.info("Models loaded successfully into global scope.")
except Exception as e:
    logging.error(f"Fatal error: Could not load models at startup: {e}")
    sys.exit(1) # Exit if models can't load

# Placeholders for unused models mentioned in original infer function
E2TTS_ema_model = None
custom_ema_model = None
pre_custom_path = None

# --- Removed unused Chat model state variables ---
# chat_model_state = None
# chat_tokenizer_state = None

# Define show_info behavior (use logging)
def log_info(message):
    logging.info(message)

# Define gr.Warning behavior (use logging)
def log_warning(message):
    logging.warning(message)

@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    #model_name, # Changed 'model' to 'model_name' to avoid conflict with loaded model variable
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=log_info,
    file_wave=None,
    seed=None,
):
    
    # --- Use the GLOBAL models ---
    global F5TTS_EMA_MODEL_GLOBAL, VOCODER_GLOBAL # Declare use of globals
    ema_model = F5TTS_EMA_MODEL_GLOBAL
    vocoder = VOCODER_GLOBAL

    if not ref_audio_orig:
        log_warning("Reference audio path is missing or invalid.")
        # Returning None or raising an exception might be better than returning text
        raise ValueError("Reference audio path is required.")

    if not os.path.exists(ref_audio_orig):
        log_warning(f"Reference audio file not found at: {ref_audio_orig}")
        raise ValueError(f"Reference audio file not found: {ref_audio_orig}")

    if not gen_text or not gen_text.strip():
        log_warning("Text to generate is empty.")
        raise ValueError("Text to generate cannot be empty.")

    if seed is None:
        seed = random.randint(0, 2**32 - 1) # Use standard int range
    seed_everything(seed)
    # seed = seed # Assigning seed to itself is redundant

    show_info(f"Processing with seed: {seed}")
    show_info("Preprocessing reference audio and text...")
    try:
        # preprocess_ref_audio_text might handle non-wav files internally via torchaudio
        ref_audio, ref_text_processed = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)
        # Use provided ref_text if available, otherwise use the processed one (if any)
        final_ref_text = ref_text if ref_text else ref_text_processed
    except Exception as e:
        show_info(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing reference audio/text: {e}")

    # --- Model selection logic (simplified as only F5TTS is loaded) ---
    #if model_name == DEFAULT_TTS_MODEL:
    # --- Removed E2-TTS and Custom model logic as they weren't loaded ---
    # elif model_name == "E2-TTS":
    #     global E2TTS_ema_model
    #     if E2TTS_ema_model is None:
    #         show_info("Loading E2-TTS model...")
    #         # E2TTS_ema_model = load_e2tts() # load_e2tts function was not defined
    #         raise NotImplementedError("E2-TTS model loading not implemented")
    #     ema_model = E2TTS_ema_model
    # elif isinstance(model_name, list) and model_name[0] == "Custom":
    #     assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
    #     global custom_ema_model, pre_custom_path
    #     if pre_custom_path != model_name[1]:
    #         show_info("Loading Custom TTS model...")
    #         # custom_ema_model = load_custom(model_name[1], vocab_path=model_name[2], model_cfg=model_name[3]) # load_custom not defined
    #         raise NotImplementedError("Custom model loading not implemented")
    #         pre_custom_path = model_name[1]
    #     ema_model = custom_ema_model
    #else:
    #     raise ValueError(f"Unsupported model name: {model_name}. Only '{DEFAULT_TTS_MODEL}' is currently supported.")

    if ema_model is None:
        raise RuntimeError("TTS model (ema_model) is not loaded.")
    if vocoder is None:
        raise RuntimeError("Vocoder is not loaded.")

    show_info("Starting TTS inference process...")
    try:
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            final_ref_text, # Use the determined reference text
            gen_text,
            ema_model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            show_info=show_info,
        )
    except Exception as e:
        show_info(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Error during speech generation: {e}")

    if file_wave is not None:
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(file_wave), exist_ok=True)
            sf.write(file_wave, final_wave, final_sample_rate)
            show_info(f"Generated audio saved to: {file_wave}")
        except Exception as e:
            show_info(f"Error saving generated wave file: {e}")
            # Don't raise here, as we still want to return the audio data if possible
    del ema_model
    del vocoder
    ema_model = None
    vocoder = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # Return tuple (sample_rate, audio_data) expected by downstream processing
    return (final_sample_rate, final_wave)


logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Initialize F5-TTS model (already done above)

# ... CORS middleware ...

# ---> MODIFY THESE LINES <---
# Use absolute paths based on the script's location
output_dir = os.path.join(script_dir, 'outputs')
resources_dir = os.path.join(script_dir, 'resources')
# --- END OF MODIFIED LINES ---

# Ensure these directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(resources_dir, exist_ok=True)
logging.info(f"Using resources directory: {resources_dir}")
logging.info(f"Using output directory: {output_dir}")


# --- Reference Audio Handling ---
# This part should now use the correct absolute resources_dir path
default_ref_audio_path_in_package = None
try:
    # This assumes f5_tts package is installed correctly
    default_ref_audio_path_in_package = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
except ImportError:
     logging.warning("Could not find 'f5_tts' package resources. Default reference audio might be unavailable.")
except Exception as e:
     logging.warning(f"Error accessing package resources: {e}")

# This now uses the absolute path defined earlier
default_ref_audio_local_path = os.path.join(resources_dir, "default_en.wav")
default_ref_text = "Some call me nature, others call me mother nature." # Define default text

if default_ref_audio_path_in_package and os.path.exists(default_ref_audio_path_in_package):
    if not os.path.exists(default_ref_audio_local_path):
        try:
            # import shutil # Already imported at the top
            shutil.copy2(default_ref_audio_path_in_package, default_ref_audio_local_path)
            logging.info(f"Copied default reference audio to {default_ref_audio_local_path}")
        except Exception as e:
            logging.error(f"Failed to copy default reference audio: {e}")
else:
    if not os.path.exists(default_ref_audio_local_path):
        logging.warning(f"Default reference audio '{default_ref_audio_local_path}' not found and could not be copied from package.")


# Simplified TTS function call
@gpu_decorator # Keep decorator if needed
def basic_tts(
    ref_audio_input,
    ref_text_input,
    gen_text_input,
    speed,
    file_wave # Pass the save path here
    ):
    # Directly call infer, assuming tts_model_choice is globally set or passed
    audio_out_tuple = infer(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        #tts_model_choice, # Use the global choice (DEFAULT_TTS_MODEL)
        speed=speed,
        file_wave=file_wave # Pass the intended save path to infer
    )
    # infer now returns (sample_rate, wave_data)
    return audio_out_tuple


# --- Removed convert_to_wav function - assuming torchaudio/pydub handles it ---
# def convert_to_wav(input_path, output_path):
#     try:
#         logging.info(f"Attempting to convert {input_path} to WAV at {output_path}")
#         audio = AudioSegment.from_file(input_path)
#         audio.export(output_path, format="wav")
#         logging.info(f"Successfully converted {input_path} to {output_path}")
#     except Exception as e:
#         logging.error(f"Error converting {input_path} to WAV: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to convert reference audio to WAV: {e}")


@app.get("/synthesize_speech/")
async def synthesize_speech(
        text: str,
        voice: str,
        speed: Optional[float] = 1.0,
):
    """
    Synthesize speech from text using a specified voice reference audio file.
    The 'voice' parameter should be the filename (without extension)
    of a reference audio file located in the 'resources' directory.
    """
    start_time = time.time()
    logging.info(f"Received request for voice '{voice}' with text: '{text[:50]}...'")

    reference_file = None
    # Look for audio files starting with the 'voice' name in the resources directory
    try:
        potential_files = [f for f in os.listdir(resources_dir) if f.lower().startswith(voice.lower() + '.')]
        if not potential_files:
             # Fallback to default if specific voice not found and it's requested? Or just error?
             # Let's error if the specific voice is not found.
             logging.error(f"No reference audio file found starting with '{voice}' in '{resources_dir}'")
             raise HTTPException(status_code=404, detail=f"Voice reference '{voice}' not found.") # Use 404

        # Prefer '.wav' files if multiple formats exist for the same name
        wav_files = [f for f in potential_files if f.lower().endswith('.wav')]
        if wav_files:
            reference_file = os.path.join(resources_dir, wav_files[0])
            logging.info(f"Using WAV reference: {reference_file}")
        else:
            # Use the first found file if no WAV exists (assuming infer can handle it)
            reference_file = os.path.join(resources_dir, potential_files[0])
            logging.info(f"Using non-WAV reference: {reference_file}. Ensure F5-TTS preprocessing handles this format.")
            # --- Conversion logic removed - rely on preprocess_ref_audio_text ---
            # wav_path = os.path.join(output_dir, f"{voice}_ref_converted.wav")
            # convert_to_wav(os.path.join(resources_dir, potential_files[0]), wav_path)
            # reference_file = wav_path

    except FileNotFoundError:
        logging.error(f"Resources directory '{resources_dir}' not found.")
        raise HTTPException(status_code=500, detail="Server configuration error: Resources directory missing.")
    except Exception as e:
        logging.error(f"Error finding reference file for voice '{voice}': {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing voice resources: {e}")


    # *** This is the corrected line ***
    # Use os.path.join for cross-platform compatibility.
    # Save to the relative 'outputs' directory.
    save_path = os.path.join(output_dir, f"{voice}_{int(time.time())}.wav")
    logging.info(f"Output will be saved to: {save_path}")

    try:
        # Call basic_tts which now calls infer internally
        # Note: basic_tts returns (sample_rate, audio_data), not just the path
        # We pass file_wave=save_path so infer saves the file.
        # We currently don't need the audio data itself for the response, just the path.
        _ = basic_tts( # Assign to _ as we don't use the returned audio data directly here
            ref_audio_input=reference_file,
            ref_text_input="", # F5-TTS doesn't strictly need ref text if audio is good
            gen_text_input=text,
            speed=speed,
            file_wave=save_path # Tell infer where to save the file
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Speech generated successfully in {elapsed_time:.2f} seconds. Path: {save_path}")

        # Return the path where the file was saved
        # Make sure this path is accessible/useful to the client.
        # If the client is external, you might need to return a URL instead of a filesystem path.
        # For now, returning the path as generated.
        return {"path": save_path}

    except ValueError as e: # Catch specific errors from infer/preprocess
        logging.error(f"Value error during TTS generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e: # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logging.exception(f"An unexpected error occurred during speech synthesis for voice '{voice}': {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error during speech synthesis: {e}")


if __name__ == "__main__":
    import uvicorn
    # Make host and port configurable via environment variables or args if needed
    uvicorn.run(app, host="0.0.0.0", port=7860  )

# --- END OF FILE server.py ---