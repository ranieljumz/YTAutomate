import os
import time
import torch
import json
import random
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional
import torchaudio
import soundfile as sf
from pydub import AudioSegment, silence
import re
from importlib.resources import files
from cached_path import cached_path
import sys
import logging
import io
import magic
from pydantic import BaseModel


# Add F5-TTS root directory to path so we can import modules
sys.path.append("/workspace/F5-TTS")
try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# load models

vocoder = load_vocoder()


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)
    
F5TTS_ema_model = load_f5tts()

chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=print,
    file_wave=None,
    seed=None,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return ref_text

    if not gen_text.strip():
        gr.Warning("Please enter text to generate.")
        return ref_text
    if seed is None:
        seed = random.randint(0, sys.maxsize)
    seed_everything(seed)
    seed = seed
        
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
    )
    if file_wave is not None:
        sf.write(file_wave, final_wave, final_sample_rate)

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

# Initialize F5-TTS model with English setting
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Copy the English reference audio to resources if it doesn't exist
resources_dir = 'resources'
os.makedirs(resources_dir, exist_ok=True)
default_ref_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
default_ref_text = "Some call me nature, others call me mother nature."

if not os.path.exists(f"{resources_dir}/default_en.wav"):
    import shutil
    shutil.copy2(default_ref_audio, f"{resources_dir}/default_en.wav")

os.makedirs("resources", exist_ok=True)

@gpu_decorator
def basic_tts(
    ref_audio_input,
    ref_text_input,
    gen_text_input,
    speed,
    file_wave
    ):
    audio_out = infer(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        tts_model_choice,
        speed=speed,
        file_wave=file_wave
    )
    return audio_out

@app.get("/synthesize_speech/")
async def synthesize_speech(
        text: str,
        voice: str,
        speed: Optional[float] = 1.0,
):
    """
    Synthesize speech from text using a specified voice and style.
    """
    start_time = time.time()
    try:
        logging.info(f'Generating speech for {voice}')

        # First try to find a WAV version
        matching_files = [f for f in os.listdir("resources") if f.startswith(voice) and f.lower().endswith('.wav')]
        logging.info(f'search {matching_files}')
        # If no WAV found, try other formats and convert
        if not matching_files:
            matching_files = [f for f in os.listdir("resources") if f.startswith(voice)]
            if not matching_files:
                raise HTTPException(status_code=400, detail="No matching voice found.")
            
            # Convert to WAV
            input_file = f'resources/{matching_files[0]}'
            wav_path = f'{output_dir}/ref_converted.wav'
            convert_to_wav(input_file, wav_path)
            reference_file = wav_path
        else:
            reference_file = f'resources/{matching_files[0]}'

        save_path = f'D:/{output_dir}/{time.time()}.wav'
        
        audio_out = basic_tts(
            ref_audio_input=reference_file,
            ref_text_input="",
            gen_text_input=text,
            speed=speed,
            file_wave=save_path
        )
        
        logging.info(audio_out)

        result = {"path":save_path}

        end_time = time.time()
        elapsed_time = end_time - start_time


        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
