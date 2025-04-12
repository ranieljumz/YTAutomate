import os
import random
import sys
import torch
import folder_paths
import logging
import utils.extra_config
import comfy.options
from typing import Sequence, Mapping, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

comfy.options.enable_args_parsing()
from comfy.cli_args import args, PerformanceFeature

app = FastAPI()

args.fast = set(PerformanceFeature)
args.use_sage_attention = True


class ImageRequest(BaseModel):
    prompt: str


@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    try:
        logging.info(
            f"Generating images: prompt : {request.prompt}"
        )
        # Validate inputs
        if not request.prompt:
            raise HTTPException(
                status_code=400, detail="Add a prompt"
            )
        result = main(request.prompt)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_value_at_index(
    obj: Union[Sequence, Mapping], index: int
) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(
        comfyui_path
    ):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        utils.extra_config.load_extra_path_config(
            extra_model_paths
        )
    else:
        print(
            "Could not find the extra_model_paths config file."
        )


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

import_custom_nodes()


def main(prompt: str):
    output_path = ""
    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_1 = unetloader.load_unet(
            unet_name="flux1-dev-fp8-e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn_fast",
        )

        dualcliploader = NODE_CLASS_MAPPINGS[
            "DualCLIPLoader"
        ]()
        dualcliploader_4 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp8_e4m3fn.safetensors",
            clip_name2="ViT-L-14-BEST-smooth-GmP-TE-only-HF-format.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS[
            "CLIPTextEncode"
        ]()
        cliptextencode_5 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(dualcliploader_4, 0),
        )

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_7 = randomnoise.get_noise(
            noise_seed=random.randint(1, 2**64)
        )

        emptysd3latentimage = NODE_CLASS_MAPPINGS[
            "EmptySD3LatentImage"
        ]()
        emptysd3latentimage_8 = (
            emptysd3latentimage.generate(
                width=576, height=1024, batch_size=1
            )
        )

        ksamplerselect = NODE_CLASS_MAPPINGS[
            "KSamplerSelect"
        ]()
        ksamplerselect_9 = ksamplerselect.get_sampler(
            sampler_name="euler"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_13 = vaeloader.load_vae(
            vae_name="ae.safetensors"
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS[
            "LoraLoaderModelOnly"
        ]()
        loraloadermodelonly_17 = (
            loraloadermodelonly.load_lora_model_only(
                lora_name="FLUX.1-TURBO-ALPHA.safetensors",
                strength_model=1.0000000000000002,
                model=get_value_at_index(unetloader_1, 0),
            )
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS[
            "UpscaleModelLoader"
        ]()
        upscalemodelloader_31 = (
            upscalemodelloader.load_model(
                model_name="RealESRGAN_x2.pth"
            )
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        teacache = NODE_CLASS_MAPPINGS["TeaCache"]()
        basicscheduler = NODE_CLASS_MAPPINGS[
            "BasicScheduler"
        ]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS[
            "SamplerCustomAdvanced"
        ]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS[
            "ImageUpscaleWithModel"
        ]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            fluxguidance_6 = fluxguidance.append(
                guidance=3.5,
                conditioning=get_value_at_index(
                    cliptextencode_5, 0
                ),
            )

            teacache_34 = teacache.apply_teacache(
                model_type="flux",
                rel_l1_thresh=0.4,
                max_skip_steps=3,
                model=get_value_at_index(
                    loraloadermodelonly_17, 0
                ),
            )

            basicscheduler_10 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=8,
                denoise=1,
                model=get_value_at_index(teacache_34, 0),
            )

            basicguider_11 = basicguider.get_guider(
                model=get_value_at_index(
                    loraloadermodelonly_17, 0
                ),
                conditioning=get_value_at_index(
                    fluxguidance_6, 0
                ),
            )

            samplercustomadvanced_12 = (
                samplercustomadvanced.sample(
                    noise=get_value_at_index(
                        randomnoise_7, 0
                    ),
                    guider=get_value_at_index(
                        basicguider_11, 0
                    ),
                    sampler=get_value_at_index(
                        ksamplerselect_9, 0
                    ),
                    sigmas=get_value_at_index(
                        basicscheduler_10, 0
                    ),
                    latent_image=get_value_at_index(
                        emptysd3latentimage_8, 0
                    ),
                )
            )

            vaedecode_14 = vaedecode.decode(
                samples=get_value_at_index(
                    samplercustomadvanced_12, 0
                ),
                vae=get_value_at_index(vaeloader_13, 0),
            )

            imageupscalewithmodel_32 = (
                imageupscalewithmodel.upscale(
                    upscale_model=get_value_at_index(
                        upscalemodelloader_31, 0
                    ),
                    image=get_value_at_index(
                        vaedecode_14, 0
                    ),
                )
            )

            saveimage_15 = saveimage.save_images(
                filename_prefix="image",
                images=get_value_at_index(
                    imageupscalewithmodel_32, 0
                ),
            )
            logging.info(saveimage_15)
            subfolder = os.path.dirname(
                os.path.normpath("flux-dev-turbo-alpha")
            )
            output_dir = folder_paths.get_output_directory()

            full_output_folder = os.path.join(
                output_dir, subfolder
            )

            output_path = os.path.join(
                full_output_folder,
                saveimage_15["ui"]["images"][0]["filename"],
            )
            logging.info(f"output path {output_path}")

    return {"status": "success", "output_path": output_path}
