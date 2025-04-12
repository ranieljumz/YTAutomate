import asyncio
from asyncio.log import logger
import logging
import os
import aiofiles
import httpx
from pydantic import BaseModel
import websockets  # Use the 'websocket-client' library for simpler sync usage, or 'websockets' for asyncio
import uuid
import json
from fastapi import FastAPI, HTTPException, Request

import folder_paths

OUTPUT_DIR = "/mnt/z/comfyui/output"

if os.path.exists(OUTPUT_DIR):
    print(f"Directory exists: {OUTPUT_DIR}")
else:
    print(f"Directory does NOT exist: {OUTPUT_DIR}")
    # You might need to create it if it doesn't exist:
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

# --- Configuration ---
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
CLIENT_ID = str(uuid.uuid4())  # Generate a unique client ID

# --- Workflow Loading ---
# Load your workflow JSON from the file saved via "Save (API Format)"

# Load your workflow JSON from the file saved via "Save (API Format)"
# Get the directory where the script itself is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the JSON file relative to the script's directory
# Assumes workflow_api.json is in the SAME directory as the script
file_path = os.path.join(script_dir, "workflow_api.json")
try:
    with open(file_path, "r") as f:
        BASE_WORKFLOW = json.load(f)
    logger.info(
        "Loaded base workflow from workflow_api.json"
    )
except FileNotFoundError:
    logger.error(
        "Error: workflow_api.json not found. Please save your workflow in API format."
    )
    # Define a fallback basic workflow if needed, but loading is preferred
    # **IMPORTANT**: Ensure this fallback matches your setup if used
    BASE_WORKFLOW = {
        "1": {
            "inputs": {
                "unet_name": "flux1-dev-fp8-e4m3fn.safetensors",
                "weight_dtype": "fp8_e4m3fn_fast",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "UNETLoader"},
        },
        "4": {
            "inputs": {
                "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                "clip_name2": "ViT-L-14-BEST-smooth-GmP-TE-only-HF-format.safetensors",
                "type": "flux",
                "device": "default",
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "DualCLIPLoader"},
        },
        "5": {
            "inputs": {
                "text": "A fluffy, 50% white, 50% gray Ragdoll cat, with large, expressive eyes, rests peacefully on a soft, plush throw blanket, creating a gentle contrast between the light and dark hues of its fur.  Its long, silky coat shimmers in the soft light, and its paws, subtly tipped in the same gray tones, have a delicate texture as they rest gently against the blanket. The cat's posture is relaxed, its tail gently swishing in slow, languid movements.",
                "clip": ["4", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Positive Prompt)"
            },
        },
        "6": {
            "inputs": {
                "guidance": 3.5,
                "conditioning": ["5", 0],
            },
            "class_type": "FluxGuidance",
            "_meta": {"title": "FluxGuidance"},
        },
        "7": {
            "inputs": {"noise_seed": 245726646861906},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "8": {
            "inputs": {
                "width": 576,
                "height": 1024,
                "batch_size": 1,
            },
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "EmptySD3LatentImage"},
        },
        "9": {
            "inputs": {"sampler_name": "euler"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "10": {
            "inputs": {
                "scheduler": "simple",
                "steps": 20,
                "denoise": 1,
                "model": ["34", 0],
            },
            "class_type": "BasicScheduler",
            "_meta": {"title": "BasicScheduler"},
        },
        "11": {
            "inputs": {
                "model": ["17", 0],
                "conditioning": ["6", 0],
            },
            "class_type": "BasicGuider",
            "_meta": {"title": "BasicGuider"},
        },
        "12": {
            "inputs": {
                "noise": ["7", 0],
                "guider": ["11", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["8", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "13": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "14": {
            "inputs": {
                "samples": ["12", 0],
                "vae": ["13", 0],
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "15": {
            "inputs": {
                "filename_prefix": "image",
                "images": ["32", 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "17": {
            "inputs": {
                "lora_name": "FLUX.1-TURBO-ALPHA.safetensors",
                "strength_model": 1.0000000000000002,
                "model": ["1", 0],
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "LoraLoaderModelOnly"},
        },
        "31": {
            "inputs": {"model_name": "RealESRGAN_x2.pth"},
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "Load Upscale Model"},
        },
        "32": {
            "inputs": {
                "upscale_model": ["31", 0],
                "image": ["14", 0],
            },
            "class_type": "ImageUpscaleWithModel",
            "_meta": {
                "title": "Upscale Image (using Model)"
            },
        },
        "34": {
            "inputs": {
                "model_type": "flux",
                "rel_l1_thresh": 0.4,
                "max_skip_steps": 3,
                "model": ["17", 0],
            },
            "class_type": "TeaCache",
            "_meta": {"title": "TeaCache"},
        },
    }
    logger.warning(
        "Using fallback workflow. Ensure node IDs and inputs are correct."
    )

# --- WebSocket Communication Functions ---


async def queue_prompt(
    prompt_workflow,
    client_id,
    host=COMFYUI_HOST,
    port=COMFYUI_PORT,
):
    """Sends the workflow prompt to the ComfyUI server via HTTP."""
    url = f"http://{host}:{port}/prompt"
    payload = {
        "prompt": prompt_workflow,
        "client_id": client_id,
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=60,
            )  # Added timeout
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            logger.info(
                f"Prompt queued successfully. Prompt ID: {data.get('prompt_id')}"
            )
            return data
        except httpx.RequestError as e:
            logger.error(
                f"Error queuing prompt (HTTP request failed): {e}"
            )
            return None
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding queue response JSON. Response text: {response.text}"
            )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during prompt queueing: {e}"
            )
            return None


async def get_history(
    prompt_id, host=COMFYUI_HOST, port=COMFYUI_PORT
):
    """Retrieves the execution history for a given prompt ID via HTTP."""
    url = f"http://{host}:{port}/history/{prompt_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data
        except httpx.RequestError as e:
            logger.error(
                f"Error getting history for prompt {prompt_id}: {e}"
            )
            return None
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding history response JSON for prompt {prompt_id}. Response text: {response.text}"
            )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during history fetch for {prompt_id}: {e}"
            )
            return None


async def get_image_data(
    filename,
    subfolder,
    folder_type,
    host=COMFYUI_HOST,
    port=COMFYUI_PORT,
):
    """Fetches image data using the /view endpoint via HTTP."""
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
    }
    url = f"http://{host}:{port}/view"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url, params=params, timeout=120
            )  # Longer timeout for image download
            response.raise_for_status()
            return (
                response.content
            )  # Return raw image bytes
        except httpx.RequestError as e:
            logger.error(
                f"Error fetching image {filename}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during image fetch for {filename}: {e}"
            )
            return None


# --- Main Execution Logic ---


class ImageRequest(BaseModel):
    prompt: str


class GenerationResponse(BaseModel):
    image_urls: list[
        str
    ]  # List of URLs pointing to the generated images
    message: str


@app.post(
    "/generate_image", response_model=GenerationResponse
)
async def generate_image(
    request: ImageRequest, http_request: Request
):
    try:
        logging.info(
            f"Generating images: prompt : {request.prompt}"
        )
        # Validate inputs
        if not request.prompt:
            raise HTTPException(
                status_code=400, detail="Add a prompt"
            )
        result = await run_prompt(request.prompt)

        if not result:
            logger.warning(
                "Image generation did not produce any saved files."
            )
            raise HTTPException(
                status_code=500,
                detail="Image generation failed or produced no output.",
            )
        output_path = os.path.join(
                "Z:/comfyui/output", result[0]
        )
        image_urls = [
            f"{output_path}"
            for filename in result
        ]

        logger.info(
            f"Generation successful. Returning URLs: {image_urls}"
        )
        return GenerationResponse(
            image_urls=image_urls,
            message="Images generated successfully.",
        )

    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except ConnectionRefusedError:
        logger.error(
            "Connection to ComfyUI WebSocket refused. Is ComfyUI running?"
        )
        raise HTTPException(
            status_code=503,
            detail="ComfyUI service unavailable.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}",
        )


async def run_prompt(prompt: str):
    """
    Connects to ComfyUI via WebSocket, runs the workflow, and returns saved image paths.

    Args:
        prompt: The positive prompt text.
        neg_prompt: The negative prompt text.
        seed: Optional seed. If None, a random seed is used.

    Returns:
        A list of paths to the saved images, relative to the OUTPUT_DIR.
        Returns an empty list if generation fails or no images are produced.
    """
    client_id = str(uuid.uuid4())
    ws_url = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws?clientId={client_id}"
    saved_image_paths = []

    # --- Modify the workflow ---
    workflow = json.loads(
        json.dumps(BASE_WORKFLOW)
    )  # Deep copy the base workflow
    # Set a unique filename prefix in the SaveImage node to help identify outputs
    if "5" in workflow:
        workflow["5"]["inputs"]["text"] = prompt
        logger.info(f"Set positive prompt (Node 5)")
    else:
        logger.warning(
            f"Positive prompt node ID '5' not found in workflow."
        )
    # --- Queue Prompt via HTTP ---
    queue_response = await queue_prompt(workflow, client_id)
    if (
        not queue_response
        or "prompt_id" not in queue_response
    ):
        logger.error("Failed to queue prompt.")
        return []  # Return empty list on failure

    prompt_id = queue_response["prompt_id"]

    # --- Connect to WebSocket and wait for execution ---
    try:
        async with websockets.connect(
            ws_url, ping_interval=None
        ) as websocket:  # Disable automatic pings if needed
            logger.info(f"WebSocket connected to {ws_url}")
            execution_finished = False
            while not execution_finished:
                try:
                    # Set a timeout for receiving messages to prevent hanging indefinitely
                    message_json = await asyncio.wait_for(
                        websocket.recv(), timeout=300.0
                    )  # 5 min timeout
                    message = json.loads(message_json)

                    if message.get("type") == "status":
                        data = message.get("data", {})
                        # logger.info(f"Status update: Queue remaining {data.get('status', {}).get('exec_info', {}).get('queue_remaining', 'N/A')}")
                        pass  # Optionally log status

                    elif message.get("type") == "progress":
                        data = message.get("data", {})
                        logger.info(
                            f"Progress: {data.get('value', 0)}/{data.get('max', 0)}"
                        )

                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if (
                            data.get("prompt_id")
                            == prompt_id
                        ):
                            node_id = data.get("node")
                            if (
                                node_id is None
                            ):  # Indicates the workflow execution for this prompt_id is likely finished
                                logger.info(
                                    f"Execution finished for prompt {prompt_id} (node is None)."
                                )
                                execution_finished = True
                            # else:
                            # logger.info(f"Executing node: {node_id}")

                    elif message.get("type") == "executed":
                        data = message.get("data", {})
                        if (
                            data.get("prompt_id")
                            == prompt_id
                        ):
                            logger.info(
                                f"Received 'executed' message for prompt {prompt_id}. Outputs might be available."
                            )
                            # Optionally check message['data']['output'] here, but history is more reliable
                            execution_finished = True  # Assume finished if we get the final executed message

                    elif (
                        message.get("type")
                        == "execution_error"
                    ):
                        data = message.get("data", {})
                        if (
                            data.get("prompt_id")
                            == prompt_id
                        ):
                            logger.error(
                                f"Execution error for prompt {prompt_id}: {data}"
                            )
                            execution_finished = True  # Stop waiting on error

                except asyncio.TimeoutError:
                    logger.error(
                        f"WebSocket receive timeout for prompt {prompt_id}. Assuming execution stalled or finished."
                    )
                    execution_finished = (
                        True  # Break loop on timeout
                    )
                except (
                    websockets.exceptions.ConnectionClosedOK
                ):
                    logger.info(
                        "WebSocket connection closed normally."
                    )
                    execution_finished = (
                        True  # Stop if connection closes
                    )
                except (
                    websockets.exceptions.ConnectionClosedError
                ) as e:
                    logger.error(
                        f"WebSocket connection closed with error: {e}"
                    )
                    execution_finished = (
                        True  # Stop on error
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Received non-JSON WebSocket message: {message_json}"
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error in WebSocket loop for prompt {prompt_id}: {e}"
                    )
                    execution_finished = (
                        True  # Stop on unexpected errors
                    )

    except (
        websockets.exceptions.InvalidURI,
        websockets.exceptions.WebSocketException,
        ConnectionRefusedError,
    ) as e:
        logger.error(f"WebSocket connection failed: {e}")
        return []  # Return empty list if connection fails
    except Exception as e:
        logger.error(
            f"An unexpected error occurred before or during WebSocket connection for prompt {prompt_id}: {e}"
        )
        return []

    logger.info(
        f"WebSocket loop finished for prompt {prompt_id}. Fetching history."
    )

    # --- Fetch History and Images ---
    history_data = await get_history(prompt_id)
    if not history_data or prompt_id not in history_data:
        logger.error(
            f"Could not retrieve valid history for prompt ID {prompt_id}."
        )
        return []

    prompt_history = history_data[prompt_id]
    outputs = prompt_history.get("outputs", {})
    logger.info(
        f"History retrieved. Processing outputs for nodes: {list(outputs.keys())}"
    )

    image_fetch_tasks = []
    output_details = []  # Store details needed for saving

    # --- Specify the ID of the node whose output you want ---
    # --- Ensure this matches your actual SaveImage node ID in the workflow ---
    TARGET_SAVE_NODE_ID = "15" # <<< Your SaveImage node ID from BASE_WORKFLOW

    if TARGET_SAVE_NODE_ID in outputs:
        node_output = outputs[TARGET_SAVE_NODE_ID] # Get output for ONLY this node
        if 'images' in node_output:
            logger.info(f"Processing images specifically from target node {TARGET_SAVE_NODE_ID}")
            for image_info in node_output['images']: # Iterate images from ONLY this node
                filename = image_info.get('filename')
                subfolder = image_info.get('subfolder', '')
                folder_type = image_info.get('type', 'output')

                if filename:
                    logger.info(f"  Queueing fetch for image: {filename} (from node {TARGET_SAVE_NODE_ID})")
                    output_details.append({"filename": filename})
                    image_fetch_tasks.append(
                        get_image_data(filename, subfolder, folder_type)
                    )
                else:
                    logger.warning("  Skipping image entry with missing filename in target node output.")
        else:
            logger.warning(f"Target node {TARGET_SAVE_NODE_ID} found in history, but has no 'images' key.")
    else:
        # If the target node isn't found, it's an issue.
        logger.warning(f"Target SaveImage node {TARGET_SAVE_NODE_ID} not found in history outputs! Check workflow execution.")
        logger.warning(f"Available nodes with outputs: {list(outputs.keys())}")
        # Consider returning [] or raising an error if the save node output is critical

    # Fetch all images concurrently
    if image_fetch_tasks:
        logger.info(
            f"Fetching {len(image_fetch_tasks)} images concurrently..."
        )
        image_datas = await asyncio.gather(
            *image_fetch_tasks
        )
    else:
        image_datas = []
        logger.warning(
            "No valid image entries found in history outputs."
        )

    # --- Save Fetched Images ---
    save_tasks = []
    for i, image_data in enumerate(image_datas):
        if image_data:
            try:
                original_filename = output_details[i][
                    "filename"
                ]
                # Use a more specific unique name for saving locally
                # Use prompt_id and index to ensure uniqueness
                save_filename = (
                    f"{prompt_id}_{i+1}_{original_filename}"
                )
                save_path = os.path.join(
                    OUTPUT_DIR, save_filename
                )

                logger.info(
                    f"  Preparing to save image {i+1} to {save_path}"
                )

                # Use aiofiles for async saving
                async def save_image(data, path):
                    async with aiofiles.open(
                        path, "wb"
                    ) as f:
                        await f.write(data)
                    logger.info(
                        f"  Successfully saved image to {path}"
                    )
                    # Return the filename part for the API response
                    return os.path.basename(path)

                save_tasks.append(
                    save_image(image_data, save_path)
                )

            except Exception as e:
                logger.error(
                    f"  Error processing image data before saving: {e}"
                )
        else:
            logger.warning(
                f"  Failed to fetch image data for entry {i} (Original: {output_details[i]['filename']}). Skipping save."
            )

    # Wait for all save operations to complete
    if save_tasks:
        logger.info(
            f"Saving {len(save_tasks)} images asynchronously..."
        )
        saved_image_paths = await asyncio.gather(
            *save_tasks
        )
        # Filter out None results if any save task failed internally and returned None
        saved_image_paths = [
            p for p in saved_image_paths if p
        ]
        logger.info(
            f"Finished saving images. Result paths: {saved_image_paths}"
        )
    else:
        logger.info(
            "No images were fetched or processed for saving."
        )

    return (
        saved_image_paths  # Return list of saved filenames
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8186)