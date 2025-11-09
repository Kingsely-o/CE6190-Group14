"""
Model loading utilities for Stable Diffusion v1.5 and v2.1
"""

import gc
import os
import sys
from typing import Optional, Union
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    SD_V15_MODEL_ID,
    SD_V21_MODEL_ID,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEVICE,
    USE_XFORMERS,
    ENABLE_ATTENTION_SLICING,
    ENABLE_VAE_SLICING
)
from utils.helpers import set_random_seeds, Timer


def load_sd_v15(device: str = DEVICE) -> StableDiffusionPipeline:
    """
    Load Stable Diffusion v1.5 model.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        StableDiffusionPipeline for SD v1.5
    """
    print(f"\n[MODEL] Loading Stable Diffusion v1.5 from {SD_V15_MODEL_ID}")

    with Timer("SD v1.5 loading"):
        pipeline = StableDiffusionPipeline.from_pretrained(
            SD_V15_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for evaluation
            requires_safety_checker=False,
            use_safetensors=True
        )

        # Move to device
        pipeline = pipeline.to(device)

        # Apply memory optimizations
        if device == "cuda":
            if ENABLE_ATTENTION_SLICING:
                pipeline.enable_attention_slicing()
                print("[MODEL] Enabled attention slicing")

            if ENABLE_VAE_SLICING:
                pipeline.enable_vae_slicing()
                print("[MODEL] Enabled VAE slicing")

            if USE_XFORMERS:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("[MODEL] Enabled xformers memory efficient attention")
                except Exception as e:
                    print(f"[MODEL] Could not enable xformers: {e}")

    print(f"[MODEL] SD v1.5 loaded successfully on {device}")
    return pipeline


def load_sd_v21(device: str = DEVICE) -> StableDiffusionPipeline:
    """
    Load Stable Diffusion v2.1 model.

    Args:
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        StableDiffusionPipeline for SD v2.1
    """
    print(f"\n[MODEL] Loading Stable Diffusion v2.1 from {SD_V21_MODEL_ID}")

    with Timer("SD v2.1 loading"):
        pipeline = StableDiffusionPipeline.from_pretrained(
            SD_V21_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable safety checker for evaluation
            requires_safety_checker=False,
            use_safetensors=True
        )

        # Move to device
        pipeline = pipeline.to(device)

        # Apply memory optimizations
        if device == "cuda":
            if ENABLE_ATTENTION_SLICING:
                pipeline.enable_attention_slicing()
                print("[MODEL] Enabled attention slicing")

            if ENABLE_VAE_SLICING:
                pipeline.enable_vae_slicing()
                print("[MODEL] Enabled VAE slicing")

            if USE_XFORMERS:
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("[MODEL] Enabled xformers memory efficient attention")
                except Exception as e:
                    print(f"[MODEL] Could not enable xformers: {e}")

    print(f"[MODEL] SD v2.1 loaded successfully on {device}")
    return pipeline


def generate_image(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: Optional[int] = DEFAULT_SEED,
    return_latents: bool = False
) -> Union[Image.Image, tuple]:
    """
    Generate an image using a Stable Diffusion pipeline.

    Args:
        pipeline: StableDiffusionPipeline instance
        prompt: Text prompt for generation
        negative_prompt: Optional negative prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        height: Image height in pixels
        width: Image width in pixels
        seed: Random seed for reproducibility (None for random)
        return_latents: If True, return latents along with image

    Returns:
        Generated PIL Image, or (Image, latents) if return_latents=True
    """
    # Set random seed if provided
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    else:
        generator = None

    # Generate image
    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="pil"
        )

    image = output.images[0]

    if return_latents:
        return image, output
    else:
        return image


def generate_images_batch(
    pipeline: StableDiffusionPipeline,
    prompts: list,
    **kwargs
) -> list:
    """
    Generate multiple images from a list of prompts.

    Args:
        pipeline: StableDiffusionPipeline instance
        prompts: List of text prompts
        **kwargs: Additional arguments passed to generate_image

    Returns:
        List of generated PIL Images
    """
    images = []
    for prompt in prompts:
        image = generate_image(pipeline, prompt, **kwargs)
        images.append(image)

    return images


def unload_model(pipeline: StableDiffusionPipeline):
    """
    Unload a model from memory.

    Args:
        pipeline: StableDiffusionPipeline to unload
    """
    print("\n[MODEL] Unloading model...")

    # Move to CPU to free GPU memory
    if hasattr(pipeline, 'to'):
        pipeline.to('cpu')

    # Delete pipeline
    del pipeline

    # Garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[MODEL] Model unloaded and memory cleared")


def get_model_info(pipeline: StableDiffusionPipeline) -> dict:
    """
    Get information about a loaded model.

    Args:
        pipeline: StableDiffusionPipeline instance

    Returns:
        Dictionary with model information
    """
    info = {
        "model_type": type(pipeline).__name__,
        "device": str(pipeline.device),
        "dtype": str(pipeline.unet.dtype),
    }

    # Get text encoder info
    if hasattr(pipeline, 'text_encoder'):
        info["text_encoder"] = type(pipeline.text_encoder).__name__

    # Get scheduler info
    if hasattr(pipeline, 'scheduler'):
        info["scheduler"] = type(pipeline.scheduler).__name__

    return info


def print_model_info(pipeline: StableDiffusionPipeline, model_name: str = "Model"):
    """
    Print information about a loaded model.

    Args:
        pipeline: StableDiffusionPipeline instance
        model_name: Name of the model for display
    """
    print(f"\n[MODEL INFO] {model_name}")
    print("=" * 60)

    info = get_model_info(pipeline)
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("=" * 60)


def test_generation(pipeline: StableDiffusionPipeline,
                   test_prompt: str = "a photograph of an astronaut riding a horse"):
    """
    Test image generation with a sample prompt.

    Args:
        pipeline: StableDiffusionPipeline instance
        test_prompt: Prompt to test with
    """
    print(f"\n[TEST] Generating test image with prompt: '{test_prompt}'")

    with Timer("Test generation"):
        image = generate_image(
            pipeline,
            prompt=test_prompt,
            num_inference_steps=25,  # Fewer steps for quick test
            seed=42
        )

    print(f"[TEST] Generated image size: {image.size}")
    return image


if __name__ == "__main__":
    print("Testing model loading...")

    # Test SD v1.5
    print("\n" + "=" * 80)
    print("Testing Stable Diffusion v1.5")
    print("=" * 80)

    pipeline_v15 = load_sd_v15()
    print_model_info(pipeline_v15, "Stable Diffusion v1.5")

    # Test generation
    test_image = test_generation(pipeline_v15)
    print(f"Test image generated: {test_image.size}")

    # Unload
    unload_model(pipeline_v15)

    print("\n" + "=" * 80)
    print("Testing Stable Diffusion v2.1")
    print("=" * 80)

    # Test SD v2.1
    pipeline_v21 = load_sd_v21()
    print_model_info(pipeline_v21, "Stable Diffusion v2.1")

    # Test generation
    test_image = test_generation(pipeline_v21)
    print(f"Test image generated: {test_image.size}")

    # Unload
    unload_model(pipeline_v21)

    print("\n[COMPLETE] Model loading tests finished!")
