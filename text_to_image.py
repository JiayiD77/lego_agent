#!/usr/bin/env python3
"""
Text to Image Generator Module
Converts textual descriptions into three-view images (front, side, top) of a model.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

logger = logging.getLogger(__name__)

@dataclass
class ViewPrompt:
    """Container for view-specific prompts and settings."""
    view_name: str
    prompt_suffix: str
    negative_prompt: str = "blurry, low quality, incomplete, cropped"
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    
    def get_full_prompt(self, base_prompt: str) -> str:
        """Combine base prompt with view-specific suffix."""
        return f"{base_prompt}, {self.prompt_suffix}"


class TextToImageGenerator:
    """Generates three-view images from textual descriptions using Stable Diffusion."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text-to-image generator.
        
        Args:
            config: Configuration dictionary with the following keys:
                - model_id: HuggingFace model ID for Stable Diffusion
                - device: Device to run inference on ('cuda' or 'cpu')
                - output_dir: Directory to save generated images
                - views: List of view configurations
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.get('output_dir', 'output/images')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define standard views if not provided
        self.views = [
            ViewPrompt(
                view_name="front",
                prompt_suffix="front view, orthographic projection, blueprint style, white background, clear details"
            ),
            ViewPrompt(
                view_name="side", 
                prompt_suffix="side view, orthographic projection, blueprint style, white background, clear details"
            ),
            ViewPrompt(
                view_name="top",
                prompt_suffix="top view, orthographic projection, blueprint style, white background, clear details"
            )
        ]
        
        # Override with config if provided
        if 'views' in config:
            self.views = [ViewPrompt(**view) for view in config['views']]
        
        # Load model
        model_id = config.get('model_id', "stabilityai/stable-diffusion-xl-base-1.0")
        logger.info(f"Loading model {model_id} on {self.device}")
        
        # For SDXL models
        if "xl" in model_id.lower():
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                use_safetensors=True
            )
        else:
            # For SD 1.x/2.x models
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None  # Disable safety checker for technical blueprint images
            )
            
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory optimization if on GPU
        if self.device == 'cuda':
            self.pipeline.enable_attention_slicing()
    
    def generate(self, text_prompt: str) -> List[Image.Image]:
        """
        Generate three-view images from a text prompt.
        
        Args:
            text_prompt: Textual description of the model to generate
            
        Returns:
            List of PIL Image objects for each view
        """
        logger.info(f"Generating images for prompt: {text_prompt}")
        
        # Enhance prompt for technical blueprint-style generation
        base_prompt = f"{text_prompt}, technical drawing, blueprint, accurate proportions, detailed"
        
        images = []
        for view in self.views:
            full_prompt = view.get_full_prompt(base_prompt)
            logger.info(f"Generating {view.view_name} view with prompt: {full_prompt}")
            
            # Generate image
            with torch.no_grad():
                image = self.pipeline(
                    prompt=full_prompt,
                    negative_prompt=view.negative_prompt,
                    guidance_scale=view.guidance_scale,
                    num_inference_steps=view.num_inference_steps,
                ).images[0]
            
            # Save image
            image_path = os.path.join(self.output_dir, f"{view.view_name}_view.png")
            image.save(image_path)
            logger.info(f"Saved {view.view_name} view to {image_path}")
            
            images.append(image)
        
        return images
    
    def generate_with_refinement(self, text_prompt: str, num_iterations: int = 3) -> List[Image.Image]:
        """
        Generate images with iterative refinement using user feedback.
        
        Args:
            text_prompt: Initial textual description
            num_iterations: Number of refinement iterations
            
        Returns:
            List of refined PIL Image objects
        """
        images = self.generate(text_prompt)
        
        # In a real implementation, this would involve user feedback
        # For this demo, we'll simulate refinement by adjusting parameters
        for i in range(1, num_iterations):
            logger.info(f"Refinement iteration {i}")
            
            # Adjust parameters for refinement (in real impl, would use feedback)
            for view in self.views:
                view.num_inference_steps += 10
                view.guidance_scale += 0.5
            
            new_images = self.generate(text_prompt)
            images = new_images  # Replace with refined images
        
        return images