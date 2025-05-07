#!/usr/bin/env python3
"""
LEGO Construction Agent - Main Controller
This script orchestrates the entire process from text input to LEGO model generation.
"""

import os
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple

# Import component modules
from text_to_image import TextToImageGenerator
from image_to_parts import ImageToPartsSegmenter
from part_to_lego import PartToLegoConverter
from submodel_to_model import SubmodelAssembler
from architect_verifier import ArchitectVerifier
from instruction_generator import InstructionBookGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegoConstructionAgent:
    """Main agent class that orchestrates the LEGO construction workflow."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LEGO Construction Agent with configurations.
        
        Args:
            config: Dictionary containing configuration parameters for each component
        """
        self.config = config
        
        # Initialize components
        self.text_to_image = TextToImageGenerator(config.get('text_to_image', {}))
        self.image_to_parts = ImageToPartsSegmenter(config.get('image_to_parts', {}))
        self.part_to_lego = PartToLegoConverter(config.get('part_to_lego', {}))
        self.submodel_assembler = SubmodelAssembler(config.get('submodel_assembler', {}))
        self.verifier = ArchitectVerifier(config.get('verifier', {}))
        self.instruction_generator = InstructionBookGenerator(config.get('instruction_generator', {}))
        
        # Create output directory if it doesn't exist
        os.makedirs(config.get('output_dir', 'output'), exist_ok=True)
        
        self.current_state = {
            'text_prompt': None,
            'images': None,
            'parts': None,
            'lego_submodels': None,
            'complete_model': None,
            'verification_result': None,
            'instruction_book': None
        }
    
    def process_text_input(self, text_prompt: str) -> Dict[str, Any]:
        """
        Process text input and generate LEGO model with instructions.
        
        Args:
            text_prompt: Text description of the desired LEGO model
            
        Returns:
            Dictionary containing results from each step of the process
        """
        logger.info(f"Processing text input: {text_prompt}")
        self.current_state['text_prompt'] = text_prompt
        
        # Step 1: Generate three-view images from text
        self.current_state['images'] = self.text_to_image.generate(text_prompt)
        logger.info(f"Generated {len(self.current_state['images'])} view images")
        
        # Step 2: Segment images into parts
        self.current_state['parts'] = self.image_to_parts.segment(self.current_state['images'])
        logger.info(f"Segmented into {len(self.current_state['parts'])} distinct parts")
        
        # Step 3: Convert parts to LEGO sub-models
        self.current_state['lego_submodels'] = self.part_to_lego.convert(self.current_state['parts'])
        logger.info(f"Converted to {len(self.current_state['lego_submodels'])} LEGO sub-models")
        
        # Step 4: Assemble sub-models into complete model
        self.current_state['complete_model'] = self.submodel_assembler.assemble(
            self.current_state['lego_submodels']
        )
        logger.info("Assembled complete LEGO model")
        
        # Step 5: Verify the assembled model
        self.current_state['verification_result'] = self.verifier.verify(
            self.current_state['complete_model'],
            self.current_state['lego_submodels']
        )
        
        if not self.current_state['verification_result']['is_valid']:
            logger.warning("Model verification failed. Attempting to fix issues...")
            self._handle_verification_failure()
        
        # Step 6: Generate instruction book
        self.current_state['instruction_book'] = self.instruction_generator.generate(
            self.current_state['parts'],
            self.current_state['complete_model']
        )
        logger.info("Generated instruction book")
        
        return self.current_state
    
    def _handle_verification_failure(self):
        """Handle verification failures by adjusting the model based on feedback."""
        issues = self.current_state['verification_result']['issues']
        
        # Try to fix submodels based on verification feedback
        updated_submodels = self.part_to_lego.fix_issues(
            self.current_state['lego_submodels'],
            issues
        )
        
        # Reassemble and verify again
        self.current_state['lego_submodels'] = updated_submodels
        self.current_state['complete_model'] = self.submodel_assembler.assemble(updated_submodels)
        self.current_state['verification_result'] = self.verifier.verify(
            self.current_state['complete_model'],
            updated_submodels
        )
        
        logger.info(f"Model fixed: {self.current_state['verification_result']['is_valid']}")
    
    def improve_with_user_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improve the model based on user feedback.
        
        Args:
            feedback: Dictionary containing user feedback for different components
            
        Returns:
            Updated state dictionary
        """
        logger.info(f"Applying user feedback: {feedback}")
        
        # Apply feedback to the appropriate component based on feedback type
        if 'text_prompt' in feedback:
            # Update text prompt and regenerate everything
            return self.process_text_input(feedback['text_prompt'])
        
        if 'images' in feedback:
            # User provided improved images
            self.current_state['images'] = feedback['images']
            
            # Rerun the pipeline from image segmentation onwards
            self.current_state['parts'] = self.image_to_parts.segment(self.current_state['images'])
            self.current_state['lego_submodels'] = self.part_to_lego.convert(self.current_state['parts'])
            self.current_state['complete_model'] = self.submodel_assembler.assemble(
                self.current_state['lego_submodels']
            )
            self.current_state['verification_result'] = self.verifier.verify(
                self.current_state['complete_model'],
                self.current_state['lego_submodels']
            )
            
            if not self.current_state['verification_result']['is_valid']:
                self._handle_verification_failure()
            
            self.current_state['instruction_book'] = self.instruction_generator.generate(
                self.current_state['parts'],
                self.current_state['complete_model']
            )
        
        if 'parts' in feedback:
            # User provided improved part segmentation
            self.current_state['parts'] = feedback['parts']
            
            # Rerun the pipeline from part conversion onwards
            self.current_state['lego_submodels'] = self.part_to_lego.convert(self.current_state['parts'])
            self.current_state['complete_model'] = self.submodel_assembler.assemble(
                self.current_state['lego_submodels']
            )
            self.current_state['verification_result'] = self.verifier.verify(
                self.current_state['complete_model'],
                self.current_state['lego_submodels']
            )
            
            if not self.current_state['verification_result']['is_valid']:
                self._handle_verification_failure()
            
            self.current_state['instruction_book'] = self.instruction_generator.generate(
                self.current_state['parts'],
                self.current_state['complete_model']
            )
        
        # Add more feedback handling as needed
        
        return self.current_state
    
    def save_results(self, output_dir: Optional[str] = None) -> str:
        """
        Save all generated artifacts to the specified directory.
        
        Args:
            output_dir: Directory to save results (uses config default if None)
            
        Returns:
            Path to the output directory
        """
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each artifact
        # Images
        for i, img in enumerate(self.current_state.get('images', [])):
            img.save(os.path.join(output_dir, f"view_{i}.png"))
        
        # Parts visualization
        self.image_to_parts.save_visualization(
            self.current_state['parts'],
            os.path.join(output_dir, "parts.png")
        )
        
        # LEGO model
        self.submodel_assembler.export_model(
            self.current_state['complete_model'],
            os.path.join(output_dir, "complete_model.obj")
        )
        
        # Instruction book
        instruction_path = os.path.join(output_dir, "instructions.pdf")
        self.instruction_generator.save(
            self.current_state['instruction_book'],
            instruction_path
        )
        
        logger.info(f"All results saved to {output_dir}")
        return output_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point for the LEGO Construction Agent."""
    parser = argparse.ArgumentParser(description="LEGO Construction Agent")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--text", type=str, help="Text prompt describing the LEGO model")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['output_dir'] = args.output
    
    agent = LegoConstructionAgent(config)
    
    if args.text:
        # Process directly from command line input
        agent.process_text_input(args.text)
        agent.save_results()
    else:
        # Interactive mode
        while True:
            text = input("Enter model description (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
                
            results = agent.process_text_input(text)
            agent.save_results()
            
            # Simple user feedback
            feedback = input("Provide feedback to improve (or press Enter to continue): ")
            if feedback:
                agent.improve_with_user_feedback({'text_prompt': feedback})
                agent.save_results()


if __name__ == "__main__":
    main()