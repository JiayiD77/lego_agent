#!/usr/bin/env python3
"""
LEGO Construction Agent - Demo Script
This script demonstrates the complete pipeline of the LEGO Construction Agent.
"""

import os
import argparse
import json
import logging
from main import LegoConstructionAgent, load_config

def main():
    """Main demo function for the LEGO Construction Agent."""
    parser = argparse.ArgumentParser(description="LEGO Construction Agent Demo")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--example", type=str, default="spaceship", 
                        choices=["spaceship", "house", "car", "robot", "custom"],
                        help="Example to run")
    parser.add_argument("--custom_text", type=str, help="Custom text prompt (if example=custom)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    config['output_dir'] = args.output
    
    # Initialize the agent
    agent = LegoConstructionAgent(config)
    
    # Select the example text prompt
    text_prompt = get_example_prompt(args.example, args.custom_text)
    
    if not text_prompt:
        logger.error("No text prompt provided. Exiting.")
        return
    
    logger.info(f"Running demo with prompt: '{text_prompt}'")
    
    # Process the text input
    results = agent.process_text_input(text_prompt)
    
    # Save the results
    output_dir = agent.save_results()
    
    logger.info(f"Demo completed successfully. Results saved to {output_dir}")
    
    # Print a summary of the results
    print_summary(results, output_dir)
    
    # Simulate some user feedback
    if args.example != "custom":
        user_feedback = get_example_feedback(args.example)
        
        if user_feedback:
            logger.info(f"Applying user feedback: '{user_feedback}'")
            
            # Process the user feedback
            updated_results = agent.improve_with_user_feedback({'text_prompt': user_feedback})
            
            # Save the updated results
            updated_output_dir = agent.save_results(os.path.join(args.output, "improved"))
            
            logger.info(f"Updated results saved to {updated_output_dir}")
            
            # Print a summary of the updated results
            print_summary(updated_results, updated_output_dir)


def get_example_prompt(example: str, custom_text: str = None) -> str:
    """Get an example text prompt based on the selected example."""
    if example == "custom" and custom_text:
        return custom_text
    
    prompts = {
        "spaceship": "A sleek futuristic spacecraft with wings and a cockpit. It has two engines on the back and landing gear underneath.",
        "house": "A two-story house with a pitched roof, windows, a front door, and a small garden.",
        "car": "A sporty race car with a low profile, large wheels, a spoiler, and a streamlined body.",
        "robot": "A humanoid robot with articulated arms, legs, a head with glowing eyes, and a control panel on its chest."
    }
    
    return prompts.get(example, "")


def get_example_feedback(example: str) -> str:
    """Get example user feedback based on the selected example."""
    feedback = {
        "spaceship": "Make the spacecraft more aerodynamic and add more details to the engines.",
        "house": "Add a garage and a chimney to the house, and make the garden larger.",
        "car": "Make the wheels bigger and add headlights and a racing stripe.",
        "robot": "Make the robot taller and add more features like antennas and tools in its hands."
    }
    
    return feedback.get(example, "")


def print_summary(results: dict, output_dir: str) -> None:
    """Print a summary of the results."""
    print("\n" + "="*50)
    print(f"LEGO Construction Agent - Results Summary")
    print("="*50)
    
    print(f"\nText Prompt: '{results['text_prompt']}'")
    print(f"Output Directory: {output_dir}")
    
    print(f"\nGenerated {len(results['images'])} view images")
    print(f"Segmented into {len(results['parts'])} distinct parts")
    print(f"Converted to {len(results['lego_submodels'])} LEGO sub-models")
    
    # Print verification result
    verification = results['verification_result']
    print(f"\nVerification Result: {'✓ Valid' if verification['is_valid'] else '✗ Invalid'}")
    print(f"Issues: {verification['error_count']} errors, {verification['warning_count']} warnings")
    
    if verification['issues']:
        print("\nTop Issues:")
        for i, issue in enumerate(verification['issues'][:3]):
            print(f"  - {issue['type']}: {issue['message']}")
            if i >= 2 and len(verification['issues']) > 3:
                print(f"  - ... and {len(verification['issues']) - 3} more")
                break
    
    # Print instruction book info
    print(f"\nGenerated instruction book with steps")
    
    print("\nFiles Generated:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for file in files:
            print(f"{indent}  {file}")
            if len(files) > 5:
                print(f"{indent}  ... and {len(files) - 5} more files")
                break
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main()