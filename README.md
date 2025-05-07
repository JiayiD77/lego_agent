# LEGO Construction Agent

A modular Python framework for converting text descriptions into complete LEGO models with building instructions.

## Architecture Overview

This system takes a text description and generates a complete LEGO model through a multi-stage pipeline:

1. **Text to Image**: Converts textual descriptions into three-view images (front, side, top)
2. **Image to Parts**: Segments the images into individual components/parts
3. **Part to LEGO Sub-model**: Converts segmented parts into LEGO sub-models
4. **Sub-model to Model**: Assembles sub-models into a complete model
5. **Architect Verifier**: Validates the model for structural integrity and buildability
6. **Instruction Generator**: Creates LEGO-style building instructions

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/lego-construction-agent.git
cd lego-construction-agent

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the Segment Anything Model (SAM) checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P data/
```

## Usage

### Basic Usage

```bash
# Run with default parameters and the spaceship example
python demo.py --example spaceship

# Run with a custom text prompt
python demo.py --example custom --custom_text "A medieval castle with towers and a moat"

# Specify a different configuration file
python demo.py --config my_config.json --example house
```

### Interactive Mode

```bash
# Run in interactive mode
python main.py

# Follow the prompts to enter your model description and provide feedback
```

### Configuration

Edit the `config.json` file to customize various aspects of the pipeline:
- Model IDs for text-to-image generation
- Segmentation parameters
- LEGO component database path
- Assembly method
- Verification strictness
- Instruction book format

## Components

### Text to Image Generator

Converts text descriptions into three-view images (front, side, top) using Stable Diffusion.

```python
from text_to_image import TextToImageGenerator

generator = TextToImageGenerator(config)
images = generator.generate("A spaceship with wings and engines")
```

### Image to Parts Segmenter

Segments three-view images into individual parts using the Segment Anything Model (SAM).

```python
from image_to_parts import ImageToPartsSegmenter

segmenter = ImageToPartsSegmenter(config)
parts = segmenter.segment(images)
```

### Part to LEGO Converter

Converts segmented parts into LEGO sub-models by matching them with appropriate LEGO components.

```python
from part_to_lego import PartToLegoConverter

converter = PartToLegoConverter(config)
lego_submodels = converter.convert(parts)
```

### Sub-model Assembler

Combines LEGO sub-models into a complete model by determining their spatial relationships.

```python
from submodel_to_model import SubmodelAssembler

assembler = SubmodelAssembler(config)
complete_model = assembler.assemble(lego_submodels)
```

### Architect Verifier

Verifies LEGO models for structural integrity and buildability.

```python
from architect_verifier import ArchitectVerifier

verifier = ArchitectVerifier(config)
verification_result = verifier.verify(complete_model, lego_submodels)
```

### Instruction Book Generator

Generates LEGO-style building instructions for the model.

```python
from instruction_generator import InstructionBookGenerator

generator = InstructionBookGenerator(config)
instruction_book = generator.generate(parts, complete_model)
generator.save(instruction_book, "instructions.pdf")
```

## Output Structure

The output directory contains:
- Three-view images (front, side, top)
- Segmented parts visualization
- LEGO sub-models (JSON and OBJ files)
- Complete model (JSON and OBJ files)
- Verification results
- Building instructions (PDF)

## Extending the System

### Adding Custom LEGO Components

Add new components to the `data/lego_components.json` file:

```json
{
  "3001": {
    "id": "3001",
    "name": "Brick 2 x 4",
    "category": "Brick",
    "dimensions": [4, 1, 2],
    "mesh_path": "meshes/3001.obj",
    "connection_points": [[0.5, 1, 0.5], [1.5, 1, 0.5], ...],
    "color": [0.7, 0.7, 0.7],
    "is_structural": true,
    "min_connections": 1,
    "max_connections": 10
  },
  ...
}
```

### Implementing Custom Assembly Methods

Extend the `SubmodelAssembler` class with your own assembly method:

```python
def _my_custom_assembly(self, model, sub_models):
    """Implement your custom assembly algorithm here."""
    # Your code here
    pass
```

Then update the `config.json` file to use your method:

```json
{
  "submodel_assembler": {
    "assembly_method": "my_custom",
    ...
  }
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) for text-to-image generation
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything) for image segmentation
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for 3D operations