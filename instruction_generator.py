#!/usr/bin/env python3
"""
Instruction Book Generator Module
Generates LEGO-style building instructions for the model.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# Import local modules
from part_to_lego import LegoComponent
from submodel_to_model import LegoModel

logger = logging.getLogger(__name__)

@dataclass
class BuildingStep:
    """Represents a single step in the building instructions."""
    step_number: int
    components: List[Dict[str, Any]]  # List of components added in this step
    position: Tuple[float, float, float]  # Camera position for this step
    rotation: Tuple[float, float, float]  # Camera rotation for this step
    sub_model_id: Optional[int] = None  # ID of the sub-model being built (if applicable)
    title: str = ""  # Step title/description
    notes: List[str] = field(default_factory=list)  # Additional notes for this step


@dataclass
class InstructionBook:
    """Represents a complete LEGO instruction book."""
    model_id: str
    title: str
    steps: List[BuildingStep] = field(default_factory=list)
    part_list: Dict[str, int] = field(default_factory=dict)  # Component ID -> count
    metadata: Dict[str, Any] = field(default_factory=dict)


class InstructionBookGenerator:
    """Generates LEGO-style building instructions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the instruction book generator.
        
        Args:
            config: Configuration dictionary with the following keys:
                - output_dir: Directory to save instruction books
                - page_size: Page size ('letter' or 'A4')
                - step_size: Number of components per step
                - include_part_list: Whether to include a part list at the beginning
                - include_sub_assembly: Whether to include sub-assembly instructions
        """
        self.config = config
        self.output_dir = config.get('output_dir', 'output/instructions')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.page_size = config.get('page_size', 'letter')
        self.step_size = config.get('step_size', 5)  # Number of components per step
        self.include_part_list = config.get('include_part_list', True)
        self.include_sub_assembly = config.get('include_sub_assembly', True)
    
    def generate(self, parts: List, model: LegoModel) -> InstructionBook:
        """
        Generate a LEGO instruction book for a model.
        
        Args:
            parts: List of original segmented parts (for reference)
            model: The complete LEGO model
            
        Returns:
            Complete instruction book
        """
        logger.info(f"Generating instruction book for model {model.id}")
        
        # Create a new instruction book
        book = InstructionBook(
            model_id=model.id,
            title=f"LEGO Model {model.id}"
        )
        
        # Set metadata
        book.metadata = {
            'dimensions': model.dimensions,
            'part_count': self._count_total_parts(model),
            'sub_model_count': len(model.sub_models),
            'generation_date': self._get_timestamp(),
        }
        
        # Generate the part list
        if self.include_part_list:
            book.part_list = self._generate_part_list(model)
        
        # Generate building steps
        if self.include_sub_assembly:
            # First, generate instructions for each sub-model
            for i, sm_data in enumerate(model.sub_models):
                sm = sm_data['sub_model']
                
                sub_steps = self._generate_sub_model_steps(sm, i + 1)
                book.steps.extend(sub_steps)
                
            # Then, generate steps for assembling the sub-models
            assembly_steps = self._generate_assembly_steps(model)
            book.steps.extend(assembly_steps)
        else:
            # Generate instructions for the entire model at once
            book.steps = self._generate_whole_model_steps(model)
        
        logger.info(f"Generated instruction book with {len(book.steps)} steps")
        return book
    
    def _count_total_parts(self, model: LegoModel) -> int:
        """Count the total number of LEGO components in the model."""
        total = 0
        
        for sm_data in model.sub_models:
            sm = sm_data['sub_model']
            total += len(sm.components)
        
        return total
    
    def _generate_part_list(self, model: LegoModel) -> Dict[str, int]:
        """Generate a parts list for the model."""
        part_counts = {}
        
        for sm_data in model.sub_models:
            sm = sm_data['sub_model']
            
            for comp_data in sm.components:
                comp = comp_data['component']
                part_counts[comp.id] = part_counts.get(comp.id, 0) + 1
        
        return part_counts
    
    def _generate_sub_model_steps(self, sub_model: Any, index: int) -> List[BuildingStep]:
        """Generate building steps for a sub-model."""
        steps = []
        components = sub_model.components
        
        # Break the components into steps
        step_components = []
        components_per_step = self.step_size
        
        for i in range(0, len(components), components_per_step):
            step_components.append(components[i:i+components_per_step])
        
        # Generate a step for each group of components
        for i, comps in enumerate(step_components):
            # Calculate camera position and rotation for this step
            position, rotation = self._calculate_step_view(sub_model, comps)
            
            # Create the step
            step = BuildingStep(
                step_number=len(steps) + 1,
                components=comps,
                position=position,
                rotation=rotation,
                sub_model_id=sub_model.id,
                title=f"Sub-model {index} - Step {i+1}",
                notes=[]
            )
            
            # Add special notes for the first step
            if i == 0:
                step.notes.append(f"Start building sub-model {index}")
            
            steps.append(step)
        
        return steps
    
    def _generate_assembly_steps(self, model: LegoModel) -> List[BuildingStep]:
        """Generate steps for assembling the sub-models."""
        steps = []
        
        # Add each sub-model in order
        for i, sm_data in enumerate(model.sub_models):
            sm = sm_data['sub_model']
            position = sm_data['position']
            rotation = sm_data['rotation']
            
            # Create a step for adding this sub-model
            step = BuildingStep(
                step_number=len(steps) + 1,
                components=[],  # No individual components, just the sub-model
                position=self._calculate_assembly_view(model, i),
                rotation=(30, 30, 0),  # Default isometric view
                sub_model_id=sm.id,
                title=f"Main Assembly - Step {i+1}",
                notes=[f"Add sub-model {i+1} at position {position}"]
            )
            
            steps.append(step)
        
        # Add a final overview step
        steps.append(BuildingStep(
            step_number=len(steps) + 1,
            components=[],
            position=self._calculate_assembly_view(model, len(model.sub_models)),
            rotation=(30, 30, 0),
            title="Final Assembly",
            notes=["Your LEGO model is complete!"]
        ))
        
        return steps
    
    def _generate_whole_model_steps(self, model: LegoModel) -> List[BuildingStep]:
        """Generate building steps for the entire model at once."""
        steps = []
        all_components = []
        
        # Collect all components from all sub-models
        for sm_data in model.sub_models:
            sm = sm_data['sub_model']
            position = sm_data['position']
            
            for comp_data in sm.components:
                # Adjust component position by sub-model position
                adjusted_position = (
                    comp_data['position'][0] + position[0],
                    comp_data['position'][1] + position[1],
                    comp_data['position'][2] + position[2]
                )
                
                all_components.append({
                    'component': comp_data['component'],
                    'position': adjusted_position,
                    'rotation': comp_data['rotation'],
                })
        
        # Sort components by height (y-coordinate) for a bottom-up build
        all_components.sort(key=lambda c: c['position'][1])
        
        # Break the components into steps
        step_components = []
        components_per_step = self.step_size
        
        for i in range(0, len(all_components), components_per_step):
            step_components.append(all_components[i:i+components_per_step])
        
        # Generate a step for each group of components
        for i, comps in enumerate(step_components):
            # Calculate camera position and rotation for this step
            # For simplicity, use a standard isometric view with progressive zoom
            zoom_factor = min(1.0, 0.5 + (i / len(step_components)) * 0.5)
            position = (
                model.center[0],
                model.center[1],
                model.center[2] + (1.0 / zoom_factor) * max(model.dimensions) * 1.5
            )
            
            # Create the step
            step = BuildingStep(
                step_number=i + 1,
                components=comps,
                position=position,
                rotation=(30, 30, 0),  # Standard isometric view
                title=f"Step {i+1}",
                notes=[]
            )
            
            # Add special notes for the first and last step
            if i == 0:
                step.notes.append("Start building the model from the bottom up")
            elif i == len(step_components) - 1:
                step.notes.append("Your LEGO model is complete!")
            
            steps.append(step)
        
        return steps
    
    def _calculate_step_view(self, sub_model: Any, components: List[Dict[str, Any]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate the ideal camera position and rotation for a building step."""
        # For simplicity, use a standard isometric view
        # In a real implementation, this would consider the components being added
        
        # Calculate the center of the sub-model or the components being added
        if not components:
            center = sub_model.center
        else:
            # Calculate center of the components being added
            positions = [comp_data['position'] for comp_data in components]
            center = np.mean(positions, axis=0)
        
        # Calculate a position looking at the center from an isometric angle
        view_distance = max(sub_model.dimensions) * 2.0
        position = (
            center[0] - view_distance,
            center[1] + view_distance,
            center[2] - view_distance
        )
        
        rotation = (30, 30, 0)  # Standard isometric view
        
        return position, rotation
    
    def _calculate_assembly_view(self, model: LegoModel, step_index: int) -> Tuple[float, float, float]:
        """Calculate the camera position for an assembly step."""
        # Use a standard isometric view that progressively zooms out
        zoom_factor = min(1.0, 0.5 + (step_index / len(model.sub_models)) * 0.5)
        position = (
            model.center[0],
            model.center[1],
            model.center[2] + (1.0 / zoom_factor) * max(model.dimensions) * 1.5
        )
        
        return position
    
    def save(self, book: InstructionBook, output_path: str) -> None:
        """
        Save an instruction book to a PDF file.
        
        Args:
            book: Instruction book to save
            output_path: Path to save the PDF file
        """
        logger.info(f"Saving instruction book to {output_path}")
        
        # Choose page size
        page_size = A4 if self.page_size.lower() == 'a4' else letter
        
        # Create a PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=page_size,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['Normal']
        
        # Create a centered style for step titles
        centered_style = ParagraphStyle(
            'Centered',
            parent=normal_style,
            alignment=TA_CENTER,
        )
        
        # Create content elements
        elements = []
        
        # Add title page
        elements.append(Paragraph(book.title, title_style))
        elements.append(Spacer(1, 0.5*inch))
        
        # Add metadata
        dimensions = book.metadata.get('dimensions', (0, 0, 0))
        dimensions_str = f"{dimensions[0]:.1f} x {dimensions[1]:.1f} x {dimensions[2]:.1f} LEGO units"
        
        metadata_text = [
            f"Parts: {book.metadata.get('part_count', 0)}",
            f"Sub-models: {book.metadata.get('sub_model_count', 0)}",
            f"Dimensions: {dimensions_str}",
            f"Generated: {book.metadata.get('generation_date', '')}",
        ]
        
        for line in metadata_text:
            elements.append(Paragraph(line, normal_style))
        
        elements.append(Spacer(1, inch))
        
        # Add part list if included
        if book.part_list:
            elements.append(Paragraph("Parts List", heading_style))
            elements.append(Spacer(1, 0.25*inch))
            
            # Create a table for the parts list
            part_data = [["Part ID", "Description", "Quantity"]]
            
            for part_id, count in book.part_list.items():
                # In a real implementation, we would look up part descriptions
                part_data.append([part_id, f"LEGO part {part_id}", str(count)])
            
            part_table = Table(part_data, colWidths=[1*inch, 3*inch, 1*inch])
            part_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(part_table)
            elements.append(Spacer(1, 0.5*inch))
        
        # Add a page break before the instructions
        elements.append(Spacer(1, inch))
        
        # Add the building steps
        for step in book.steps:
            elements.append(Paragraph(f"Step {step.step_number}: {step.title}", heading_style))
            elements.append(Spacer(1, 0.25*inch))
            
            # In a real implementation, we would render an image of the step
            # For the demo, we'll use a placeholder
            step_image = self._render_step_image(step)
            if step_image:
                elements.append(Image(step_image, width=6*inch, height=4*inch))
            
            # Add step notes
            for note in step.notes:
                elements.append(Paragraph(note, normal_style))
            
            elements.append(Spacer(1, 0.5*inch))
        
        # Build the PDF
        doc.build(elements)
        
        logger.info(f"Saved instruction book to {output_path}")
    
    def _render_step_image(self, step: BuildingStep) -> Optional[str]:
        """
        Render an image for a building step.
        
        Args:
            step: Building step to render
            
        Returns:
            Path to the rendered image file or None if rendering failed
        """
        # For the demo, we'll create a simple placeholder image
        # In a real implementation, this would render a 3D view of the step
        
        # Create a directory for step images if it doesn't exist
        step_dir = os.path.join(self.output_dir, 'steps')
        os.makedirs(step_dir, exist_ok=True)
        
        # Generate a filename for this step
        filename = os.path.join(step_dir, f"step_{step.step_number}.png")
        
        # Create a simple placeholder image
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Set background color
            ax.set_facecolor('#f0f0f0')
            
            # Draw a border
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2, transform=ax.transAxes))
            
            # Add step title
            ax.text(0.5, 0.9, f"Step {step.step_number}: {step.title}", ha='center', va='center', fontsize=14)
            
            # Add component information
            if step.components:
                component_text = f"Add {len(step.components)} component(s)"
                ax.text(0.5, 0.5, component_text, ha='center', va='center', fontsize=12)
                
                # In a real implementation, we would render the actual components
            else:
                # For assembly steps
                ax.text(0.5, 0.5, f"Add sub-model {step.sub_model_id if step.sub_model_id is not None else ''}", 
                       ha='center', va='center', fontsize=12)
            
            # Add notes at the bottom
            note_y = 0.2
            for note in step.notes:
                ax.text(0.5, note_y, note, ha='center', va='center', fontsize=10, style='italic')
                note_y -= 0.05
            
            # Add camera info at the bottom
            camera_text = f"View from: {step.position}"
            ax.text(0.5, 0.1, camera_text, ha='center', va='center', fontsize=8, color='gray')
            
            # Remove axes
            ax.axis('off')
            
            # Save the figure
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return filename
        except Exception as e:
            logger.error(f"Error rendering step image: {e}")
            return None
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp as a string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")