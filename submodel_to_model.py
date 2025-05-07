#!/usr/bin/env python3
"""
Sub-Model to Model Assembler Module
Combines LEGO sub-models into a complete model by determining their spatial relationships.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
from dataclasses import dataclass, field
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Transform3d

# Import local modules
from part_to_lego import LegoSubModel, LegoComponent

logger = logging.getLogger(__name__)

@dataclass
class LegoModel:
    """Represents a complete LEGO model composed of sub-models."""
    id: str
    sub_models: List[Dict[str, Any]] = field(default_factory=list)
    transform: Optional[Transform3d] = None
    bounding_box: Optional[Tuple[float, float, float, float, float, float]] = None
    
    def add_sub_model(self, sub_model: LegoSubModel, position: Tuple[float, float, float], 
                     rotation: Tuple[float, float, float]) -> None:
        """Add a sub-model to the complete model with position and rotation."""
        self.sub_models.append({
            'sub_model': sub_model,
            'position': position,
            'rotation': rotation,
        })
        self._update_bounding_box()
    
    def _update_bounding_box(self) -> None:
        """Update the bounding box of the complete model."""
        if not self.sub_models:
            self.bounding_box = None
            return
            
        x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
        x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')
        
        for sub_model_data in self.sub_models:
            sub_model = sub_model_data['sub_model']
            position = sub_model_data['position']
            
            if sub_model.bounding_box is None:
                continue
                
            sm_x_min, sm_y_min, sm_z_min, sm_x_max, sm_y_max, sm_z_max = sub_model.bounding_box
            
            # Adjust by position
            x_min = min(x_min, sm_x_min + position[0])
            y_min = min(y_min, sm_y_min + position[1])
            z_min = min(z_min, sm_z_min + position[2])
            
            x_max = max(x_max, sm_x_max + position[0])
            y_max = max(y_max, sm_y_max + position[1])
            z_max = max(z_max, sm_z_max + position[2])
        
        self.bounding_box = (x_min, y_min, z_min, x_max, y_max, z_max)
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Return the center point of the model."""
        if not self.bounding_box:
            return (0, 0, 0)
            
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounding_box
        return ((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2)
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Return the dimensions of the model."""
        if not self.bounding_box:
            return (0, 0, 0)
            
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounding_box
        return (x_max - x_min, y_max - y_min, z_max - z_min)


class SubmodelAssembler:
    """Assembles LEGO sub-models into a complete model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sub-model assembler.
        
        Args:
            config: Configuration dictionary with the following keys:
                - device: Device to run inference on
                - output_dir: Directory to save assembled models
                - assembly_method: Method to use for assembly ('graph', 'optimization', 'random')
                - spacing: Spacing between sub-models
                - max_iterations: Maximum iterations for optimization
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.get('output_dir', 'output/complete_models')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.assembly_method = config.get('assembly_method', 'graph')
        self.spacing = config.get('spacing', 2.0)  # Spacing between sub-models in LEGO units
        self.max_iterations = config.get('max_iterations', 100)
    
    def assemble(self, sub_models: List[LegoSubModel]) -> LegoModel:
        """
        Assemble LEGO sub-models into a complete model.
        
        Args:
            sub_models: List of LEGO sub-models
            
        Returns:
            Complete LEGO model
        """
        logger.info(f"Assembling {len(sub_models)} sub-models into a complete model")
        
        # Create a new model with a unique ID
        model_id = f"model_{len(sub_models)}_parts"
        model = LegoModel(id=model_id)
        
        # Choose the assembly method
        if self.assembly_method == 'graph':
            self._assemble_graph_based(model, sub_models)
        elif self.assembly_method == 'optimization':
            self._assemble_optimization_based(model, sub_models)
        else:
            # Default to simple grid layout
            self._assemble_grid_layout(model, sub_models)
        
        # Save the assembled model
        self._save_model(model)
        
        logger.info(f"Assembled model with {len(model.sub_models)} sub-models")
        return model
    
    def _assemble_graph_based(self, model: LegoModel, sub_models: List[LegoSubModel]) -> None:
        """
        Assemble sub-models using a graph-based approach.
        
        Args:
            model: Target model to add the assembled sub-models to
            sub_models: List of sub-models to assemble
        """
        logger.info("Using graph-based assembly method")
        
        if not sub_models:
            return
        
        # Build a graph of sub-model relationships
        graph = self._build_relationship_graph(sub_models)
        
        # Place sub-models based on the graph
        # Start with the most connected sub-model as the center
        if len(graph) == 0:
            # No relationships, fall back to grid layout
            self._assemble_grid_layout(model, sub_models)
            return
            
        # Find the most connected sub-model
        most_connected_id = max(graph.keys(), key=lambda k: len(graph[k]))
        most_connected_idx = next(i for i, sm in enumerate(sub_models) if sm.id == most_connected_id)
        
        # Place the most connected sub-model at the center
        center_sub_model = sub_models[most_connected_idx]
        model.add_sub_model(center_sub_model, (0, 0, 0), (0, 0, 0))
        
        # Keep track of placed sub-models
        placed_sub_models = {center_sub_model.id}
        
        # Place the rest of the sub-models
        iterations = 0
        while len(placed_sub_models) < len(sub_models) and iterations < self.max_iterations:
            iterations += 1
            
            # Find a sub-model that has a relationship with a placed sub-model
            for sm_id, connected_ids in graph.items():
                if sm_id in placed_sub_models:
                    continue
                    
                connected_placed = [c_id for c_id in connected_ids if c_id in placed_sub_models]
                
                if connected_placed:
                    # Place this sub-model relative to one of its placed connections
                    reference_id = connected_placed[0]
                    
                    # Find the sub-model objects
                    sm_idx = next(i for i, sm in enumerate(sub_models) if sm.id == sm_id)
                    sm_to_place = sub_models[sm_idx]
                    
                    # Find the reference sub-model in the model
                    ref_sm_data = next(sm_data for sm_data in model.sub_models 
                                       if sm_data['sub_model'].id == reference_id)
                    ref_position = ref_sm_data['position']
                    
                    # Calculate a position relative to the reference
                    offset = self._calculate_relative_position(sm_to_place, ref_sm_data['sub_model'])
                    position = (
                        ref_position[0] + offset[0],
                        ref_position[1] + offset[1],
                        ref_position[2] + offset[2]
                    )
                    
                    # Add the sub-model to the model
                    model.add_sub_model(sm_to_place, position, (0, 0, 0))
                    placed_sub_models.add(sm_id)
                    
                    # Restart the loop to find more connected sub-models
                    break
            else:
                # If we didn't place any sub-models in this iteration, place the remaining ones in a grid
                remaining = [sm for sm in sub_models if sm.id not in placed_sub_models]
                if remaining:
                    # Calculate a position away from the existing model
                    if model.bounding_box:
                        x_min, y_min, z_min, x_max, y_max, z_max = model.bounding_box
                        base_pos = (x_max + self.spacing, y_min, z_min)
                    else:
                        base_pos = (0, 0, 0)
                    
                    # Place the remaining sub-models in a grid starting from base_pos
                    grid_size = int(np.ceil(np.sqrt(len(remaining))))
                    for i, sm in enumerate(remaining):
                        row = i // grid_size
                        col = i % grid_size
                        
                        position = (
                            base_pos[0] + col * (self.spacing + 5),
                            base_pos[1],
                            base_pos[2] + row * (self.spacing + 5)
                        )
                        
                        model.add_sub_model(sm, position, (0, 0, 0))
                        placed_sub_models.add(sm.id)
                    
                    break
    
    def _build_relationship_graph(self, sub_models: List[LegoSubModel]) -> Dict[int, List[int]]:
        """
        Build a graph of relationships between sub-models.
        
        Args:
            sub_models: List of sub-models
            
        Returns:
            Dictionary mapping sub-model IDs to lists of connected sub-model IDs
        """
        graph = {sm.id: [] for sm in sub_models}
        
        # For each pair of sub-models, determine if they have a relationship
        for i, sm1 in enumerate(sub_models):
            for j, sm2 in enumerate(sub_models):
                if i == j:
                    continue
                
                # Check if there's a relationship between the sub-models
                # In a real implementation, this would use more sophisticated heuristics
                if self._has_relationship(sm1, sm2):
                    graph[sm1.id].append(sm2.id)
        
        return graph
    
    def _has_relationship(self, sm1: LegoSubModel, sm2: LegoSubModel) -> bool:
        """
        Determine if two sub-models have a relationship.
        
        Args:
            sm1: First sub-model
            sm2: Second sub-model
            
        Returns:
            True if the sub-models have a relationship, False otherwise
        """
        # In a real implementation, this would check for semantic or spatial relationships
        # For the demo, we'll just use a simple heuristic: sub-models from the same part have a relationship
        return sm1.part_id == sm2.part_id
    
    def _calculate_relative_position(self, sm1: LegoSubModel, sm2: LegoSubModel) -> Tuple[float, float, float]:
        """
        Calculate a reasonable relative position for a sub-model.
        
        Args:
            sm1: Sub-model to position
            sm2: Reference sub-model
            
        Returns:
            Offset (x, y, z) from the reference sub-model
        """
        # In a real implementation, this would use more sophisticated techniques
        # For the demo, just use a simple offset based on bounding boxes
        
        if sm1.bounding_box is None or sm2.bounding_box is None:
            return (self.spacing, 0, 0)
            
        sm1_dims = sm1.dimensions
        sm2_dims = sm2.dimensions
        
        # Calculate an offset to place sm1 next to sm2 without overlapping
        offset_x = (sm2_dims[0] + sm1_dims[0]) / 2 + self.spacing
        
        # Place side by side if they have the same part_id, otherwise stack
        if sm1.part_id == sm2.part_id:
            return (offset_x, 0, 0)
        else:
            offset_y = (sm2_dims[1] + sm1_dims[1]) / 2 + self.spacing
            return (0, offset_y, 0)
    
    def _assemble_optimization_based(self, model: LegoModel, sub_models: List[LegoSubModel]) -> None:
        """
        Assemble sub-models using an optimization-based approach.
        
        Args:
            model: Target model to add the assembled sub-models to
            sub_models: List of sub-models to assemble
        """
        logger.info("Using optimization-based assembly method")
        
        if not sub_models:
            return
        
        # For the demo, we'll use a simplified approach
        # In a real implementation, this would use more sophisticated optimization techniques
        
        # Initial placement: all sub-models at the origin
        for sm in sub_models:
            model.add_sub_model(sm, (0, 0, 0), (0, 0, 0))
        
        # Simple optimization: iteratively move sub-models to reduce overlap
        for iteration in range(self.max_iterations):
            # Calculate pairwise overlaps
            overlap_exists = False
            
            for i, sm_data1 in enumerate(model.sub_models):
                sm1 = sm_data1['sub_model']
                pos1 = sm_data1['position']
                
                for j, sm_data2 in enumerate(model.sub_models[i+1:], i+1):
                    sm2 = sm_data2['sub_model']
                    pos2 = sm_data2['position']
                    
                    # Check for overlap
                    overlap = self._check_overlap(sm1, pos1, sm2, pos2)
                    
                    if overlap:
                        overlap_exists = True
                        
                        # Calculate repulsion vector
                        repulsion = self._calculate_repulsion(sm1, pos1, sm2, pos2)
                        
                        # Move both sub-models away from each other
                        new_pos1 = (
                            pos1[0] - repulsion[0] / 2,
                            pos1[1] - repulsion[1] / 2,
                            pos1[2] - repulsion[2] / 2
                        )
                        
                        new_pos2 = (
                            pos2[0] + repulsion[0] / 2,
                            pos2[1] + repulsion[1] / 2,
                            pos2[2] + repulsion[2] / 2
                        )
                        
                        # Update positions
                        model.sub_models[i]['position'] = new_pos1
                        model.sub_models[j]['position'] = new_pos2
            
            # If no overlaps, we're done
            if not overlap_exists:
                break
        
        # Update bounding box after optimization
        model._update_bounding_box()
    
    def _check_overlap(
        self, 
        sm1: LegoSubModel, 
        pos1: Tuple[float, float, float], 
        sm2: LegoSubModel, 
        pos2: Tuple[float, float, float]
    ) -> bool:
        """
        Check if two sub-models overlap.
        
        Args:
            sm1: First sub-model
            pos1: Position of the first sub-model
            sm2: Second sub-model
            pos2: Position of the second sub-model
            
        Returns:
            True if the sub-models overlap, False otherwise
        """
        if sm1.bounding_box is None or sm2.bounding_box is None:
            return False
            
        sm1_min_x, sm1_min_y, sm1_min_z, sm1_max_x, sm1_max_y, sm1_max_z = sm1.bounding_box
        sm2_min_x, sm2_min_y, sm2_min_z, sm2_max_x, sm2_max_y, sm2_max_z = sm2.bounding_box
        
        # Adjust by position
        sm1_min_x += pos1[0]
        sm1_min_y += pos1[1]
        sm1_min_z += pos1[2]
        sm1_max_x += pos1[0]
        sm1_max_y += pos1[1]
        sm1_max_z += pos1[2]
        
        sm2_min_x += pos2[0]
        sm2_min_y += pos2[1]
        sm2_min_z += pos2[2]
        sm2_max_x += pos2[0]
        sm2_max_y += pos2[1]
        sm2_max_z += pos2[2]
        
        # Check for overlap in all three dimensions
        overlap_x = sm1_min_x <= sm2_max_x and sm2_min_x <= sm1_max_x
        overlap_y = sm1_min_y <= sm2_max_y and sm2_min_y <= sm1_max_y
        overlap_z = sm1_min_z <= sm2_max_z and sm2_min_z <= sm1_max_z
        
        return overlap_x and overlap_y and overlap_z
    
    def _calculate_repulsion(
        self, 
        sm1: LegoSubModel, 
        pos1: Tuple[float, float, float], 
        sm2: LegoSubModel, 
        pos2: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Calculate a repulsion vector to separate overlapping sub-models.
        
        Args:
            sm1: First sub-model
            pos1: Position of the first sub-model
            sm2: Second sub-model
            pos2: Position of the second sub-model
            
        Returns:
            Repulsion vector (x, y, z)
        """
        # Calculate the vector from sm1 to sm2
        vec = (pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2])
        
        # Calculate the distance
        dist = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        
        # If the distance is zero, use a random direction
        if dist < 1e-6:
            vec = (np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5)
            dist = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        
        # Normalize the vector
        normalized = (vec[0] / dist, vec[1] / dist, vec[2] / dist)
        
        # Calculate minimum separation based on bounding boxes
        if sm1.bounding_box is None or sm2.bounding_box is None:
            min_separation = self.spacing
        else:
            sm1_dims = sm1.dimensions
            sm2_dims = sm2.dimensions
            min_separation = max(
                (sm1_dims[0] + sm2_dims[0]) / 2,
                (sm1_dims[1] + sm2_dims[1]) / 2,
                (sm1_dims[2] + sm2_dims[2]) / 2
            ) + self.spacing
        
        # Scale the vector to achieve the minimum separation
        return (
            normalized[0] * (min_separation - dist),
            normalized[1] * (min_separation - dist),
            normalized[2] * (min_separation - dist)
        )
    
    def _assemble_grid_layout(self, model: LegoModel, sub_models: List[LegoSubModel]) -> None:
        """
        Assemble sub-models in a simple grid layout.
        
        Args:
            model: Target model to add the assembled sub-models to
            sub_models: List of sub-models to assemble
        """
        logger.info("Using grid layout assembly method")
        
        if not sub_models:
            return
        
        # Arrange sub-models in a grid
        grid_size = int(np.ceil(np.sqrt(len(sub_models))))
        
        for i, sm in enumerate(sub_models):
            row = i // grid_size
            col = i % grid_size
            
            # Calculate spacing based on average sub-model size
            avg_size = np.mean([dim for sm in sub_models if sm.bounding_box for dim in sm.dimensions])
            if np.isnan(avg_size) or avg_size < 1e-6:
                avg_size = 10  # Default size if we can't calculate
            
            spacing = avg_size + self.spacing
            
            position = (col * spacing, 0, row * spacing)
            model.add_sub_model(sm, position, (0, 0, 0))
    
    def _save_model(self, model: LegoModel) -> None:
        """
        Save a complete LEGO model to disk.
        
        Args:
            model: LEGO model to save
        """
        # Create the output directory if it doesn't exist
        model_dir = os.path.join(self.output_dir, model.id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model as JSON for easy loading
        model_data = {
            'id': model.id,
            'sub_models': [
                {
                    'sub_model_id': sm_data['sub_model'].id,
                    'part_id': sm_data['sub_model'].part_id,
                    'position': sm_data['position'],
                    'rotation': sm_data['rotation'],
                }
                for sm_data in model.sub_models
            ],
            'bounding_box': model.bounding_box,
        }
        
        with open(os.path.join(model_dir, 'model.json'), 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Saved model {model.id} to {model_dir}")
    
    def export_model(self, model: LegoModel, output_path: str) -> None:
        """
        Export a LEGO model to a 3D file format.
        
        Args:
            model: LEGO model to export
            output_path: Path to save the exported model
        """
        # For a real implementation, this would export to OBJ, STL, etc.
        # For the demo, we'll just create a placeholder OBJ file
        
        with open(output_path, 'w') as f:
            f.write("# LEGO model export\n")
            f.write(f"# Model ID: {model.id}\n")
            f.write(f"# Sub-models: {len(model.sub_models)}\n")
            
            # Write vertices
            vertex_count = 0
            for i, sm_data in enumerate(model.sub_models):
                sm = sm_data['sub_model']
                pos = sm_data['position']
                
                f.write(f"# Sub-model {i}: {sm.id} at position {pos}\n")
                
                # Create a simple cube for each sub-model
                if sm.bounding_box:
                    min_x, min_y, min_z, max_x, max_y, max_z = sm.bounding_box
                    
                    # Adjust by position
                    min_x += pos[0]
                    min_y += pos[1]
                    min_z += pos[2]
                    max_x += pos[0]
                    max_y += pos[1]
                    max_z += pos[2]
                    
                    # Write the 8 vertices of the cube
                    f.write(f"v {min_x} {min_y} {min_z}\n")
                    f.write(f"v {max_x} {min_y} {min_z}\n")
                    f.write(f"v {min_x} {max_y} {min_z}\n")
                    f.write(f"v {max_x} {max_y} {min_z}\n")
                    f.write(f"v {min_x} {min_y} {max_z}\n")
                    f.write(f"v {max_x} {min_y} {max_z}\n")
                    f.write(f"v {min_x} {max_y} {max_z}\n")
                    f.write(f"v {max_x} {max_y} {max_z}\n")
                    
                    # Write the 12 triangular faces of the cube
                    # Bottom face
                    f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+4}\n")
                    f.write(f"f {vertex_count+2} {vertex_count+4} {vertex_count+3}\n")
                    
                    # Top face
                    f.write(f"f {vertex_count+5} {vertex_count+6} {vertex_count+8}\n")
                    f.write(f"f {vertex_count+6} {vertex_count+8} {vertex_count+7}\n")
                    
                    # Front face
                    f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+6}\n")
                    f.write(f"f {vertex_count+1} {vertex_count+6} {vertex_count+5}\n")
                    
                    # Back face
                    f.write(f"f {vertex_count+3} {vertex_count+4} {vertex_count+8}\n")
                    f.write(f"f {vertex_count+3} {vertex_count+8} {vertex_count+7}\n")
                    
                    # Left face
                    f.write(f"f {vertex_count+1} {vertex_count+3} {vertex_count+7}\n")
                    f.write(f"f {vertex_count+1} {vertex_count+7} {vertex_count+5}\n")
                    
                    # Right face
                    f.write(f"f {vertex_count+2} {vertex_count+4} {vertex_count+8}\n")
                    f.write(f"f {vertex_count+2} {vertex_count+8} {vertex_count+6}\n")
                    
                    vertex_count += 8
        
        logger.info(f"Exported model {model.id} to {output_path}")