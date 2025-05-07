#!/usr/bin/env python3
"""
Part to LEGO Sub-Model Converter Module
Converts segmented parts into LEGO sub-models by matching them with appropriate LEGO components.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
from dataclasses import dataclass, field
import torch
from PIL import Image

# For 3D reconstruction and LEGO part matching
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import Transform3d

# Import local modules
from image_to_parts import Part

logger = logging.getLogger(__name__)

@dataclass
class LegoComponent:
    """Represents a LEGO component with its properties."""
    id: str
    name: str
    category: str
    dimensions: Tuple[float, float, float]  # width, height, depth in LEGO units
    mesh_path: str
    connection_points: List[Tuple[float, float, float]]  # Locations where this component can connect to others
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)  # Default to light gray
    
    # Additional properties
    is_structural: bool = False
    min_connections: int = 1
    max_connections: int = 10


@dataclass
class LegoSubModel:
    """Represents a LEGO sub-model assembled from components."""
    id: int
    part_id: int  # Reference to the originating image part
    components: List[Dict[str, Any]] = field(default_factory=list)
    transform: Optional[Transform3d] = None
    bounding_box: Optional[Tuple[float, float, float, float, float, float]] = None
    
    def add_component(self, component: LegoComponent, position: Tuple[float, float, float], 
                     rotation: Tuple[float, float, float]) -> None:
        """Add a component to the sub-model with position and rotation."""
        self.components.append({
            'component': component,
            'position': position,
            'rotation': rotation,
        })
        self._update_bounding_box()
    
    def _update_bounding_box(self) -> None:
        """Update the bounding box of the sub-model."""
        if not self.components:
            self.bounding_box = None
            return
            
        x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
        x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')
        
        for comp_data in self.components:
            component = comp_data['component']
            position = comp_data['position']
            w, h, d = component.dimensions
            
            # Simplified bounding box calculation (ignoring rotation for simplicity)
            x_min = min(x_min, position[0] - w/2)
            y_min = min(y_min, position[1] - h/2)
            z_min = min(z_min, position[2] - d/2)
            
            x_max = max(x_max, position[0] + w/2)
            y_max = max(y_max, position[1] + h/2)
            z_max = max(z_max, position[2] + d/2)
        
        self.bounding_box = (x_min, y_min, z_min, x_max, y_max, z_max)
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Return the center point of the sub-model."""
        if not self.bounding_box:
            return (0, 0, 0)
            
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounding_box
        return ((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2)
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Return the dimensions of the sub-model."""
        if not self.bounding_box:
            return (0, 0, 0)
            
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounding_box
        return (x_max - x_min, y_max - y_min, z_max - z_min)


class PartToLegoConverter:
    """Converts segmented parts into LEGO sub-models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the part-to-LEGO converter.
        
        Args:
            config: Configuration dictionary with the following keys:
                - lego_db_path: Path to LEGO components database
                - device: Device to run inference on
                - output_dir: Directory to save conversion results
                - voxel_resolution: Resolution for voxelization
                - min_components: Minimum number of LEGO components per sub-model
                - max_components: Maximum number of LEGO components per sub-model
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.get('output_dir', 'output/lego_models')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.voxel_resolution = config.get('voxel_resolution', 32)
        self.min_components = config.get('min_components', 5)
        self.max_components = config.get('max_components', 50)
        
        # Load LEGO component database
        self.lego_components = self._load_lego_database(config.get('lego_db_path', 'data/lego_components.json'))
        
        # Initialize 3D reconstruction model
        # For a real implementation, we would use a pre-trained 3D reconstruction model here
        self.reconstruction_model = self._initialize_reconstruction_model()
    
    def _load_lego_database(self, db_path: str) -> Dict[str, LegoComponent]:
        """
        Load the LEGO component database.
        
        Args:
            db_path: Path to the LEGO database JSON file
            
        Returns:
            Dictionary mapping component IDs to LegoComponent objects
        """
        # For the demo, we'll create a minimal set of components if the file doesn't exist
        if not os.path.exists(db_path):
            logger.warning(f"LEGO database not found at {db_path}. Creating a minimal set.")
            return self._create_demo_lego_database()
        
        try:
            with open(db_path, 'r') as f:
                lego_data = json.load(f)
            
            components = {}
            for comp_id, comp_data in lego_data.items():
                components[comp_id] = LegoComponent(
                    id=comp_id,
                    name=comp_data['name'],
                    category=comp_data['category'],
                    dimensions=tuple(comp_data['dimensions']),
                    mesh_path=comp_data['mesh_path'],
                    connection_points=comp_data['connection_points'],
                    color=tuple(comp_data.get('color', (0.7, 0.7, 0.7))),
                    is_structural=comp_data.get('is_structural', False),
                    min_connections=comp_data.get('min_connections', 1),
                    max_connections=comp_data.get('max_connections', 10),
                )
            
            logger.info(f"Loaded {len(components)} LEGO components from database")
            return components
        except Exception as e:
            logger.error(f"Error loading LEGO database: {e}")
            return self._create_demo_lego_database()
    
    def _create_demo_lego_database(self) -> Dict[str, LegoComponent]:
        """Create a minimal set of LEGO components for the demo."""
        components = {}
        
        # Add basic bricks
        components['3001'] = LegoComponent(
            id='3001',
            name='Brick 2 x 4',
            category='Brick',
            dimensions=(4, 1, 2),  # Width x Height x Depth in LEGO units
            mesh_path='meshes/3001.obj',
            connection_points=[(x+0.5, 1, z+0.5) for x in range(4) for z in range(2)],
            is_structural=True,
        )
        
        components['3003'] = LegoComponent(
            id='3003',
            name='Brick 2 x 2',
            category='Brick',
            dimensions=(2, 1, 2),
            mesh_path='meshes/3003.obj',
            connection_points=[(x+0.5, 1, z+0.5) for x in range(2) for z in range(2)],
            is_structural=True,
        )
        
        components['3004'] = LegoComponent(
            id='3004',
            name='Brick 1 x 2',
            category='Brick',
            dimensions=(2, 1, 1),
            mesh_path='meshes/3004.obj',
            connection_points=[(x+0.5, 1, 0.5) for x in range(2)],
            is_structural=True,
        )
        
        components['3005'] = LegoComponent(
            id='3005',
            name='Brick 1 x 1',
            category='Brick',
            dimensions=(1, 1, 1),
            mesh_path='meshes/3005.obj',
            connection_points=[(0.5, 1, 0.5)],
            is_structural=True,
        )
        
        # Add plates
        components['3020'] = LegoComponent(
            id='3020',
            name='Plate 2 x 4',
            category='Plate',
            dimensions=(4, 0.33, 2),
            mesh_path='meshes/3020.obj',
            connection_points=[(x+0.5, 0.33, z+0.5) for x in range(4) for z in range(2)],
        )
        
        components['3022'] = LegoComponent(
            id='3022',
            name='Plate 2 x 2',
            category='Plate',
            dimensions=(2, 0.33, 2),
            mesh_path='meshes/3022.obj',
            connection_points=[(x+0.5, 0.33, z+0.5) for x in range(2) for z in range(2)],
        )
        
        components['3023'] = LegoComponent(
            id='3023',
            name='Plate 1 x 2',
            category='Plate',
            dimensions=(2, 0.33, 1),
            mesh_path='meshes/3023.obj',
            connection_points=[(x+0.5, 0.33, 0.5) for x in range(2)],
        )
        
        # Add specialized parts
        components['3010'] = LegoComponent(
            id='3010',
            name='Brick 1 x 4',
            category='Brick',
            dimensions=(4, 1, 1),
            mesh_path='meshes/3010.obj',
            connection_points=[(x+0.5, 1, 0.5) for x in range(4)],
            is_structural=True,
        )
        
        components['3040'] = LegoComponent(
            id='3040',
            name='Slope 45° 2 x 1',
            category='Slope',
            dimensions=(1, 1, 2),
            mesh_path='meshes/3040.obj',
            connection_points=[(0.5, 0, z+0.5) for z in range(2)],
        )
        
        logger.info(f"Created demo LEGO database with {len(components)} components")
        return components
    
    def _initialize_reconstruction_model(self):
        """Initialize the 3D reconstruction model."""
        # This is a placeholder for a real 3D reconstruction model
        # In a real implementation, this would load a pre-trained model
        logger.info("Initializing 3D reconstruction model (placeholder)")
        
        # For demo purposes, just return a simple dummy model
        class DummyReconstructionModel:
            def __init__(self, device):
                self.device = device
            
            def reconstruct(self, masks, views):
                """Simple placeholder for 3D reconstruction."""
                # Return a simple cube voxel grid
                voxel_grid = np.zeros((32, 32, 32), dtype=np.float32)
                center = 32 // 2
                size = 10
                voxel_grid[
                    center-size//2:center+size//2,
                    center-size//2:center+size//2,
                    center-size//2:center+size//2
                ] = 1.0
                return voxel_grid
        
        return DummyReconstructionModel(self.device)
    
    def convert(self, parts: List[Part]) -> List[LegoSubModel]:
        """
        Convert segmented parts into LEGO sub-models.
        
        Args:
            parts: List of segmented parts
            
        Returns:
            List of LEGO sub-models
        """
        logger.info(f"Converting {len(parts)} parts to LEGO sub-models")
        
        sub_models = []
        
        for part in parts:
            logger.info(f"Converting part {part.id} with {part.num_views} views")
            
            # Skip parts that don't have enough views for reconstruction
            if part.num_views < 2:
                logger.warning(f"Skipping part {part.id} - insufficient views for reconstruction")
                continue
            
            # 1. Reconstruct 3D voxel representation from part masks
            voxel_grid = self._reconstruct_3d_voxels(part)
            
            # 2. Convert voxels to LEGO components
            sub_model = self._voxels_to_lego(part.id, voxel_grid)
            
            # 3. Save the sub-model
            self._save_sub_model(sub_model)
            
            sub_models.append(sub_model)
        
        logger.info(f"Converted {len(sub_models)} parts to LEGO sub-models")
        return sub_models
    
    def _reconstruct_3d_voxels(self, part: Part) -> np.ndarray:
        """
        Reconstruct 3D voxel representation from part masks.
        
        Args:
            part: Segmented part with masks from multiple views
            
        Returns:
            3D voxel grid representation of the part
        """
        # Gather masks from different views
        masks = {}
        for view_name, mask in part.masks.items():
            masks[view_name] = mask
        
        # Use the reconstruction model to generate 3D voxels
        voxel_grid = self.reconstruction_model.reconstruct(masks, list(masks.keys()))
        
        return voxel_grid
    
    def _voxels_to_lego(self, part_id: int, voxel_grid: np.ndarray) -> LegoSubModel:
        """
        Convert 3D voxel representation to LEGO components.
        
        Args:
            part_id: ID of the original part
            voxel_grid: 3D voxel grid representation
            
        Returns:
            LEGO sub-model composed of LEGO components
        """
        logger.info(f"Converting voxels to LEGO components for part {part_id}")
        
        # Initialize a new sub-model
        sub_model = LegoSubModel(id=part_id, part_id=part_id)
        
        # Calculate the scale factor based on voxel grid size
        scale_factor = 1.0  # This would be adjusted in a real implementation
        
        # Simple greedy algorithm to place LEGO bricks
        remaining_voxels = voxel_grid.copy()
        component_count = 0
        max_attempts = 1000  # Prevent infinite loops
        attempts = 0
        
        while np.sum(remaining_voxels) > 0 and component_count < self.max_components and attempts < max_attempts:
            attempts += 1
            
            # Find the largest connected component in the remaining voxels
            largest_component = self._find_largest_connected_component(remaining_voxels)
            
            if largest_component is None or np.sum(largest_component) == 0:
                break
            
            # Find the best fitting LEGO component for this voxel component
            best_component, position, rotation = self._find_best_component(largest_component, scale_factor)
            
            if best_component is None:
                # If we can't find a good component, remove a small part of the voxels and try again
                # (In a real implementation, we would be more sophisticated here)
                largest_component[0, 0, 0] = 0  # Just remove one voxel for the demo
                remaining_voxels = remaining_voxels * (1 - largest_component)
                continue
            
            # Add the component to the sub-model
            sub_model.add_component(best_component, position, rotation)
            component_count += 1
            
            # Remove the voxels covered by this component
            component_voxels = self._component_to_voxels(best_component, position, rotation, voxel_grid.shape)
            remaining_voxels = remaining_voxels * (1 - component_voxels)
        
        # If we didn't add enough components, add some basic structural components
        while component_count < self.min_components:
            # Add a basic brick at the center
            basic_brick = self.lego_components['3003']  # 2x2 brick
            center = np.array(voxel_grid.shape) / 2
            position = tuple(center * scale_factor)
            rotation = (0, 0, 0)
            
            sub_model.add_component(basic_brick, position, rotation)
            component_count += 1
        
        logger.info(f"Added {component_count} LEGO components to sub-model for part {part_id}")
        return sub_model
    
    def _find_largest_connected_component(self, voxel_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest connected component in a voxel grid.
        
        Args:
            voxel_grid: 3D voxel grid
            
        Returns:
            Binary mask of the largest connected component
        """
        # For simplicity in the demo, we'll just return the entire grid
        # In a real implementation, we would use a connected component algorithm
        if np.sum(voxel_grid) == 0:
            return None
            
        return (voxel_grid > 0).astype(np.float32)
    
    def _find_best_component(
        self, 
        voxel_component: np.ndarray, 
        scale_factor: float
    ) -> Tuple[Optional[LegoComponent], Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Find the best fitting LEGO component for a voxel component.
        
        Args:
            voxel_component: Binary voxel grid of a connected component
            scale_factor: Scale factor between voxel and LEGO units
            
        Returns:
            Tuple of (best component, position, rotation) or (None, None, None) if no good fit
        """
        # Get the bounding box of the voxel component
        indices = np.where(voxel_component > 0)
        if len(indices[0]) == 0:
            return None, (0, 0, 0), (0, 0, 0)
            
        min_x, max_x = np.min(indices[0]), np.max(indices[0])
        min_y, max_y = np.min(indices[1]), np.max(indices[1])
        min_z, max_z = np.min(indices[2]), np.max(indices[2])
        
        bbox_size = np.array([max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1])
        bbox_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
        
        # Calculate the volume of the voxel component
        voxel_volume = np.sum(voxel_component)
        
        # Find the best component based on simple heuristics
        best_component = None
        best_score = -float('inf')
        best_rotation = (0, 0, 0)
        
        # Try different rotations (0, 90, 180, 270 degrees around each axis)
        possible_rotations = [
            (0, 0, 0),
            (0, 0, 90),
            (0, 0, 180),
            (0, 0, 270),
            (0, 90, 0),
            (0, 180, 0),
            (0, 270, 0),
            (90, 0, 0),
            (180, 0, 0),
            (270, 0, 0),
        ]
        
        for component_id, component in self.lego_components.items():
            # Skip components that are too large
            if any(dim * scale_factor > size for dim, size in zip(component.dimensions, bbox_size)):
                continue
            
            # Try different rotations
            for rotation in possible_rotations:
                # Get the dimensions in this rotation
                rotated_dims = self._rotate_dimensions(component.dimensions, rotation)
                
                # Calculate the fill ratio (volume of the component / volume of the voxel component)
                component_volume = rotated_dims[0] * rotated_dims[1] * rotated_dims[2]
                fill_ratio = component_volume / voxel_volume
                
                # Calculate the coverage ratio (how much of the bounding box is covered)
                coverage_ratio = np.prod(rotated_dims) / np.prod(bbox_size)
                
                # Component score based on fill ratio and coverage
                # We want to maximize fill without overshooting too much
                score = fill_ratio * (1 - abs(coverage_ratio - 0.8))
                
                if score > best_score:
                    best_component = component
                    best_score = score
                    best_rotation = rotation
        
        if best_component is None:
            return None, (0, 0, 0), (0, 0, 0)
        
        # Calculate the position (center of the bounding box)
        position = tuple(bbox_center * scale_factor)
        
        return best_component, position, best_rotation
    
    def _rotate_dimensions(
        self, 
        dimensions: Tuple[float, float, float], 
        rotation: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Rotate the dimensions of a component.
        
        Args:
            dimensions: Original dimensions (width, height, depth)
            rotation: Rotation angles in degrees (x, y, z)
            
        Returns:
            Rotated dimensions
        """
        # Simplified rotation for demo purposes
        # In a real implementation, we would use proper 3D rotation matrices
        
        w, h, d = dimensions
        
        # Handle 90-degree rotations around the major axes
        if rotation[0] in [90, 270]:
            h, d = d, h
        
        if rotation[1] in [90, 270]:
            w, d = d, w
        
        if rotation[2] in [90, 270]:
            w, h = h, w
        
        return (w, h, d)
    
    def _component_to_voxels(
        self, 
        component: LegoComponent, 
        position: Tuple[float, float, float], 
        rotation: Tuple[float, float, float], 
        grid_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Convert a LEGO component to a voxel representation.
        
        Args:
            component: LEGO component
            position: Position in the grid
            rotation: Rotation angles
            grid_shape: Shape of the voxel grid
            
        Returns:
            Binary voxel grid representation of the component
        """
        # Create an empty voxel grid
        voxel_grid = np.zeros(grid_shape, dtype=np.float32)
        
        # Get the rotated dimensions
        w, h, d = self._rotate_dimensions(component.dimensions, rotation)
        
        # Calculate the bounding box
        min_x = max(0, int(position[0] - w/2))
        max_x = min(grid_shape[0], int(position[0] + w/2))
        min_y = max(0, int(position[1] - h/2))
        max_y = min(grid_shape[1], int(position[1] + h/2))
        min_z = max(0, int(position[2] - d/2))
        max_z = min(grid_shape[2], int(position[2] + d/2))
        
        # Fill the voxel grid
        voxel_grid[min_x:max_x, min_y:max_y, min_z:max_z] = 1.0
        
        return voxel_grid
    
    def _save_sub_model(self, sub_model: LegoSubModel) -> None:
        """
        Save a LEGO sub-model to disk.
        
        Args:
            sub_model: LEGO sub-model to save
        """
        # Create the output directory if it doesn't exist
        sub_model_dir = os.path.join(self.output_dir, f"part_{sub_model.part_id}")
        os.makedirs(sub_model_dir, exist_ok=True)
        
        # Save the sub-model as JSON for easy loading
        sub_model_data = {
            'id': sub_model.id,
            'part_id': sub_model.part_id,
            'components': [
                {
                    'component_id': comp_data['component'].id,
                    'position': comp_data['position'],
                    'rotation': comp_data['rotation'],
                }
                for comp_data in sub_model.components
            ],
            'bounding_box': sub_model.bounding_box,
        }
        
        with open(os.path.join(sub_model_dir, 'sub_model.json'), 'w') as f:
            json.dump(sub_model_data, f, indent=2)
        
        logger.info(f"Saved sub-model {sub_model.id} to {sub_model_dir}")
    
    def fix_issues(self, sub_models: List[LegoSubModel], issues: List[Dict[str, Any]]) -> List[LegoSubModel]:
        """
        Fix issues in sub-models based on verification feedback.
        
        Args:
            sub_models: List of LEGO sub-models
            issues: List of issues to fix
            
        Returns:
            Updated list of LEGO sub-models
        """
        logger.info(f"Fixing {len(issues)} issues in {len(sub_models)} sub-models")
        
        # Create a copy of the sub-models to modify
        updated_sub_models = []
        
        for sub_model in sub_models:
            # Check if this sub-model has issues
            sub_model_issues = [issue for issue in issues if issue['sub_model_id'] == sub_model.id]
            
            if not sub_model_issues:
                # No issues, keep as is
                updated_sub_models.append(sub_model)
                continue
            
            # Create a modified copy of the sub-model
            fixed_sub_model = LegoSubModel(
                id=sub_model.id,
                part_id=sub_model.part_id,
                components=sub_model.components.copy(),
                transform=sub_model.transform,
                bounding_box=sub_model.bounding_box,
            )
            
            # Apply fixes based on issue type
            for issue in sub_model_issues:
                issue_type = issue['type']
                
                if issue_type == 'disconnected_component':
                    # Add a connecting brick
                    self._fix_disconnected_component(fixed_sub_model, issue)
                
                elif issue_type == 'unstable_structure':
                    # Add support or reinforce the structure
                    self._fix_unstable_structure(fixed_sub_model, issue)
                
                elif issue_type == 'invalid_connection':
                    # Fix invalid connections between bricks
                    self._fix_invalid_connection(fixed_sub_model, issue)
                
                elif issue_type == 'overlap':
                    # Fix overlapping bricks
                    self._fix_overlap(fixed_sub_model, issue)
            
            updated_sub_models.append(fixed_sub_model)
        
        logger.info(f"Fixed issues in {len(updated_sub_models)} sub-models")
        return updated_sub_models
    
    def _fix_disconnected_component(self, sub_model: LegoSubModel, issue: Dict[str, Any]) -> None:
        """Fix a disconnected component by adding connecting bricks."""
        # Get the affected components
        component_indices = issue.get('component_indices', [])
        if len(component_indices) < 2:
            return
            
        # Get the positions of the disconnected components
        positions = [sub_model.components[idx]['position'] for idx in component_indices]
        
        # Calculate the midpoint between the components
        midpoint = tuple(np.mean([np.array(pos) for pos in positions], axis=0))
        
        # Add a connecting brick
        connecting_brick = self.lego_components['3003']  # 2x2 brick as a simple connector
        sub_model.add_component(connecting_brick, midpoint, (0, 0, 0))
        
        logger.info(f"Added connecting brick at {midpoint} to fix disconnected component")
    
    def _fix_unstable_structure(self, sub_model: LegoSubModel, issue: Dict[str, Any]) -> None:
        """Fix an unstable structure by adding support."""
        # Get the affected region
        region = issue.get('region', None)
        if region is None:
            return
            
        # Add a supporting plate underneath
        x_min, y_min, z_min, x_max, y_max, z_max = region
        center = ((x_min + x_max) / 2, y_min - 0.5, (z_min + z_max) / 2)
        
        # Choose an appropriate plate based on size
        width = x_max - x_min
        depth = z_max - z_min
        
        if width <= 2 and depth <= 2:
            plate = self.lego_components['3022']  # 2x2 plate
        elif width <= 2 and depth <= 4:
            plate = self.lego_components['3020']  # 2x4 plate
        else:
            plate = self.lego_components['3020']  # 2x4 plate (multiple may be needed)
        
        sub_model.add_component(plate, center, (0, 0, 0))
        
        logger.info(f"Added supporting plate at {center} to fix unstable structure")
    
    def _fix_invalid_connection(self, sub_model: LegoSubModel, issue: Dict[str, Any]) -> None:
        """Fix invalid connections between bricks."""
        # Get the affected components
        component_indices = issue.get('component_indices', [])
        if len(component_indices) != 2:
            return
            
        # Remove one of the problematic components
        idx_to_remove = component_indices[0]
        removed_component = sub_model.components.pop(idx_to_remove)
        
        logger.info(f"Removed component to fix invalid connection")
    
    def _fix_overlap(self, sub_model: LegoSubModel, issue: Dict[str, Any]) -> None:
        """Fix overlapping bricks."""
        # Get the affected components
        component_indices = issue.get('component_indices', [])
        if len(component_indices) < 2:
            return
            
        # Sort indices in descending order to avoid index shifting
        component_indices.sort(reverse=True)
        
        # Remove all but the first component
        for idx in component_indices[1:]:
            if idx < len(sub_model.components):
                removed_component = sub_model.components.pop(idx)
                
        logger.info(f"Removed overlapping components")