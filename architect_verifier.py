#!/usr/bin/env python3
"""
Architect Verifier Module
Verifies that LEGO models are structurally sound, buildable, and follow LEGO design principles.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json
from dataclasses import dataclass

# Import local modules
from part_to_lego import LegoSubModel, LegoComponent
from submodel_to_model import LegoModel

logger = logging.getLogger(__name__)

@dataclass
class VerificationIssue:
    """Represents an issue found during verification."""
    type: str  # Type of issue (e.g., disconnected_component, unstable_structure)
    severity: str  # Severity level (e.g., warning, error)
    message: str  # Description of the issue
    sub_model_id: Optional[int] = None  # ID of the sub-model with the issue (if applicable)
    component_indices: Optional[List[int]] = None  # Indices of the components with the issue
    region: Optional[Tuple[float, float, float, float, float, float]] = None  # Bounding box of the issue region
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue to a dictionary for serialization."""
        result = {
            'type': self.type,
            'severity': self.severity,
            'message': self.message,
        }
        
        if self.sub_model_id is not None:
            result['sub_model_id'] = self.sub_model_id
            
        if self.component_indices is not None:
            result['component_indices'] = self.component_indices
            
        if self.region is not None:
            result['region'] = self.region
            
        return result


class ArchitectVerifier:
    """Verifies LEGO models for structural integrity and buildability."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the architect verifier.
        
        Args:
            config: Configuration dictionary with the following keys:
                - output_dir: Directory to save verification results
                - strict_mode: If True, apply stricter verification rules
                - connection_tolerance: Maximum distance for valid connections
                - stability_factor: Minimum stability factor required
        """
        self.config = config
        self.output_dir = config.get('output_dir', 'output/verification')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.strict_mode = config.get('strict_mode', False)
        self.connection_tolerance = config.get('connection_tolerance', 0.1)  # LEGO units
        self.stability_factor = config.get('stability_factor', 0.3)  # Minimum ratio of supported area
    
    def verify(self, model: LegoModel, sub_models: List[LegoSubModel]) -> Dict[str, Any]:
        """
        Verify a LEGO model for structural integrity and buildability.
        
        Args:
            model: The complete LEGO model
            sub_models: List of all sub-models (for reference)
            
        Returns:
            Dictionary with verification results and any issues found
        """
        logger.info(f"Verifying LEGO model {model.id}")
        
        issues = []
        
        # 1. Verify each sub-model individually
        for sm_data in model.sub_models:
            sm = sm_data['sub_model']
            position = sm_data['position']
            rotation = sm_data['rotation']
            
            # Verify connectivity within the sub-model
            connectivity_issues = self._verify_connectivity(sm)
            issues.extend(connectivity_issues)
            
            # Verify stability of the sub-model
            stability_issues = self._verify_stability(sm)
            issues.extend(stability_issues)
        
        # 2. Verify sub-model relationships and global model properties
        # Check for overlapping sub-models
        overlap_issues = self._verify_no_overlaps(model)
        issues.extend(overlap_issues)
        
        # Check for global stability
        global_stability_issues = self._verify_global_stability(model)
        issues.extend(global_stability_issues)
        
        # Check buildability (can it be assembled in a reasonable sequence)
        buildability_issues = self._verify_buildability(model)
        issues.extend(buildability_issues)
        
        # Categorize issues by severity
        errors = [issue for issue in issues if issue.severity == 'error']
        warnings = [issue for issue in issues if issue.severity == 'warning']
        
        # Determine if the model is valid
        is_valid = len(errors) == 0
        
        # Create the verification result
        result = {
            'is_valid': is_valid,
            'issues': [issue.to_dict() for issue in issues],
            'error_count': len(errors),
            'warning_count': len(warnings),
            'timestamp': self._get_timestamp(),
        }
        
        # Save the verification result
        self._save_verification_result(model.id, result)
        
        logger.info(f"Verification complete. Model is {'valid' if is_valid else 'invalid'} "
                   f"with {len(errors)} errors and {len(warnings)} warnings.")
        return result
    
    def _verify_connectivity(self, sub_model: LegoSubModel) -> List[VerificationIssue]:
        """
        Verify that all components in a sub-model are connected.
        
        Args:
            sub_model: The sub-model to verify
            
        Returns:
            List of connectivity issues found
        """
        issues = []
        
        if len(sub_model.components) <= 1:
            return issues  # No connectivity issues possible with 0 or 1 components
        
        # Build a connectivity graph
        connection_graph = {i: [] for i in range(len(sub_model.components))}
        
        # Check connectivity between each pair of components
        for i, comp_data1 in enumerate(sub_model.components):
            comp1 = comp_data1['component']
            pos1 = comp_data1['position']
            rot1 = comp_data1['rotation']
            
            for j, comp_data2 in enumerate(sub_model.components[i+1:], i+1):
                comp2 = comp_data2['component']
                pos2 = comp_data2['position']
                rot2 = comp_data2['rotation']
                
                # Check if these components are connected
                if self._are_components_connected(comp1, pos1, rot1, comp2, pos2, rot2):
                    connection_graph[i].append(j)
                    connection_graph[j].append(i)
        
        # Check if the graph is connected using BFS
        visited = set()
        queue = [0]  # Start from the first component
        
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend([n for n in connection_graph[node] if n not in visited])
        
        # If not all components are visited, there are disconnected components
        if len(visited) < len(sub_model.components):
            disconnected = [i for i in range(len(sub_model.components)) if i not in visited]
            
            for disc_idx in disconnected:
                # Create a disconnected component issue
                issues.append(VerificationIssue(
                    type='disconnected_component',
                    severity='error' if self.strict_mode else 'warning',
                    message=f"Component {disc_idx} is disconnected from the main structure",
                    sub_model_id=sub_model.id,
                    component_indices=[disc_idx],
                ))
        
        return issues
    
    def _are_components_connected(
        self, 
        comp1: LegoComponent, 
        pos1: Tuple[float, float, float], 
        rot1: Tuple[float, float, float], 
        comp2: LegoComponent, 
        pos2: Tuple[float, float, float], 
        rot2: Tuple[float, float, float]
    ) -> bool:
        """
        Check if two LEGO components are connected.
        
        Args:
            comp1: First component
            pos1: Position of the first component
            rot1: Rotation of the first component
            comp2: Second component
            pos2: Position of the second component
            rot2: Rotation of the second component
            
        Returns:
            True if the components are connected, False otherwise
        """
        # In a real implementation, this would check for valid LEGO connections
        # considering studs, anti-studs, connection points, and physical constraints
        
        # For the demo, we'll use a simple distance-based heuristic
        # Calculate the distance between component centers
        dist = np.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )
        
        # Calculate the maximum distance for a valid connection based on component sizes
        max_dist = np.sqrt(
            (comp1.dimensions[0] / 2 + comp2.dimensions[0] / 2)**2 +
            (comp1.dimensions[1] / 2 + comp2.dimensions[1] / 2)**2 +
            (comp1.dimensions[2] / 2 + comp2.dimensions[2] / 2)**2
        )
        
        # Check if the components are close enough to be connected
        return dist <= max_dist + self.connection_tolerance
    
    def _verify_stability(self, sub_model: LegoSubModel) -> List[VerificationIssue]:
        """
        Verify that a sub-model is stable and won't collapse.
        
        Args:
            sub_model: The sub-model to verify
            
        Returns:
            List of stability issues found
        """
        issues = []
        
        if len(sub_model.components) <= 1:
            return issues  # No stability issues possible with 0 or 1 components
        
        # Calculate the center of mass
        total_mass = 0
        weighted_pos = np.array([0.0, 0.0, 0.0])
        
        for comp_data in sub_model.components:
            comp = comp_data['component']
            pos = comp_data['position']
            
            # Calculate component mass (proportional to volume)
            mass = comp.dimensions[0] * comp.dimensions[1] * comp.dimensions[2]
            total_mass += mass
            
            # Add weighted position
            weighted_pos += np.array(pos) * mass
        
        if total_mass > 0:
            center_of_mass = weighted_pos / total_mass
        else:
            center_of_mass = np.array([0.0, 0.0, 0.0])
        
        # Check if the center of mass is supported
        is_supported = False
        
        # Get the bounding box of the sub-model
        if sub_model.bounding_box:
            x_min, y_min, z_min, x_max, y_max, z_max = sub_model.bounding_box
            
            # Check if there are components below the center of mass
            supporting_components = []
            
            for i, comp_data in enumerate(sub_model.components):
                comp = comp_data['component']
                pos = comp_data['position']
                
                # Component bounding box
                comp_x_min = pos[0] - comp.dimensions[0] / 2
                comp_x_max = pos[0] + comp.dimensions[0] / 2
                comp_y_min = pos[1] - comp.dimensions[1] / 2
                comp_y_max = pos[1] + comp.dimensions[1] / 2
                comp_z_min = pos[2] - comp.dimensions[2] / 2
                comp_z_max = pos[2] + comp.dimensions[2] / 2
                
                # Check if this component is below the center of mass
                if (comp_y_max <= center_of_mass[1] and
                    comp_x_min <= center_of_mass[0] <= comp_x_max and
                    comp_z_min <= center_of_mass[2] <= comp_z_max):
                    supporting_components.append(i)
            
            # If there are supporting components, the structure is stable
            is_supported = len(supporting_components) > 0
            
            # For more complex stability analysis, consider:
            # 1. The supported area ratio
            # 2. Cantilever effects
            # 3. Weak connection points
            
            # Simple supported area check
            footprint_area = (x_max - x_min) * (z_max - z_min)
            supported_area = 0
            
            for comp_data in sub_model.components:
                comp = comp_data['component']
                pos = comp_data['position']
                
                # Only consider components at the bottom
                if abs(pos[1] - y_min) < self.connection_tolerance:
                    supported_area += comp.dimensions[0] * comp.dimensions[2]
            
            # Calculate the stability factor
            stability_ratio = supported_area / max(1e-6, footprint_area)
            
            if stability_ratio < self.stability_factor:
                # Create an unstable structure issue
                region = sub_model.bounding_box
                issues.append(VerificationIssue(
                    type='unstable_structure',
                    severity='warning',
                    message=f"Structure may be unstable (stability factor: {stability_ratio:.2f})",
                    sub_model_id=sub_model.id,
                    region=region,
                ))
        
        if not is_supported:
            # Create an unsupported structure issue
            issues.append(VerificationIssue(
                type='unsupported_structure',
                severity='error',
                message="Structure has no supporting components under its center of mass",
                sub_model_id=sub_model.id,
            ))
        
        return issues
    
    def _verify_no_overlaps(self, model: LegoModel) -> List[VerificationIssue]:
        """
        Verify that sub-models don't overlap.
        
        Args:
            model: The complete LEGO model
            
        Returns:
            List of overlap issues found
        """
        issues = []
        
        for i, sm_data1 in enumerate(model.sub_models):
            sm1 = sm_data1['sub_model']
            pos1 = sm_data1['position']
            
            for j, sm_data2 in enumerate(model.sub_models[i+1:], i+1):
                sm2 = sm_data2['sub_model']
                pos2 = sm_data2['position']
                
                # Check for overlap
                if self._check_sub_model_overlap(sm1, pos1, sm2, pos2):
                    # Create an overlap issue
                    issues.append(VerificationIssue(
                        type='sub_model_overlap',
                        severity='error',
                        message=f"Sub-models {sm1.id} and {sm2.id} overlap",
                        sub_model_id=sm1.id,
                        component_indices=[j],  # Reference to the overlapping sub-model
                    ))
        
        return issues
    
    def _check_sub_model_overlap(
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
    
    def _verify_global_stability(self, model: LegoModel) -> List[VerificationIssue]:
        """
        Verify global stability of the complete model.
        
        Args:
            model: The complete LEGO model
            
        Returns:
            List of global stability issues found
        """
        issues = []
        
        # Calculate the center of mass for the entire model
        total_mass = 0
        weighted_pos = np.array([0.0, 0.0, 0.0])
        
        for sm_data in model.sub_models:
            sm = sm_data['sub_model']
            pos = sm_data['position']
            
            # Calculate sub-model mass (estimate based on bounding box)
            if sm.bounding_box:
                x_min, y_min, z_min, x_max, y_max, z_max = sm.bounding_box
                volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
                mass = volume
                
                total_mass += mass
                weighted_pos += np.array(pos) * mass
        
        if total_mass > 0:
            center_of_mass = weighted_pos / total_mass
        else:
            center_of_mass = np.array([0.0, 0.0, 0.0])
        
        # Check if the model has a stable base
        if model.bounding_box:
            x_min, y_min, z_min, x_max, y_max, z_max = model.bounding_box
            
            # Calculate the model footprint
            footprint_area = (x_max - x_min) * (z_max - z_min)
            
            # Check if the center of mass is within the footprint
            com_in_footprint = (
                x_min <= center_of_mass[0] <= x_max and
                z_min <= center_of_mass[2] <= z_max
            )
            
            if not com_in_footprint:
                # Create a global stability issue
                issues.append(VerificationIssue(
                    type='global_instability',
                    severity='error',
                    message="Center of mass is outside the model's footprint",
                ))
            
            # Check if the model has supporting elements at the bottom
            # Count sub-models that touch the ground
            ground_supported = False
            
            for sm_data in model.sub_models:
                sm = sm_data['sub_model']
                pos = sm_data['position']
                
                if sm.bounding_box:
                    sm_y_min = sm.bounding_box[1] + pos[1]
                    
                    if abs(sm_y_min - y_min) < self.connection_tolerance:
                        ground_supported = True
                        break
            
            if not ground_supported:
                # Create a floating model issue
                issues.append(VerificationIssue(
                    type='floating_model',
                    severity='error',
                    message="Model has no supporting elements touching the ground",
                ))
        
        return issues
    
    def _verify_buildability(self, model: LegoModel) -> List[VerificationIssue]:
        """
        Verify that the model can be built in a reasonable sequence.
        
        Args:
            model: The complete LEGO model
            
        Returns:
            List of buildability issues found
        """
        issues = []
        
        # For a real implementation, this would check for:
        # 1. Impossible build sequences
        # 2. Internal structures that can't be accessed
        # 3. Weak or unstable intermediate stages
        
        # For the demo, we'll use a simple heuristic: check for fully enclosed spaces
        enclosed_spaces = self._find_enclosed_spaces(model)
        
        for space in enclosed_spaces:
            # Create an enclosed space issue
            issues.append(VerificationIssue(
                type='enclosed_space',
                severity='warning',
                message="Model contains enclosed spaces that may be difficult to build",
                region=space,
            ))
        
        return issues
    
    def _find_enclosed_spaces(self, model: LegoModel) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Find enclosed spaces within the model that may be difficult to build.
        
        Args:
            model: The complete LEGO model
            
        Returns:
            List of bounding boxes for enclosed spaces
        """
        # For the demo, this is a placeholder
        # In a real implementation, this would use a more sophisticated algorithm
        
        # Placeholder: just return an empty list for now
        return []
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp as a string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_verification_result(self, model_id: str, result: Dict[str, Any]) -> None:
        """
        Save verification results to disk.
        
        Args:
            model_id: ID of the verified model
            result: Verification result dictionary
        """
        # Create a directory for this model if it doesn't exist
        model_dir = os.path.join(self.output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = result['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"verification_{timestamp}.json"
        
        # Save the result as JSON
        with open(os.path.join(model_dir, filename), 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved verification result to {os.path.join(model_dir, filename)}")