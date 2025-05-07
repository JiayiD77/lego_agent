#!/usr/bin/env python3
"""
Image to Parts Segmentation Module
Processes three-view images and segments them into individual components/parts.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import cv2
from dataclasses import dataclass

# Import Segment Anything Model
from segment_anything import sam_model_registry, SamPredictor

logger = logging.getLogger(__name__)

@dataclass
class Part:
    """Represents a segmented part with its properties."""
    id: int
    masks: Dict[str, np.ndarray]  # View name -> binary mask
    bounding_boxes: Dict[str, Tuple[int, int, int, int]]  # View name -> (x1, y1, x2, y2)
    features: Dict[str, Any]  # Additional features like color, texture, etc.
    correspondence: Dict[str, List[int]]  # Maps to part IDs in other views
    
    @property
    def num_views(self) -> int:
        """Return the number of views this part appears in."""
        return len(self.masks)


class ImageToPartsSegmenter:
    """Segments three-view images into individual parts using the Segment Anything Model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image-to-parts segmenter.
        
        Args:
            config: Configuration dictionary with the following keys:
                - sam_checkpoint: Path to SAM checkpoint
                - sam_model_type: SAM model type (default: vit_h)
                - device: Device to run inference on
                - output_dir: Directory to save segmentation results
                - view_names: List of view names
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.get('output_dir', 'output/segmentation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.view_names = config.get('view_names', ['front', 'side', 'top'])
        
        # Initialize Segment Anything Model
        sam_checkpoint = config.get('sam_checkpoint', 'sam_vit_h_4b8939.pth')
        sam_model_type = config.get('sam_model_type', 'vit_h')
        
        logger.info(f"Loading SAM model {sam_model_type} from {sam_checkpoint}")
        
        try:
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
        except Exception as e:
            logger.warning(f"Failed to load SAM model: {e}")
            logger.warning("Falling back to simplified segmentation approach")
            self.predictor = None
        
        # For part correspondence matching across views
        self.feature_extractor = self._initialize_feature_extractor()
    
    def _initialize_feature_extractor(self):
        """Initialize a feature extractor for part matching across views."""
        # In a real implementation, this could use a pre-trained feature extractor
        # For this demo, we'll use a simplified approach with SIFT features
        return cv2.SIFT_create()
    
    def segment(self, images: List[Image.Image]) -> List[Part]:
        """
        Segment the input images into parts.
        
        Args:
            images: List of PIL images in the order [front, side, top]
            
        Returns:
            List of segmented Part objects
        """
        logger.info(f"Segmenting {len(images)} images")
        
        if len(images) != len(self.view_names):
            logger.warning(f"Expected {len(self.view_names)} views, got {len(images)}. Using available views.")
        
        # Convert PIL images to numpy arrays for processing
        np_images = [np.array(img) for img in images]
        
        # Dictionary to store segmentation results for each view
        view_segments = {}
        
        # Process each view image
        for i, (view_name, image) in enumerate(zip(self.view_names, np_images)):
            if i >= len(images):
                break
                
            logger.info(f"Processing {view_name} view")
            view_segments[view_name] = self._segment_single_image(image, view_name)
        
        # Establish correspondences between parts in different views
        parts = self._establish_part_correspondence(view_segments)
        
        logger.info(f"Segmented {len(parts)} parts across all views")
        return parts
    
    def _segment_single_image(self, image: np.ndarray, view_name: str) -> Dict[int, Dict[str, Any]]:
        """
        Segment a single view image into parts.
        
        Args:
            image: Numpy array image
            view_name: Name of the view (front, side, top)
            
        Returns:
            Dictionary mapping part ID to part data for this view
        """
        # Convert to RGB if needed
        if image.shape[2] == 4:  # with alpha channel
            image = image[:, :, :3]
        
        # Use SAM if available, otherwise fall back to simpler methods
        if self.predictor is not None:
            return self._segment_with_sam(image, view_name)
        else:
            return self._segment_with_contours(image, view_name)
    
    def _segment_with_sam(self, image: np.ndarray, view_name: str) -> Dict[int, Dict[str, Any]]:
        """Use Segment Anything Model for image segmentation."""
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Set image in SAM predictor
        self.predictor.set_image(image)
        
        # Generate automatic masks
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True
        )
        
        # Filter masks by score and size
        min_area = 100  # Minimum mask area in pixels
        min_score = 0.8  # Minimum confidence score
        
        filtered_parts = {}
        part_id = 0
        
        for mask, score in zip(masks, scores):
            if score < min_score:
                continue
                
            # Calculate mask area
            area = np.sum(mask)
            if area < min_area:
                continue
            
            # Find bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            
            # Extract features for later matching
            masked_image = image.copy()
            masked_image[~mask] = 0
            features = self._extract_features(masked_image, mask)
            
            filtered_parts[part_id] = {
                'mask': mask,
                'bbox': (x1, y1, x2, y2),
                'score': score,
                'features': features,
            }
            
            part_id += 1
        
        logger.info(f"Segmented {len(filtered_parts)} parts in {view_name} view")
        return filtered_parts
    
    def _segment_with_contours(self, image: np.ndarray, view_name: str) -> Dict[int, Dict[str, Any]]:
        """Fallback method using contour detection for segmentation."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, threshold = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        parts = {}
        for i, contour in enumerate(filtered_contours):
            # Create mask from contour
            mask = np.zeros_like(gray, dtype=bool)
            cv2.drawContours(mask, [contour], 0, True, -1)
            
            # Find bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract features for matching
            masked_image = image.copy()
            if len(masked_image.shape) == 3:
                mask_3d = np.stack([mask] * 3, axis=2)
                masked_image[~mask_3d] = 0
            else:
                masked_image[~mask] = 0
                
            features = self._extract_features(masked_image, mask)
            
            parts[i] = {
                'mask': mask,
                'bbox': (x, y, x + w, y + h),
                'features': features,
            }
        
        logger.info(f"Segmented {len(parts)} parts in {view_name} view using contour method")
        return parts
    
    def _extract_features(self, masked_image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from a masked part for matching across views.
        
        Args:
            masked_image: Image with only the part visible
            mask: Boolean mask of the part
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic shape features
        features['area'] = np.sum(mask)
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            features['aspect_ratio'] = (np.max(x_indices) - np.min(x_indices)) / max(1, (np.max(y_indices) - np.min(y_indices)))
            
            # Centroid
            features['centroid'] = (
                np.mean(x_indices),
                np.mean(y_indices)
            )
        
        # Color features if color image
        if len(masked_image.shape) == 3:
            # Average color of the part
            r, g, b = cv2.split(masked_image)
            features['avg_color'] = (
                np.sum(r) / max(1, features['area']),
                np.sum(g) / max(1, features['area']),
                np.sum(b) / max(1, features['area'])
            )
        
        # Extract SIFT keypoints and descriptors if possible
        try:
            gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY) if len(masked_image.shape) == 3 else masked_image
            keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, mask.astype(np.uint8))
            if descriptors is not None:
                features['descriptors'] = descriptors
                features['num_keypoints'] = len(keypoints)
        except Exception as e:
            logger.warning(f"Failed to extract SIFT features: {e}")
        
        return features
    
    def _establish_part_correspondence(self, view_segments: Dict[str, Dict[int, Dict[str, Any]]]) -> List[Part]:
        """
        Establish correspondences between parts across different views.
        
        Args:
            view_segments: Dictionary mapping view names to segmentation results
            
        Returns:
            List of Part objects with established correspondences
        """
        logger.info("Establishing part correspondences across views")
        
        global_part_id = 0
        parts = []
        
        # Start with front view as reference
        if 'front' in view_segments:
            front_parts = view_segments['front']
            
            for front_id, front_data in front_parts.items():
                # Initialize a new part
                part = Part(
                    id=global_part_id,
                    masks={'front': front_data['mask']},
                    bounding_boxes={'front': front_data['bbox']},
                    features=front_data['features'],
                    correspondence={'front': [front_id]}
                )
                
                # Find correspondences in other views
                for view_name, view_data in view_segments.items():
                    if view_name == 'front':
                        continue
                    
                    best_match_id = self._find_best_match(front_data, view_data)
                    if best_match_id is not None:
                        part.masks[view_name] = view_data[best_match_id]['mask']
                        part.bounding_boxes[view_name] = view_data[best_match_id]['bbox']
                        part.correspondence[view_name] = [best_match_id]
                
                parts.append(part)
                global_part_id += 1
        
        # Handle parts in other views that don't correspond to front view parts
        for view_name, view_data in view_segments.items():
            if view_name == 'front':
                continue
                
            for part_id, part_data in view_data.items():
                # Check if this part is already assigned to an existing part
                is_assigned = any(
                    view_name in part.correspondence and part_id in part.correspondence[view_name]
                    for part in parts
                )
                
                if not is_assigned:
                    # Create a new part
                    part = Part(
                        id=global_part_id,
                        masks={view_name: part_data['mask']},
                        bounding_boxes={view_name: part_data['bbox']},
                        features=part_data['features'],
                        correspondence={view_name: [part_id]}
                    )
                    
                    parts.append(part)
                    global_part_id += 1
        
        logger.info(f"Established {len(parts)} unique parts with correspondences")
        return parts
    
    def _find_best_match(
        self, 
        source_part: Dict[str, Any], 
        target_parts: Dict[int, Dict[str, Any]]
    ) -> Optional[int]:
        """
        Find the best matching part in the target view.
        
        Args:
            source_part: Source part data
            target_parts: Dictionary of parts in the target view
            
        Returns:
            ID of the best matching part or None if no good match found
        """
        best_match_id = None
        best_match_score = 0
        
        for target_id, target_part in target_parts.items():
            # Calculate similarity score between source and target
            score = self._calculate_similarity(source_part, target_part)
            
            if score > best_match_score and score > 0.5:  # Threshold for matching
                best_match_score = score
                best_match_id = target_id
        
        return best_match_id
    
    def _calculate_similarity(self, part1: Dict[str, Any], part2: Dict[str, Any]) -> float:
        """
        Calculate similarity score between two parts.
        
        Args:
            part1: First part data
            part2: Second part data
            
        Returns:
            Similarity score between 0 and 1
        """
        # Initialize with shape similarity
        area_ratio = min(
            part1['features']['area'] / max(1, part2['features']['area']),
            part2['features']['area'] / max(1, part1['features']['area'])
        )
        
        # Add aspect ratio similarity if available
        aspect_similarity = 0
        if 'aspect_ratio' in part1['features'] and 'aspect_ratio' in part2['features']:
            ar1 = part1['features']['aspect_ratio']
            ar2 = part2['features']['aspect_ratio']
            aspect_similarity = 1 - min(1, abs(ar1 - ar2) / max(1, max(ar1, ar2)))
        
        # Add color similarity if available
        color_similarity = 0
        if 'avg_color' in part1['features'] and 'avg_color' in part2['features']:
            # Simple Euclidean distance in RGB space
            c1 = np.array(part1['features']['avg_color'])
            c2 = np.array(part2['features']['avg_color'])
            color_dist = np.linalg.norm(c1 - c2)
            color_similarity = max(0, 1 - color_dist / 255)
        
        # Add SIFT descriptor matching if available
        descriptor_similarity = 0
        if ('descriptors' in part1['features'] and 'descriptors' in part2['features'] and 
                part1['features']['descriptors'] is not None and part2['features']['descriptors'] is not None):
            
            desc1 = part1['features']['descriptors']
            desc2 = part2['features']['descriptors']
            
            try:
                # FLANN-based matcher
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc1, desc2, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                match_ratio = len(good_matches) / max(1, min(len(desc1), len(desc2)))
                descriptor_similarity = match_ratio
            except Exception as e:
                logger.warning(f"Error in descriptor matching: {e}")
        
        # Weighted combination of similarities
        similarity = (
            0.3 * area_ratio + 
            0.2 * aspect_similarity + 
            0.2 * color_similarity + 
            0.3 * descriptor_similarity
        )
        
        return similarity
    
    def save_visualization(self, parts: List[Part], output_path: str) -> None:
        """
        Save a visualization of the segmented parts.
        
        Args:
            parts: List of segmented parts
            output_path: Path to save the visualization
        """
        # Create a figure with subplots for each view
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb
        
        n_views = len(self.view_names)
        fig, axs = plt.subplots(1, n_views, figsize=(5 * n_views, 5))
        
        # If only one view, wrap the axis in a list
        if n_views == 1:
            axs = [axs]
        
        # Create a visualization for each view
        for view_idx, view_name in enumerate(self.view_names):
            ax = axs[view_idx]
            
            # Create a blank image
            viz_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            
            # Add each part with a unique color
            for i, part in enumerate(parts):
                if view_name in part.masks:
                    # Generate a color based on part ID
                    hue = (i * 0.618033988749895) % 1.0
                    color_hsv = np.array([hue, 0.8, 0.8])
                    color_rgb = hsv_to_rgb(color_hsv)
                    
                    # Resize mask to fit visualization
                    mask = part.masks[view_name]
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), 
                        (512, 512), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    # Apply the colored mask
                    viz_image[mask_resized] = (color_rgb * 255).astype(np.uint8)
            
            # Display in the subplot
            ax.imshow(viz_image)
            ax.set_title(f"{view_name.capitalize()} View")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        logger.info(f"Saved part visualization to {output_path}")