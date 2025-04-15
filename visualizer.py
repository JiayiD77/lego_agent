import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import Rectangle
import json
import os
from PIL import Image
from io import BytesIO

from pdb import set_trace

class LegoBrickVisualizer:
    def __init__(self, api_key=None):
        """
        Initialize the LEGO brick visualizer.
        
        Args:
            api_key (str): Rebrickable API key. If None, will look for REBRICKABLE_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.environ.get("REBRICKABLE_API_KEY")
            if api_key is None:
                raise ValueError("API key must be provided or set as REBRICKABLE_API_KEY environment variable")
        
        self.api_key = api_key
        self.base_url = "https://rebrickable.com/api/v3/lego"
        self.headers = {"Authorization": f"key {self.api_key}"}
        
        # Standard brick dimensions (in LEGO units)
        self.stud_diameter = 0.8
        self.stud_height = 0.17
        self.brick_height = 1.0
        self.brick_width_per_stud = 1.0
        self.brick_length_per_stud = 1.0
        
    def get_brick_data(self, part_number):
        """
        Fetch brick data from Rebrickable API.
        
        Args:
            part_number (str): LEGO part number (e.g., "3001" for a 2x4 brick)
            
        Returns:
            dict: Brick data if successful, None otherwise
        """
        url = f"{self.base_url}/parts/{part_number}/"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            brick_data = response.json()
            
            # Also get part images
            images_url = f"{self.base_url}/parts/{part_number}/images/"
            images_response = requests.get(images_url, headers=self.headers, timeout=1)
            
            if images_response.status_code == 200:
                brick_data['images'] = images_response.json()['results']
            else:
                brick_data['images'] = []
                
            return brick_data
        else:
            print(f"Error fetching brick data: {response.status_code}")
            print(response.text)
            return None
    
    def display_brick_image(self, brick_data):
        """
        Display the brick image if available.
        
        Args:
            brick_data (dict): Brick data from the API
            
        Returns:
            bool: True if image was displayed, False otherwise
        """
        # if not brick_data.get('images'):
        #     return False
        
        # for img_data in brick_data['images']:
        #     if img_data.get('url'):
        #         try:
        #             img_response = requests.get(img_data['url'])
        #             if img_response.status_code == 200:
        #                 img = Image.open(BytesIO(img_response.content))
        #                 plt.figure(figsize=(8, 6))
        #                 plt.imshow(img)
        #                 plt.title(f"LEGO Brick: {brick_data['name']} (Part #{brick_data['part_num']})")
        #                 plt.axis('off')
        #                 plt.show()
        #                 return True
        #         except Exception as e:
        #             print(f"Error displaying image: {e}")
        
        img_response = requests.get(brick_data.get('part_img_url'))
        if img_response.status_code == 200:
            img = Image.open(BytesIO(img_response.content))
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"LEGO Brick: {brick_data['name']} (Part #{brick_data['part_num']})")
            plt.axis('off')
            plt.show()
            return True
        print("No part_img_url available")
        return False
            
    def visualize_brick_3d(self, brick_data, dimensions=None):
        """
        Create a 3D visualization of the brick.
        
        Args:
            brick_data (dict): Brick data from the API
            dimensions (tuple): Optional (length, width, height) in studs. 
                               If None, tries to extract from the name or uses a default.
        """
        # Try to determine dimensions from name if not provided
        if dimensions is None:
            dimensions = self._extract_dimensions_from_name(brick_data['name'])
        
        length, width, height = dimensions
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Main brick body
        self._draw_brick_body(ax, length, width, height)
        
        # Add studs on top
        self._draw_studs(ax, length, width, height)
        
        # Set equal aspect ratio
        ax.set_box_aspect([length, width, height])
        
        # Labels
        ax.set_title(f"LEGO Brick: {brick_data['name']} (Part #{brick_data['part_num']})")
        ax.set_xlabel('Length (studs)')
        ax.set_ylabel('Width (studs)')
        ax.set_zlabel('Height (studs)')
        
        # Set view angle
        ax.view_init(30, 30)
        
        plt.tight_layout()
        plt.show()
        
        # Print brick information
        print(f"Brick: {brick_data['name']} (Part #{brick_data['part_num']})")
        print(f"Dimensions: {length}x{width}x{height} studs")
        if brick_data.get('year_from'):
            print(f"First appeared: {brick_data['year_from']}")
        print(f"Part categories: {brick_data.get('part_categories', [])}")
    
    def _extract_dimensions_from_name(self, name):
        """
        Try to extract brick dimensions from name.
        Returns default 2x4x1 if unsuccessful.
        """
        # Common pattern for brick names is "Brick LxW"
        import re
        
        # Try to find "2x4" or similar pattern
        pattern = r'(\d+)\s*[xX]\s*(\d+)'
        match = re.search(pattern, name)
        
        if match:
            width = int(match.group(1))
            length = int(match.group(2))
            return (length, width, 1)  # Standard brick height is 1
        
        # Default to 2x4 brick if we can't determine
        return (4, 2, 1)
    
    def _draw_brick_body(self, ax, length, width, height):
        """Draw the main body of the brick."""
        # Vertices of the brick (bottom)
        bottom_vertices = np.array([
            [0, 0, 0],
            [length, 0, 0],
            [length, width, 0],
            [0, width, 0]
        ])
        
        # Vertices of the brick (top)
        top_vertices = bottom_vertices.copy()
        top_vertices[:, 2] = height * self.brick_height
        
        # Draw the faces
        faces = [
            # Bottom
            [bottom_vertices[0], bottom_vertices[1], bottom_vertices[2], bottom_vertices[3]],
            # Top
            [top_vertices[0], top_vertices[1], top_vertices[2], top_vertices[3]],
            # Sides
            [bottom_vertices[0], bottom_vertices[1], top_vertices[1], top_vertices[0]],
            [bottom_vertices[1], bottom_vertices[2], top_vertices[2], top_vertices[1]],
            [bottom_vertices[2], bottom_vertices[3], top_vertices[3], top_vertices[2]],
            [bottom_vertices[3], bottom_vertices[0], top_vertices[0], top_vertices[3]]
        ]
        
        # Colors for different faces
        colors = ['red', 'red', 'red', 'red', 'red', 'red']
        alphas = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        
        # Plot each face
        for i, face in enumerate(faces):
            face = np.array(face)
            X, Y, Z = face[:, 0], face[:, 1], face[:, 2]
            
            # Add a fifth point to close the polygon
            X = np.append(X, X[0])
            Y = np.append(Y, Y[0])
            Z = np.append(Z, Z[0])
            
            ax.plot_surface(
                np.array([X[0], X[1], X[2], X[3]]).reshape(2, 2),
                np.array([Y[0], Y[1], Y[2], Y[3]]).reshape(2, 2),
                np.array([Z[0], Z[1], Z[2], Z[3]]).reshape(2, 2),
                color=colors[i], alpha=alphas[i]
            )
    
    def _draw_studs(self, ax, length, width, height):
        """Draw the studs on top of the brick."""
        z_base = height * self.brick_height
        
        for i in range(length):
            for j in range(width):
                # The center of each stud
                x = i + 0.5
                y = j + 0.5
                
                # Draw a cylinder for each stud
                self._draw_cylinder(
                    ax, 
                    center=(x, y, z_base),
                    radius=self.stud_diameter/2,
                    height=self.stud_height
                )
    
    def _draw_cylinder(self, ax, center, radius, height, color='red', resolution=20):
        """Draw a cylinder at the specified position."""
        x_center, y_center, z_base = center
        
        # Generate points on the circumference
        theta = np.linspace(0, 2*np.pi, resolution)
        x = x_center + radius * np.cos(theta)
        y = y_center + radius * np.sin(theta)
        
        # Draw the bottom circle
        ax.plot(x, y, np.full_like(theta, z_base), color=color)
        
        # Draw the top circle
        z_top = z_base + height
        ax.plot(x, y, np.full_like(theta, z_top), color=color)
        
        # Draw the walls of the cylinder
        for i in range(resolution):
            ax.plot(
                [x[i], x[i]],
                [y[i], y[i]],
                [z_base, z_top],
                color=color
            )
            
        # Fill the top with a surface
        u = np.linspace(0, 2*np.pi, resolution)
        v = np.linspace(0, radius, 5)
        V, U = np.meshgrid(v, u)
        
        X = x_center + V * np.cos(U)
        Y = y_center + V * np.sin(U)
        Z = np.full_like(X, z_top)
        
        ax.plot_surface(X, Y, Z, color=color, alpha=0.9)


def main():
    """Main function to visualize a LEGO brick."""
    print("LEGO Brick Visualizer")
    print("---------------------")
    
    # Get API key
    api_key = os.environ.get("REBRICKABLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Rebrickable API key: ")
    
    # Create visualizer
    visualizer = LegoBrickVisualizer(api_key)
    
    # Get part number
    part_number = input("Enter LEGO part number (default: 3001 for 2x4 brick): ") or "3001"
    
    # Fetch brick data
    print(f"Fetching data for part #{part_number}...")
    brick_data = visualizer.get_brick_data(part_number)
    print(brick_data)
    
    if brick_data:
        print(f"Found brick: {brick_data['name']}")
        
        # Try to display the brick image
        if not visualizer.display_brick_image(brick_data):
            print("No image available, generating 3D model...")
        
        # # Ask for custom dimensions
        # custom_dims = input("Enter custom dimensions as 'length width height' in studs (or press Enter for auto-detection): ")
        # if custom_dims:
        #     try:
        #         l, w, h = map(int, custom_dims.split())
        #         dimensions = (l, w, h)
        #     except:
        #         print("Invalid dimensions, using auto-detection.")
        #         dimensions = None
        # else:
        #     dimensions = None
        
        # # Visualize the brick
        # visualizer.visualize_brick_3d(brick_data, dimensions)
    else:
        print(f"Failed to retrieve data for part #{part_number}")
        print("Try common LEGO brick numbers like: 3001, 3002, 3003, 3004, 3005, 3008, 3010")


if __name__ == "__main__":
    main()