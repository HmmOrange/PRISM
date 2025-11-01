import os
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

from PIL import Image

from prism.tools.tool_registry import register_tool

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Image processing tools will be limited.")

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    warnings.warn("PIL not available. Some image processing tools will not work.")

IMAGE_TAGS = ["image processing", "computer vision"]

class ImageProcessor:
    """Base class for image processing operations."""
    
    def __init__(self):
        if not (HAS_CV2 or HAS_PIL):
            raise ImportError("Either OpenCV or PIL must be installed for image processing")
    
    def process_image(self, image_path: str, **kwargs):
        """Process a single image. Override in subclasses."""
        raise NotImplementedError


@register_tool(tags=[*IMAGE_TAGS, "image splitting"])
class ImageSplitter(ImageProcessor):
    """
    Split images into smaller patches/tiles.
    """
    
    def split_image(self, image_path: str, patch_size: Tuple[int, int] = (256, 256), overlap: int = 0) -> List[Image.Image]:
        """
        Split an image into patches.
        
        Args:
            image_path (str): Path to input image
            patch_size (Tuple[int, int]): Size of each patch (height, width)
            overlap (int): Overlap between patches in pixels
            
        Returns:
            List[Image.Image]: List of image patches as PIL Images
        """
        if not HAS_CV2 or not HAS_PIL:
            raise ImportError("Both OpenCV and PIL are required for image splitting")
            
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        h, w = image.shape[:2]
        patch_h, patch_w = patch_size
        
        patches = []
        
        # Calculate step size
        step_h = patch_h - overlap
        step_w = patch_w - overlap
        
        for y in range(0, h - patch_h + 1, step_h):
            for x in range(0, w - patch_w + 1, step_w):
                # Extract patch
                patch = image[y:y+patch_h, x:x+patch_w]
                
                # Convert BGR to RGB for PIL
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(patch_rgb)
                patches.append(pil_image)
        
        return patches


@register_tool(tags=[*IMAGE_TAGS, "image cropping"])
class ImageCropper(ImageProcessor):
    """
    Crop images to specified dimensions.
    """
    
    def crop_image(self, image_path: str, crop_box: Optional[Tuple[int, int, int, int]] = None, 
                   center_crop: bool = True, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Crop an image.
        
        Args:
            image_path (str): Path to input image
            crop_box (Optional[Tuple[int, int, int, int]]): (x, y, width, height) for cropping
            center_crop (bool): Whether to crop from center
            target_size (Optional[Tuple[int, int]]): Target size for center crop (width, height)
            
        Returns:
            Image.Image: Cropped image as PIL Image
        """
        if HAS_PIL:
            with Image.open(image_path) as img:
                if crop_box:
                    # Crop using specified box
                    x, y, w, h = crop_box
                    cropped = img.crop((x, y, x + w, y + h))
                elif center_crop and target_size:
                    # Center crop to target size
                    cropped = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
                else:
                    cropped = img.copy()
                
                return cropped
                
        elif HAS_CV2:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            if crop_box:
                x, y, w, h = crop_box
                cropped = image[y:y+h, x:x+w]
            elif center_crop and target_size:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # Calculate center crop coordinates
                start_x = max(0, (w - target_w) // 2)
                start_y = max(0, (h - target_h) // 2)
                end_x = min(w, start_x + target_w)
                end_y = min(h, start_y + target_h)
                
                cropped = image[start_y:end_y, start_x:end_x]
            else:
                cropped = image
            
            # Convert BGR to RGB for PIL
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        
        raise RuntimeError("Neither PIL nor OpenCV is available")


@register_tool(tags=IMAGE_TAGS)
class ImageResizer(ImageProcessor):
    """
    Resize images to specified dimensions.
    """
    
    def resize_image(self, image_path: str, target_size: Tuple[int, int], keep_aspect_ratio: bool = True) -> Image.Image:
        """
        Resize an image.
        
        Args:
            image_path (str): Path to input image
            target_size (Tuple[int, int]): Target size (width, height)
            keep_aspect_ratio (bool): Whether to maintain aspect ratio
            
        Returns:
            Image.Image: Resized image as PIL Image
        """
        if HAS_PIL:
            with Image.open(image_path) as img:
                if keep_aspect_ratio:
                    img_copy = img.copy()
                    img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
                    resized = img_copy
                else:
                    resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                return resized
                
        elif HAS_CV2:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            if keep_aspect_ratio:
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # Calculate scaling factor
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert BGR to RGB for PIL
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return Image.fromarray(resized_rgb)
        
        raise RuntimeError("Neither PIL nor OpenCV is available")