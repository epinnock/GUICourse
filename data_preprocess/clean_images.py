import json
import re
from PIL import Image
import numpy as np
import concurrent.futures
import os
import random

import numpy as np
from PIL import Image
from scipy.stats import mode

def is_solid_color_or_blank(image, tolerance=10, required_percent=0.99):
    """
    Check if an image is solid color or blank, with tolerance for small variations.
    
    Args:
    image (PIL.Image): The image to check.
    tolerance (int): The maximum difference in pixel values to consider as the same color.
    required_percent (float): The required percentage of pixels to be within the tolerance.
    
    Returns:
    bool: True if the image is solid color or blank, False otherwise.
    """
    img_array = np.array(image)

    # Check if the image is completely transparent (for RGBA images)
    if image.mode == 'RGBA' and np.all(img_array[:, :, 3] == 0):
        return True

    # For grayscale images, we don't need to process channels separately
    if len(img_array.shape) == 2:
        img_array = img_array[:,:,np.newaxis]

    # Flatten the array to 2D: (pixels, channels)
    flat_array = img_array.reshape(-1, img_array.shape[-1])

    # Find the most common color
    most_common_color = mode(flat_array, axis=0).mode.ravel()

    # Calculate the Manhattan distance of each pixel from the most common color
    distances = np.abs(flat_array - most_common_color).sum(axis=1)

    # Check if the required percentage of pixels are within the tolerance
    pixels_within_tolerance = np.sum(distances <= tolerance)
    percent_within_tolerance = pixels_within_tolerance / flat_array.shape[0]

    return percent_within_tolerance >= required_percent

    # Note: This method is sensitive to small variations in pixel values.
    # Images with slight compression artifacts or anti-aliasing might not be
    # detected as solid color, even if they appear so to the human eye.

def extract_image_path(value):
    """
    Extract image path from a string.
    
    Args:
    value (str): The string containing the image path.
    
    Returns:
    str or None: The extracted image path, or None if not found.
    """
    match = re.search(r'<img>(.*?)</img>', value)
    return match.group(1) if match else None

def process_image(image_path):
    """
    Process an image file.
    
    Args:
    image_path (str): The path to the image file.
    
    Returns:
    dict: A dictionary containing the image path and processing result.
    """
    try:
        full_path = os.path.join(os.path.dirname(__file__), '..', image_path)
        with Image.open(full_path) as image:
            return {
                'path': image_path,
                'is_solid_or_blank': is_solid_color_or_blank(image)
            }
    except Exception as e:
        return {
            'path': image_path,
            'error': str(e)
        }

def check_images_in_json(json_file_path, max_workers=10, cutoff=0):
    """
    Check images referenced in a JSON file.
    
    Args:
    json_file_path (str): The path to the JSON file.
    max_workers (int): The maximum number of worker threads to use.
    cutoff (int): The number of entries to process. If 0, process all entries.
    
    Returns:
    list: A list of dictionaries containing the processing results for each image.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    if cutoff > 0 and cutoff < len(data):
        data = random.sample(data, cutoff)
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {}
        for item in data:
            for conversation in item.get('conversations', []):
                if conversation['from'] == 'user':
                    image_path = extract_image_path(conversation['value'])
                    if image_path:
                        future = executor.submit(process_image, image_path)
                        future_to_item[future] = item['id']
        
        for future in concurrent.futures.as_completed(future_to_item):
            item_id = future_to_item[future]
            result = future.result()
            results.append({
                'id': item_id,
                'image': result
            })
    
    return results

# Example usage
if __name__ == "__main__":
    json_file_path = "./data/training_data_qwen.json"
    cutoff = 100  # Set to 0 to process all entries, or a positive integer to randomly sample
    results = check_images_in_json(json_file_path, cutoff=cutoff)
    
    failed_count = 0
    for item in results:
        print(f"ID: {item['id']}")
        image = item['image']
        if 'error' in image:
            print(f"  {image['path']}: Error - {image['error']}")
            failed_count += 1
        else:
            status = "Solid color or blank" if image['is_solid_or_blank'] else "Normal image"
            print(f"  {image['path']}: {status}")
        print()
    
    print(f"Total images processed: {len(results)}")
    print(f"Failed images: {failed_count}")
    print(f"Success rate: {(len(results) - failed_count) / len(results) * 100:.2f}%")