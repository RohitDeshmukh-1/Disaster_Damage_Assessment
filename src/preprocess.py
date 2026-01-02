import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
from shapely import wkt
import sys

# --- CONFIGURATION ---
RAW_DATA_DIR = "../data/raw/xbd/tier1" 
OUTPUT_DIR_MODEL = "../data/processed/train_masks"       # For Training (Integers)
OUTPUT_DIR_VIZ = "../data/processed/train_masks_viz"     # For Eyes (RGB Colors)

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

# 1. MAPPING FOR MODEL (Integer IDs)
DAMAGE_MAP_INT = {
    "background": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 
}

# 2. MAPPING FOR HUMAN EYES (RGB)
# Format: (Blue, Green, Red) because OpenCV uses BGR, not RGB!
DAMAGE_MAP_RGB = {
    0: (0, 0, 0),       # Background (Black)
    1: (0, 255, 0),     # No Damage (Green)
    2: (0, 255, 255),   # Minor Damage (Yellow)
    3: (0, 165, 255),   # Major Damage (Orange) - OpenCV uses BGR!
    4: (0, 0, 255)      # Destroyed (Red)
}

def parse_json(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def get_geometry_and_labels(pre_path, post_path):
    pre_data = parse_json(pre_path)
    post_data = parse_json(post_path)
    
    if not pre_data or not post_data:
        return None

    # Map: UID -> Polygon (from PRE)
    geometry_map = {}
    features_pre = pre_data.get('features', {}).get('xy', [])
    for feat in features_pre:
        uid = feat['properties']['uid']
        try:
            poly = wkt.loads(feat['wkt'])
            coords = list(poly.exterior.coords)
            geometry_map[uid] = np.array(coords, np.int32)
        except:
            continue

    # Map: UID -> Damage Class (from POST)
    damage_map = {}
    features_post = post_data.get('features', {}).get('xy', [])
    for feat in features_post:
        uid = feat['properties']['uid']
        subtype = feat['properties'].get('subtype', 'no-damage')
        damage_map[uid] = DAMAGE_MAP_INT.get(subtype, 1)

    return geometry_map, damage_map

def generate_masks(pre_path, post_path, filename):
    data = get_geometry_and_labels(pre_path, post_path)
    if not data:
        return

    geometry_map, damage_label_map = data
    
    # --- A. Create Integer Mask (For Model) ---
    mask_int = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    
    # --- B. Create RGB Mask (For Visualization) ---
    mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    for uid, polygon in geometry_map.items():
        damage_class = damage_label_map.get(uid, 1)
        
        # Draw on Integer Mask
        cv2.fillPoly(mask_int, [polygon], color=damage_class)
        
        # Draw on RGB Mask (Look up color from dictionary)
        color_bgr = DAMAGE_MAP_RGB.get(damage_class, (255, 255, 255))
        cv2.fillPoly(mask_rgb, [polygon], color=color_bgr)

    # Save Both
    cv2.imwrite(os.path.join(OUTPUT_DIR_MODEL, filename), mask_int)
    cv2.imwrite(os.path.join(OUTPUT_DIR_VIZ, filename), mask_rgb)

def main():
    os.makedirs(OUTPUT_DIR_MODEL, exist_ok=True)
    os.makedirs(OUTPUT_DIR_VIZ, exist_ok=True)
    
    search_path = os.path.join(RAW_DATA_DIR, "labels", "*_pre_disaster.json")
    pre_files = glob.glob(search_path)
    
    print(f"Found {len(pre_files)} scenes. generating dual masks...")
    
    for pre_file in tqdm(pre_files):
        post_file = pre_file.replace("_pre_disaster.json", "_post_disaster.json")
        if os.path.exists(post_file):
            filename = os.path.basename(post_file).replace(".json", ".png")
            generate_masks(pre_file, post_file, filename)

if __name__ == "__main__":
    main()