import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
from shapely import wkt
import argparse

# --- CONFIGURATION (Defaults) ---
# Default to current directory data structure if none provided
DEFAULT_RAW_DATA_DIR = "./data/raw/xbd/tier1"  
DEFAULT_OUTPUT_DIR_MODEL = "./data/processed/train_masks"       
DEFAULT_OUTPUT_DIR_VIZ = "./data/processed/train_masks_viz"     

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

DAMAGE_MAP_INT = {
    "background": 0,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1 
}

# OpenCV uses BGR
DAMAGE_MAP_BGR = {
    0: (0, 0, 0),       # Background (Black)
    1: (0, 255, 0),     # No Damage (Green)
    2: (0, 255, 255),   # Minor Damage (Yellow)
    3: (0, 165, 255),   # Major Damage (Orange)
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

    damage_map = {}
    features_post = post_data.get('features', {}).get('xy', [])
    for feat in features_post:
        uid = feat['properties']['uid']
        subtype = feat['properties'].get('subtype', 'no-damage')
        damage_map[uid] = DAMAGE_MAP_INT.get(subtype, 1)

    return geometry_map, damage_map

def generate_masks(pre_path, post_path, filename, out_model, out_viz):
    data = get_geometry_and_labels(pre_path, post_path)
    if not data:
        return

    geometry_map, damage_label_map = data
    mask_int = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    mask_rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    for uid, polygon in geometry_map.items():
        damage_class = damage_label_map.get(uid, 1)
        cv2.fillPoly(mask_int, [polygon], color=damage_class)
        color_bgr = DAMAGE_MAP_BGR.get(damage_class, (255, 255, 255))
        cv2.fillPoly(mask_rgb, [polygon], color=color_bgr)

    cv2.imwrite(os.path.join(out_model, filename), mask_int)
    cv2.imwrite(os.path.join(out_viz, filename), mask_rgb)

def main():
    parser = argparse.ArgumentParser(description="xBD Mask Generation")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_RAW_DATA_DIR, help="Path to tier1 directory")
    parser.add_argument("--out_model", type=str, default=DEFAULT_OUTPUT_DIR_MODEL, help="Output for integer masks")
    parser.add_argument("--out_viz", type=str, default=DEFAULT_OUTPUT_DIR_VIZ, help="Output for viz masks")
    args = parser.parse_args()

    os.makedirs(args.out_model, exist_ok=True)
    os.makedirs(args.out_viz, exist_ok=True) 
    
    labels_dir = os.path.join(args.data_dir, "labels")
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return

    search_path = os.path.join(labels_dir, "*_pre_disaster.json")
    pre_files = glob.glob(search_path)
    
    print(f"Found {len(pre_files)} scenes. Generating masks...")
    
    for pre_file in tqdm(pre_files):
        post_file = pre_file.replace("_pre_disaster.json", "_post_disaster.json")
        if os.path.exists(post_file):
            filename = os.path.basename(post_file).replace(".json", ".png")
            generate_masks(pre_file, post_file, filename, args.out_model, args.out_viz)

if __name__ == "__main__":
    main()
