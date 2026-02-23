import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import argparse
import sys
import random
import glob

# --- 1. SMART PATH DETECTION ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Damage Level Mapping
DAMAGE_MAP_RGB = {
    0: (30, 30, 30),     # Background
    1: (46, 204, 113),   # No Damage (Class 1)
    2: (241, 196, 15),   # Minor Damage (Class 2)
    3: (230, 126, 34),   # Major Damage (Class 3)
    4: (231, 76, 60)     # Destroyed (Class 4)
}

DAMAGE_DESCRIPTIONS = {
    0: "Background/Vegetation",
    1: "No Damage (Intact)",
    2: "Minor Damage (Functional)",
    3: "Major Damage (Structural Risk)",
    4: "Destroyed (Total Loss)"
}

def load_inference_data(pre_path, post_path, mask_dir, img_size=(512, 512)):
    """Loads images and the ground truth mask if available."""
    img_pre = cv2.imread(pre_path)
    img_post = cv2.imread(post_path)
    
    # Try to find the Ground Truth mask
    mask_name = os.path.basename(post_path)
    mask_path = os.path.join(mask_dir, mask_name)
    gt_mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None

    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
    img_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2RGB)
    
    x1 = cv2.resize(img_pre, img_size).astype(np.float32) / 255.0
    x2 = cv2.resize(img_post, img_size).astype(np.float32) / 255.0
    
    if gt_mask is not None:
        gt_mask = cv2.resize(gt_mask, img_size, interpolation=cv2.INTER_NEAREST)

    return (np.expand_dims(x1, 0), np.expand_dims(x2, 0)), (img_pre, img_post), gt_mask

def calculate_stats(pred_mask, gt_mask=None):
    """Calculates distribution weights and Accuracy score."""
    building_pixels = pred_mask[pred_mask > 0]
    total_buildings = len(building_pixels)
    if total_buildings == 0: return None, 0.0, 0.0

    # Damage Breakdown
    counts = {i: np.sum(building_pixels == i) for i in range(1, 5)}
    stats = {i: (counts[i]/total_buildings)*100 for i in range(1, 5)}
    
    # Severity Score
    severity = ((counts[2] * 2.5) + (counts[3] * 6.5) + (counts[4] * 10.0)) / (total_buildings * 10.0) * 10.0
    
    # Accuracy Score (Pixel comparison if GT exists)
    accuracy = 0.0
    if gt_mask is not None:
        correct_pixels = np.sum(pred_mask == gt_mask)
        accuracy = (correct_pixels / (pred_mask.shape[0] * pred_mask.shape[1])) * 100
        
    return stats, severity, accuracy

def mask_to_rgb(mask):
    """Converts int mask to displayable RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in DAMAGE_MAP_RGB.items():
        rgb[mask == cid] = color
    return rgb

def create_report(pre_img, post_img, prediction, gt_mask, save_path):
    """Generates the side-by-side comparison report."""
    
    # --- Architecture Agnostic Handling ---
    if isinstance(prediction, list):
        mask_pred = prediction[0]
        scene_pred = prediction[1] if len(prediction) > 1 else None
    else:
        mask_pred = prediction
        scene_pred = None
    
    pred_mask = np.argmax(mask_pred, axis=-1)[0]
    stats, severity, accuracy = calculate_stats(pred_mask, gt_mask)
    
    # Global Scene Assessment & Confidence
    if scene_pred is not None:
        scene_class = np.argmax(scene_pred[0])
        scene_confidence = np.max(scene_pred[0]) * 100
        scene_label = DAMAGE_DESCRIPTIONS[scene_class]
    else:
        # Heuristic fallback: Use the worst damage found in the mask
        unique_in_mask = np.unique(pred_mask)
        unique_in_mask = unique_in_mask[unique_in_mask > 0]
        worst_class = int(np.max(unique_in_mask)) if len(unique_in_mask) > 0 else 1
        scene_label = DAMAGE_DESCRIPTIONS[worst_class] + " (Heuristic)"
        scene_confidence = 100.0

    print("\n" + "="*60)
    print(f" ASSESSMENT REPORT | SEVERITY: {severity:.2f}/10.0 ")
    print(f" GLOBAL SCENE RATING: {scene_label} ({scene_confidence:.1f}%)")
    if gt_mask is not None:
        print(f" PIXEL-LEVEL ACCURACY vs GROUND TRUTH: {accuracy:.2f}%")
    print("="*60)
    if stats:
        for i in range(1, 5):
            print(f" [Class {i}] {DAMAGE_DESCRIPTIONS[i]:30} | {stats.get(i, 0.0):6.2f}%")
    print("="*60 + "\n")

    # Visualization
    fig = plt.figure(figsize=(24, 12))
    plt.suptitle(f"Damage Assessment Comparison | Scene Rating: {scene_label}", fontsize=22, fontweight='bold')

    # Top Row: Images
    ax1 = fig.add_subplot(2, 3, 1); ax1.imshow(pre_img); ax1.set_title("1. Pre-Disaster Image"); ax1.axis('off')
    ax2 = fig.add_subplot(2, 3, 2); ax2.imshow(post_img); ax2.set_title("2. Post-Disaster Image"); ax2.axis('off')
    
    # Bottom Row: Masks
    ax3 = fig.add_subplot(2, 3, 4)
    if gt_mask is not None:
        ax3.imshow(mask_to_rgb(gt_mask))
        ax3.set_title("3. Actual Mask (Ground Truth)")
    else:
        ax3.text(0.5, 0.5, "GT Mask Not Found", ha='center')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 3, 5); ax4.imshow(mask_to_rgb(pred_mask)); ax4.set_title("4. AI Generated Mask"); ax4.axis('off')

    # Stats Panel
    ax5 = fig.add_subplot(1, 3, 3)
    if stats:
        labels = [f"Class {i}: {DAMAGE_DESCRIPTIONS[i]}" for i in range(1, 5)]
        values = [stats.get(i, 0.0) for i in range(1, 5)]
        ax5.barh(labels, values, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
        ax5.set_title(f"Damage Distribution (%) | Confidence: {scene_confidence:.1f}%")
        ax5.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.show()

def get_random_sample():
    """Finds a random imagery pair and its mask automatically."""
    home = os.path.expanduser("~")
    base_dirs = [
        os.path.join(PROJECT_ROOT, "data", "tier1"),
        os.path.join(home, ".cache", "kagglehub", "datasets", "qianlanzz", "xbd-dataset", "versions", "1", "xbd", "tier1"),
    ]
    for d in base_dirs:
        img_p = os.path.join(d, "images")
        if os.path.exists(img_p):
            pre_files = glob.glob(os.path.join(img_p, "*_pre_disaster.png"))
            if pre_files:
                pre = random.choice(pre_files)
                post = pre.replace("_pre_disaster.png", "_post_disaster.png")
                # Also find where masks were processed
                mask_p = os.path.join(PROJECT_ROOT, "data", "processed", "train_masks")
                return pre, post, mask_p
    return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", default=None)
    parser.add_argument("--post", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # Find Model
    model_path = args.model
    if not model_path:
        for c in [os.path.join(PROJECT_ROOT, "results", "models", "xbd_tier1_best.keras"), "results/models/xbd_tier1_best.keras"]:
            if os.path.exists(c): model_path = c; break

    if not model_path: print("Error: Model not found."); return

    # Find Data
    pre, post, mask_dir = args.pre, args.post, args.mask_dir
    if not pre:
        print("Selecting random sample for validation...")
        pre, post, mask_dir = get_random_sample()

    if not pre: print("Error: Data not found."); return

    print(f"Loading Model: {os.path.basename(model_path)}")
    model = keras.models.load_model(model_path, compile=False)
    
    try:
        inputs, display, gt_mask = load_inference_data(pre, post, mask_dir)
        print(f"Running Analysis on {os.path.basename(pre)}...")
        pred = model.predict(inputs, verbose=0)
        create_report(display[0], display[1], pred, gt_mask, "assessment_report.png")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
