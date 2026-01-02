import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# --- PATHS ---
# Make sure these match your preprocess.py paths
MODEL_MASK_DIR = "../data/processed/train_masks"
VIZ_MASK_DIR = "../data/processed/train_masks_viz"

def validate_and_visualize(num_samples=3):
    # Get all generated mask files
    mask_files = glob.glob(os.path.join(MODEL_MASK_DIR, "*.png"))
    
    if not mask_files:
        print(f"❌ No files found in {MODEL_MASK_DIR}. Did preprocess.py run?")
        return

    print(f"✅ Found {len(mask_files)} masks. Inspecting {num_samples} random samples...\n")
    
    # Pick random samples
    samples = random.sample(mask_files, min(num_samples, len(mask_files)))

    for i, model_mask_path in enumerate(samples):
        filename = os.path.basename(model_mask_path)
        viz_mask_path = os.path.join(VIZ_MASK_DIR, filename)

        # 1. Load the Model Mask (The one for training)
        # MUST load as grayscale (flag 0) or OpenCV loads it as 3-channel BGR
        mask_int = cv2.imread(model_mask_path, 0)
        
        # 2. Load the Visualization Mask (The one for eyes)
        # Load as color (flag 1) and convert BGR -> RGB for Matplotlib
        mask_viz = cv2.imread(viz_mask_path, 1)
        mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_BGR2RGB)

        # --- VALIDATION LOGIC ---
        unique_values = np.unique(mask_int)
        print(f"🔎 Sample {i+1}: {filename}")
        print(f"   Shape: {mask_int.shape}")
        print(f"   Unique Class IDs found: {unique_values}")
        
        # Check if we have suspicious values (anything > 4)
        if np.max(unique_values) > 4:
            print("   ⚠️  WARNING: Found values > 4. Check your mapping!")
        else:
            print("   ✅ Data looks valid for training.")

        # --- VISUALIZATION ---
        plt.figure(figsize=(10, 5))
        
        # Plot Model Mask (we multiply by 50 just to make the dark pixels visible to human eye)
        plt.subplot(1, 2, 1)
        plt.imshow(mask_int * 50, cmap='gray') 
        plt.title(f"Model Input (Scaled x50)\nClasses: {unique_values}")
        plt.axis('off')

        # Plot Viz Mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask_viz)
        plt.title("Human Visualization\n(RGB)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        print("-" * 50)

if __name__ == "__main__":
    validate_and_visualize()