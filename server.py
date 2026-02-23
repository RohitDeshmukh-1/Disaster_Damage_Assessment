import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import io

# --- CONFIG ---
MODEL_PATH = "results/models/xbd_tier1_best.keras"
IMG_SIZE = (512, 512)

# Damage Level Mapping (Matching inference.py)
DAMAGE_MAP_RGB = {
    0: (30, 30, 30),     # Background
    1: (46, 204, 113),   # No Damage (Green)
    2: (241, 196, 15),   # Minor Damage (Yellow)
    3: (230, 126, 34),   # Major Damage (Orange)
    4: (231, 76, 60)     # Destroyed (Red)
}

DAMAGE_DESCRIPTIONS = {
    0: "Background/Vegetation",
    1: "No Damage (Intact)",
    2: "Minor Damage (Functional)",
    3: "Major Damage (Structural Risk)",
    4: "Destroyed (Total Loss)"
}

app = FastAPI(title="Building Damage Assessment API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model once at startup
print(f"Loading Model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print("WARNING: Model file not found. API will fail on inference.")
    model = None
else:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

def process_image(file_content):
    """Convert uploaded file content to normalized numpy array."""
    nparr = np.frombuffer(file_content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please ensure you are uploading valid image files (PNG/JPG).")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    return img_resized.astype(np.float32) / 255.0

def mask_to_base64(mask):
    """Convert integer mask to RGB Base64 string."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in DAMAGE_MAP_RGB.items():
        rgb[mask == cid] = color
    
    # Convert to PIL Image and then to Base64
    img = Image.fromarray(rgb)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def calculate_stats(pred_mask):
    """Calculate severity and distribution stats."""
    building_pixels = pred_mask[pred_mask > 0]
    total_buildings = len(building_pixels)
    
    if total_buildings == 0:
        return {"severity": 0.0, "distribution": {}}

    counts = {i: int(np.sum(building_pixels == i)) for i in range(1, 5)}
    dist = {i: (counts[i]/total_buildings)*100 for i in range(1, 5)}
    
    # Matching formula from inference.py
    severity = ((counts[2] * 2.5) + (counts[3] * 6.5) + (counts[4] * 10.0)) / (total_buildings * 10.0) * 10.0
    
    return {
        "severity": round(severity, 2),
        "distribution": {DAMAGE_DESCRIPTIONS[i]: round(dist[i], 1) for i in range(1, 5)}
    }

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/analyze")
async def analyze_scene(pre_image: UploadFile = File(...), post_image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        # Load and preprocess
        pre_content = await pre_image.read()
        post_content = await post_image.read()
        
        x1 = process_image(pre_content)
        x2 = process_image(post_content)
        
        inputs = [np.expand_dims(x1, 0), np.expand_dims(x2, 0)]
        
        # Inference
        pred = model.predict(inputs, verbose=0)
        
        # --- Handle Architecture Mismatch ---
        # If model.predict returns a single array, it's (1, 512, 512, 5)
        # If it returns a list, it's [mask_pred, scene_pred]
        if isinstance(pred, list):
            mask_pred = pred[0]
            scene_pred = pred[1] if len(pred) > 1 else None
        else:
            mask_pred = pred
            scene_pred = None
        
        # Process Results
        pred_mask = np.argmax(mask_pred, axis=-1)[0]
        stats = calculate_stats(pred_mask)
        
        # Global Assessment (Fallback if head missing)
        if scene_pred is not None:
            scene_class = int(np.argmax(scene_pred[0]))
            scene_conf = float(np.max(scene_pred[0]))
            scene_label = DAMAGE_DESCRIPTIONS[scene_class]
        else:
            # Heuristic fallback: Use the worst damage found in the mask
            unique_in_mask = np.unique(pred_mask)
            unique_in_mask = unique_in_mask[unique_in_mask > 0]
            worst_class = int(np.max(unique_in_mask)) if len(unique_in_mask) > 0 else 1
            scene_label = DAMAGE_DESCRIPTIONS[worst_class] + " (Heuristic)"
            scene_conf = 1.0 # 100% confidence in our heuristic
        
        mask_b64 = mask_to_base64(pred_mask)
        
        return {
            "mask": f"data:image/png;base64,{mask_b64}",
            "severity": stats["severity"],
            "distribution": stats["distribution"],
            "scene_assessment": scene_label,
            "confidence": round(scene_conf * 100, 1)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
