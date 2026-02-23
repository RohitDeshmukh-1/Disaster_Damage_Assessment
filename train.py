import os
import sys
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add 'src' to path
sys.path.append(os.path.abspath("src"))
from model import build_siamese_unet
from dataloader import DisasterDataGenerator

# --- 0. ROBUST CUSTOM METRIC FOR SPARSE IOU ---
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    """Handles shape mismatch by converting softmax to argmax before IoU calculation."""
    def __init__(self, num_classes, name='mean_iou', dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities (None, 512, 512, 5) to class indices (None, 512, 512)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Ensure y_true is also (None, 512, 512) for matching
        y_true = tf.squeeze(y_true, axis=-1) if len(y_true.shape) == 4 else y_true
        
        return super().update_state(y_true, y_pred, sample_weight)

# --- 1. GPU / SPEED SETUP ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU Detected: {gpus}")

# Enable Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed Precision Enabled: {policy.name}")

# --- 2. CONFIGURATION ---
home = os.path.expanduser("~")
KAGGLER_CACHE = os.path.join(home, ".cache", "kagglehub", "datasets", "qianlanzz", "xbd-dataset", "versions", "1", "xbd", "tier1")
TIER1_DIR = KAGGLER_CACHE if os.path.exists(KAGGLER_CACHE) else "data/tier1"

IMAGES_DIR = os.path.join(TIER1_DIR, "images")
MASK_DIR = "data/processed/train_masks"

INPUT_SHAPE = (512, 512, 3)
BATCH_SIZE = 16  
EPOCHS = 50
LEARNING_RATE = 2e-4

os.makedirs("results/models", exist_ok=True)
print(f"Dataset Path: {TIER1_DIR}")

# --- 3. DATA PREPARATION ---
all_pre_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*_pre_disaster.png")))
if not all_pre_files:
    print(f"No images found at {IMAGES_DIR}. Check permissions.")
    sys.exit(1)

train_files, val_files = train_test_split(all_pre_files, test_size=0.15, random_state=42)

train_gen = DisasterDataGenerator(train_files, MASK_DIR, batch_size=BATCH_SIZE, img_size=INPUT_SHAPE[:2], augment=True)
val_gen = DisasterDataGenerator(val_files, MASK_DIR, batch_size=BATCH_SIZE, img_size=INPUT_SHAPE[:2], augment=False)
print(f"Total batches: {len(train_gen)}")

# --- 4. MODEL BUILDING & COMPILATION ---
model = build_siamese_unet(input_shape=INPUT_SHAPE, num_classes=5)

# Use our fixed Sparse IoU metric for the mask head
iou_metric = SparseMeanIoU(num_classes=5, name='mean_iou')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={
        "mask_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        "assessment_output": tf.keras.losses.SparseCategoricalCrossentropy()
    },
    loss_weights={
        "mask_output": 1.0,
        "assessment_output": 0.5 # Give less weight to global assessment
    },
    metrics={
        "mask_output": ['accuracy', iou_metric],
        "assessment_output": ['accuracy']
    },
    jit_compile=False 
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "results/models/xbd_tier1_best.keras", 
        save_best_only=True, 
        monitor='val_mask_output_mean_iou', 
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_mask_output_mean_iou', patience=10, restore_best_weights=True, mode='max')
]

# --- 5. START TRAINING ---
print("Starting Training (XLA Disabled for stability)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)
print("Training Complete.")
