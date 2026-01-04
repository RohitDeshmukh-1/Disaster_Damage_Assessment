import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks

# Import our custom modules
from dataloader import DisasterDataGenerator
from model import build_siamese_unet

# --- CONFIGURATION ---
# Paths
TRAIN_IMG_DIR = "../data/raw/xbd/tier1/images"
TRAIN_MASK_DIR = "../data/processed/train_masks"

# Hyperparameters
INPUT_SHAPE = (512, 512, 3) # Must match what you set in dataloader
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

# Where to save artifacts
CHECKPOINT_DIR = "../results/models"
LOG_DIR = "../results/logs"

def get_class_weights():
    """
    Optional: Define weights to force model to pay attention to rare classes (Destroyed).
    Format: {Class_ID: Weight}
    """
    return {
        0: 0.1,  # Background (Ignore mostly)
        1: 1.0,  # No Damage
        2: 5.0,  # Minor Damage (Rare)
        3: 5.0,  # Major Damage (Rare)
        4: 10.0  # Destroyed (Very Rare - High Priority)
    }

def main():
    # 1. Setup Directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. Initialize Data Loader
    # Note: We split training data for validation manually for this prototype
    # In production, you'd have a separate validation folder.
    print("Initializing Data Loader...")
    train_gen = DisasterDataGenerator(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR, 
        batch_size=BATCH_SIZE, 
        img_size=INPUT_SHAPE[:2]
    )

    # 3. Build Model
    print("Building Siamese U-Net...")
    model = build_siamese_unet(input_shape=INPUT_SHAPE, num_classes=5)
    
    # 4. Compile Model
    # We use Sparse Categorical Crossentropy because our targets are Integers (0-4)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = keras.losses.SparseCategoricalCrossentropy()
    
    # Metrics: Accuracy is okay, but MeanIoU is better for segmentation
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # 5. Define Callbacks
    
    # Checkpoint: Save the model every time the loss improves
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "xbd_model_best.keras")
    cb_checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        monitor='loss', # In real training, use 'val_loss'
        mode='min',
        verbose=1
    )
    
    # TensorBoard: Visual logs (loss curves)
    log_path = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    cb_tensorboard = callbacks.TensorBoard(log_dir=log_path)
    
    # Early Stopping: Stop if loss stops improving for 5 epochs
    cb_early_stop = callbacks.EarlyStopping(
        monitor='loss', 
        patience=5, 
        restore_best_weights=True
    )

    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Check TensorBoard with: tensorboard --logdir {LOG_DIR}")

    # 6. Start Training
    # Note: Keras 'sample_weight' is tricky with generators. 
    # For a prototype, standard fitting is often stable enough.
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        callbacks=[cb_checkpoint, cb_tensorboard, cb_early_stop],
        # If you want to use validation data, pass a separate validation generator here:
        # validation_data=val_gen 
    )

    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()