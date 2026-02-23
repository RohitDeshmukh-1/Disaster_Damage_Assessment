import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras

CLASS_WEIGHTS = {
    0: 0.1,   # Background
    1: 1.0,   # No Damage
    2: 12.0,  # Minor Damage (Increased slightly to improve minority class accuracy)
    3: 12.0,  # Major Damage
    4: 18.0   # Destroyed
}

class DisasterDataGenerator(keras.utils.Sequence):
    def __init__(self, image_list, mask_dir, batch_size=8, img_size=(512, 512), shuffle=True, augment=True):
        """
        Custom Keras Data Generator for Siamese U-Net.
        
        Args:
            image_list (list): List of paths to pre-disaster images.
            mask_dir (str): Path to processed integer masks.
            batch_size (int): Number of samples per batch.
            img_size (tuple): Target size (height, width).
            shuffle (bool): Whether to shuffle data.
            augment (bool): Whether to apply data augmentation.
        """
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Filter: Only keep samples that actually have a generated mask
        self.pre_image_paths = []
        for path in image_list:
            filename = os.path.basename(path).replace("_pre_disaster.png", "_post_disaster.png")
            mask_path = os.path.join(mask_dir, filename)
            if os.path.exists(mask_path):
                self.pre_image_paths.append(path)
                
        self.indexes = np.arange(len(self.pre_image_paths))
        print(f"Generator initialized with {len(self.pre_image_paths)} valid samples (Augment={augment}).")

    def __len__(self):
        return int(np.floor(len(self.pre_image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_paths_temp = [self.pre_image_paths[k] for k in indexes]
        return self.__data_generation(list_paths_temp)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp):
        X1 = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        X2 = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_size, 1), dtype=np.int32)
        y_assessment = np.empty((self.batch_size,), dtype=np.int32)
        sample_weights = np.empty((self.batch_size, *self.img_size), dtype=np.float32)

        for i, pre_path in enumerate(list_paths_temp):
            post_path = pre_path.replace("_pre_disaster.png", "_post_disaster.png")
            filename = os.path.basename(post_path) 
            mask_path = os.path.join(self.mask_dir, filename)

            # Load
            img_pre = cv2.imread(pre_path)
            img_post = cv2.imread(post_path)
            mask = cv2.imread(mask_path, 0) 

            # Initial Resize if needed
            if img_pre.shape[:2] != self.img_size:
                img_pre = cv2.resize(img_pre, self.img_size)
                img_post = cv2.resize(img_post, self.img_size)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            # --- Data Augmentation ---
            if self.augment:
                # Random Flips
                if np.random.rand() > 0.5:
                    flip_code = np.random.choice([0, 1, -1]) # 0: vert, 1: horiz, -1: both
                    img_pre = cv2.flip(img_pre, flip_code)
                    img_post = cv2.flip(img_post, flip_code)
                    mask = cv2.flip(mask, flip_code)
                
                # Random Rotations (90 deg increments)
                if np.random.rand() > 0.5:
                    k = np.random.randint(1, 4)
                    img_pre = np.rot90(img_pre, k)
                    img_post = np.rot90(img_post, k)
                    mask = np.rot90(mask, k)

            # Normalize and Color Convert
            X1[i,] = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB) / 255.0
            X2[i,] = cv2.cvtColor(img_post, cv2.COLOR_BGR2RGB) / 255.0
            y[i,] = np.expand_dims(mask, axis=-1)

            # --- Global Assessment Label ---
            # Most severe damage class present in the scene
            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes > 0] # Ignore background
            y_assessment[i] = np.max(unique_classes) if len(unique_classes) > 0 else 0

            # Calculate Sample Weights
            weights = np.zeros_like(mask, dtype=np.float32)
            for cls_id, weight in CLASS_WEIGHTS.items():
                weights[mask == cls_id] = weight
            sample_weights[i,] = weights

        return (X1, X2), {"mask_output": y, "assessment_output": y_assessment}, {"mask_output": sample_weights}
