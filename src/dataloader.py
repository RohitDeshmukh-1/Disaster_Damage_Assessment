import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras

class DisasterDataGenerator(keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=8, img_size=(512, 512), shuffle=True):
        """
        Custom Keras Data Generator for Siamese U-Net.
        
        Args:
            image_dir (str): Path to raw images (containing *pre_disaster.png)
            mask_dir (str): Path to processed integer masks
            batch_size (int): Number of samples per batch
            img_size (tuple): Target size (height, width) to resize images to
            shuffle (bool): Whether to shuffle data at the end of every epoch
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        # 1. List all PRE-disaster images
        # We use PRE images as the "anchor" to find the corresponding POST and MASK
        self.pre_image_paths = sorted(glob.glob(os.path.join(image_dir, "*_pre_disaster.png")))
        
        # Filter: Only keep samples that actually have a generated mask
        # (Sometimes preprocessing skips corrupt files, so we must check)
        self.valid_indices = []
        for i, path in enumerate(self.pre_image_paths):
            filename = os.path.basename(path).replace("_pre_disaster.png", "_post_disaster.png")
            mask_path = os.path.join(mask_dir, filename)
            if os.path.exists(mask_path):
                self.valid_indices.append(i)
                
        # Update lists to only include valid pairs
        self.pre_image_paths = [self.pre_image_paths[i] for i in self.valid_indices]
        self.indexes = np.arange(len(self.pre_image_paths))
        
        print(f"Found {len(self.pre_image_paths)} valid Pre/Post/Mask triplets.")

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.pre_image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_paths_temp = [self.pre_image_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_paths_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        # X1: Pre-images, X2: Post-images
        X1 = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        X2 = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_size, 1), dtype=np.int32)

        for i, pre_path in enumerate(list_paths_temp):
            # 1. Define Paths
            post_path = pre_path.replace("_pre_disaster.png", "_post_disaster.png")
            filename = os.path.basename(post_path) # Mask has same name as post image
            mask_path = os.path.join(self.mask_dir, filename)

            # 2. Load Images (RGB)
            img_pre = cv2.imread(pre_path)
            img_post = cv2.imread(post_path)
            
            # Convert BGR -> RGB and Normalize (0-1)
            img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB) / 255.0
            img_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2RGB) / 255.0

            # 3. Load Mask (Grayscale)
            # Flag 0 ensures it reads as (H, W), not (H, W, 3)
            mask = cv2.imread(mask_path, 0) 

            # 4. Resize (if necessary)
            if img_pre.shape[:2] != self.img_size:
                img_pre = cv2.resize(img_pre, self.img_size)
                img_post = cv2.resize(img_post, self.img_size)
                # Use Nearest Neighbor for mask to keep classes as integers (0,1,2...), not floats
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

            # 5. Assign to Batch
            X1[i,] = img_pre
            X2[i,] = img_post
            y[i,] = np.expand_dims(mask, axis=-1) # Needs to be (H, W, 1)

        # Return: ([Pre, Post], Mask)
        return [X1, X2], y