import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_siamese_unet(input_shape=(512, 512, 3), num_classes=5):
    """
    Builds a Siamese U-Net for Building Damage Assessment.
    
    Args:
        input_shape: Tuple (H, W, 3) - shape of a SINGLE image (Pre or Post)
        num_classes: 5 (Background, No-Damage, Minor, Major, Destroyed)
        
    Returns:
        keras.Model: The compiled model expecting [input_pre, input_post]
    """
    
    # --- 1. Define Inputs ---
    input_pre = layers.Input(shape=input_shape, name="input_pre_disaster")
    input_post = layers.Input(shape=input_shape, name="input_post_disaster")

    # --- 2. The Shared Encoder (ResNet50 Backbone) ---
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Extract features for U-Net skip connections
    layer_names = [
        "conv1_relu",        # Scale 1/2
        "conv2_block3_out",   # Scale 1/4
        "conv3_block4_out",   # Scale 1/8
        "conv4_block6_out",   # Scale 1/16
        "conv5_block3_out"    # Scale 1/32 (Bottleneck)
    ]
    
    outputs = [base_model.get_layer(name).output for name in layer_names]
    encoder = models.Model(inputs=base_model.input, outputs=outputs, name="shared_encoder")
    encoder.trainable = True 

    # --- 3. Pass Inputs through Shared Encoder ---
    features_pre = encoder(input_pre)
    features_post = encoder(input_post)
    
    p1, p2, p3, p4, p5 = features_pre
    q1, q2, q3, q4, q5 = features_post

    # --- 4. The Fusion Block (Concatenation) ---
    f5 = layers.Concatenate()([p5, q5]) # Bottleneck
    f4 = layers.Concatenate()([p4, q4])
    f3 = layers.Concatenate()([p3, q3])
    f2 = layers.Concatenate()([p2, q2])
    f1 = layers.Concatenate()([p1, q1])

    # --- 5. The Decoder (Upsampling) ---
    def decoder_block(input_tensor, skip_tensor, filters):
        x = layers.UpSampling2D((2, 2))(input_tensor)
        x = layers.Concatenate()([x, skip_tensor])
        
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    # Bottleneck processing
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(f5)
    
    # Upsample pathway
    x = decoder_block(x, f4, 256)
    x = decoder_block(x, f3, 128)
    x = decoder_block(x, f2, 64)
    x = decoder_block(x, f1, 32)
    
    # Final upsample to original resolution
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    # --- 6. The Output Heads ---
    # Head A: Segmentation Mask (Pixel-wise Damage)
    # Mixed precision requires the final output to be float32 for softmax stability
    mask_output = layers.Conv2D(num_classes, 1, activation="softmax", name="mask_output", dtype='float32')(x)

    # Head B: Global Scene Assessment (Classification of overall damage level)
    # We use the bottleneck features f5 for the high-level assessment
    gap = layers.GlobalAveragePooling2D()(f5)
    dense_1 = layers.Dense(128, activation="relu")(gap)
    dense_1 = layers.Dropout(0.3)(dense_1)
    assessment_output = layers.Dense(num_classes, activation="softmax", name="assessment_output", dtype='float32')(dense_1)

    # --- 7. Build Model ---
    model = models.Model(
        inputs=[input_pre, input_post], 
        outputs=[mask_output, assessment_output], 
        name="siamese_multitask_unet"
    )
    
    return model

if __name__ == "__main__":
    model = build_siamese_unet(input_shape=(512, 512, 3))
    model.summary()
