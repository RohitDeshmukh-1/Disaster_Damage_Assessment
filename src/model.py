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
    # We load ResNet50 without the top layers (include_top=False)
    # We use 'imagenet' weights to speed up convergence
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # We want to extract features at different scales for the U-Net skip connections
    # Layer names for ResNet50: 
    # - conv1_relu (Scale 1/2)
    # - conv2_block3_out (Scale 1/4)
    # - conv3_block4_out (Scale 1/8)
    # - conv4_block6_out (Scale 1/16)
    # - conv5_block3_out (Scale 1/32) -> The bottleneck
    
    layer_names = [
        "conv1_relu",
        "conv2_block3_out",
        "conv3_block4_out",
        "conv4_block6_out",
        "conv5_block3_out"
    ]
    
    # Create a sub-model that outputs these specific layers
    # This acts as our "Feature Extractor"
    outputs = [base_model.get_layer(name).output for name in layer_names]
    encoder = models.Model(inputs=base_model.input, outputs=outputs, name="shared_encoder")
    
    # Lock the first few layers to prevent destroying ImageNet weights early on
    # (Optional, but recommended for small datasets)
    encoder.trainable = True 

    # --- 3. Pass Inputs through Shared Encoder ---
    # We get a list of 5 feature maps for PRE
    features_pre = encoder(input_pre)
    # We get a list of 5 feature maps for POST
    features_post = encoder(input_post)
    
    # Unpack the list (f1 is largest/high-res, f5 is smallest/bottleneck)
    p1, p2, p3, p4, p5 = features_pre
    q1, q2, q3, q4, q5 = features_post

    # --- 4. The Fusion Block (Concatenation) ---
    # At every scale, we concatenate Pre and Post features
    # This forces the model to compare "What was there" vs "What is there now"
    f5 = layers.Concatenate()([p5, q5]) # Bottleneck Fusion
    f4 = layers.Concatenate()([p4, q4])
    f3 = layers.Concatenate()([p3, q3])
    f2 = layers.Concatenate()([p2, q2])
    f1 = layers.Concatenate()([p1, q1])

    # --- 5. The Decoder (Upsampling) ---
    
    def decoder_block(input_tensor, skip_tensor, filters):
        """Standard U-Net decoder block: Upsample -> Concat -> Conv -> Conv"""
        x = layers.UpSampling2D((2, 2))(input_tensor)
        
        # We must concatenate the upsampled input with the fused skip connection
        x = layers.Concatenate()([x, skip_tensor])
        
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    # Bottleneck processing
    x = layers.Conv2D(512, 3, padding="same", activation="relu")(f5)
    
    # Upsample pathway (5 blocks corresponding to ResNet scales)
    x = decoder_block(x, f4, 256) # Scale 1/16
    x = decoder_block(x, f3, 128) # Scale 1/8
    x = decoder_block(x, f2, 64)  # Scale 1/4
    x = decoder_block(x, f1, 32)  # Scale 1/2
    
    # Final upsample to get back to original resolution (1/1)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

    # --- 6. The Output Head ---
    # Output has 5 channels (one for each class 0-4)
    # Softmax activation forces pixels to choose one class
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", name="damage_output")(x)

    # --- 7. Build Model ---
    model = models.Model(inputs=[input_pre, input_post], outputs=outputs, name="siamese_unet")
    
    return model

if __name__ == "__main__":
    # Quick sanity check to print summary
    model = build_siamese_unet(input_shape=(512, 512, 3))
    model.summary()