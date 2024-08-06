import os
import numpy as np
import tifffile
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import datetime
import psutil
from tensorflow.keras import backend as K
from skimage.transform import resize

# Configuration variables
IMAGE_PATH = 'tomographic_image.tiff'
MASK_PATH = 'mask_FULL.tiff'
OUTPUT_PATH = 'segmented_output_Full.tiff'
CHUNK_SIZE = 10
EPOCHS = 100
BATCH_SIZE = 1  
NUM_Z_SLICES = None
RESIZE_FACTOR = 1  

# GPU Configuration
def configure_gpus():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPUs: {[gpu.name for gpu in physical_devices]}")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found. Using CPU.")
    return physical_devices

# Data Loading and Preprocessing
def load_data_generator(image_path, mask_path, chunk_size, batch_size, num_z_slices=None, resize_factor=1.0):
    image = tifffile.imread(image_path)
    mask = tifffile.imread(mask_path)
    
    original_shape = image.shape

    # If num_z_slices is specified, limit the data
    if num_z_slices is not None:
        image = image[:num_z_slices]
        mask = mask[:num_z_slices]
    
    # Resize x and y dimensions if resize_factor is not 1.0
    if resize_factor != 1.0:
        new_shape = (image.shape[0], 
                     int(image.shape[1] * resize_factor), 
                     int(image.shape[2] * resize_factor))
        image = resize(image, new_shape, order=1, preserve_range=True, anti_aliasing=True)
        mask = resize(mask, new_shape, order=0, preserve_range=True, anti_aliasing=False)
    
    # Normalize image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Ensure mask is binary
    mask = (mask > 0).astype(np.float32)
    
    depth = image.shape[0]
    
    def generator():
        for i in range(0, depth - chunk_size + 1, chunk_size):
            chunk = image[i:i+chunk_size]
            mask_chunk = mask[i:i+chunk_size]
            yield np.expand_dims(chunk, axis=-1), np.expand_dims(mask_chunk, axis=-1)
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((chunk_size, image.shape[1], image.shape[2], 1),
                       (chunk_size, mask.shape[1], mask.shape[2], 1))
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), original_shape

# 3D U-Net Model
def create_3d_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Encoder (downsampling)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)  # Pool only on x and y dimensions
    
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv2)  # Pool only on x and y dimensions
    
    # Bridge
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    
    # Decoder (upsampling)
    up4 = layers.UpSampling3D(size=(1, 2, 2))(conv3)  # Upsample only on x and y dimensions
    up4 = layers.concatenate([up4, conv2])
    conv4 = layers.Conv3D(64, 3, activation='relu', padding='same')(up4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    
    up5 = layers.UpSampling3D(size=(1, 2, 2))(conv4)  # Upsample only on x and y dimensions
    up5 = layers.concatenate([up5, conv1])
    conv5 = layers.Conv3D(32, 3, activation='relu', padding='same')(up5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(conv5)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Dice coefficient for loss and metric
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Custom callback for epoch information
class EpochInfoCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EpochInfoCallback, self).__init__()
        self.epoch_info = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        info = {
            "epoch": epoch + 1,
            "loss": logs.get("loss"),
            "dice_coefficient": logs.get("dice_coefficient"),
            "val_loss": logs.get("val_loss"),
            "val_dice_coefficient": logs.get("val_dice_coefficient"),
            "contains_nan": any(np.isnan(value) for value in logs.values())
        }
        self.epoch_info.append(info)
        # Save epoch info to a file
        with open(f'epoch_info_epoch_{epoch + 1}.json', 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Epoch {epoch + 1} information saved.")

# Training function
def train_model(strategy, train_dataset, val_dataset):
    with strategy.scope():
        model = create_3d_unet((CHUNK_SIZE, None, None, 1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('3d_segmentation_model.keras', save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    epoch_info_cb = EpochInfoCallback()
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb, epoch_info_cb]
    )
    
    return history, model

# Prediction function
def predict_3d_segmentation(model, dataset, original_shape):
    predictions = []
    for batch in dataset:
        print(f"Batch shape: {batch[0].shape}")
        batch_predictions = model.predict(batch[0]) 
        predictions.append(batch_predictions)
    predictions = np.concatenate(predictions, axis=0)

    # Flatten predictions along the z-axis
    flattened_predictions = predictions[..., 0].reshape(-1, predictions.shape[2], predictions.shape[3])
    print(f"Flattened predictions shape: {flattened_predictions.shape}") 

    # Resize the entire 3D stack
    resized_predictions = resize(flattened_predictions, 
                                 (original_shape[0], original_shape[1], original_shape[2]), 
                                 order=1, 
                                 preserve_range=True, 
                                 anti_aliasing=True)
    resized_predictions = resized_predictions[..., np.newaxis]  

    print(f"Resized predictions final shape: {resized_predictions.shape}")  

    return resized_predictions


# Training Report Generation
def generate_training_report(history, model, gpus, dataset_size, train_size, start_time, end_time):
    report = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_image": IMAGE_PATH,
        "input_mask": MASK_PATH,
        "output_path": OUTPUT_PATH,
        "chunk_size": CHUNK_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_z_slices": NUM_Z_SLICES,
        "resize_factor": RESIZE_FACTOR,
        "dataset_size": dataset_size,
        "train_size": train_size,
        "validation_size": dataset_size - train_size,
        "gpu_info": [gpu.name for gpu in gpus] if gpus else "No GPUs used",
        "training_time": str(end_time - start_time),
        "final_train_loss": history.history['loss'][-1],
        "final_train_dice_coefficient": history.history['dice_coefficient'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "final_val_dice_coefficient": history.history['val_dice_coefficient'][-1],
        "best_val_loss": min(history.history['val_loss']),
        "best_val_dice_coefficient": max(history.history['val_dice_coefficient']),
        "model_summary": [],
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }
    
    # Capture model summary
    model.summary(print_fn=lambda x: report["model_summary"].append(x))
    
    # Save report as JSON
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Training report saved as 'training_report.json'")

# Main execution
def main():
    start_time = datetime.datetime.now()
    
    # Configure GPUs
    gpus = configure_gpus()
    
    # Set up MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    full_dataset, original_shape = load_data_generator(IMAGE_PATH, MASK_PATH, CHUNK_SIZE, BATCH_SIZE, NUM_Z_SLICES, RESIZE_FACTOR)
    
    # Split data into train and validation sets
    dataset_size = sum(1 for _ in full_dataset)
    train_size = int(0.8 * dataset_size)
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    
    # Create and train model
    print("Creating and training model...")
    with strategy.scope():
        history, trained_model = train_model(strategy, train_dataset, val_dataset)
    
    # Predict on full dataset
    print("Predicting on full dataset...")
    predictions = predict_3d_segmentation(trained_model, full_dataset, original_shape)
    
    # Save the result
    print("Saving segmentation result...")
    tifffile.imwrite(OUTPUT_PATH, predictions)
    
    end_time = datetime.datetime.now()
    
    # Generate and save training report
    generate_training_report(history, trained_model, gpus, dataset_size, train_size, start_time, end_time)
    
    print(f"Segmentation complete. Result saved as '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()