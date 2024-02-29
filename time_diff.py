import tensorflow as tf
import time

# Define your dataset loading function or method (train_ds, test_ds, val_ds)

# Define the directory where your dataset is stored
dataset_dir = 'C:\\Users\\82109\\Desktop\\Deep Learning Project\\img'

# Define a function to load the dataset with pipeline optimizations
def load_dataset_with_optimizations(dataset_dir):
    # Create TensorFlow data loaders with caching, shuffling, and prefetching
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        batch_size=32,
        image_size=(256, 256)
    )
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds

# Define a function to load the dataset without pipeline optimizations
def load_dataset_without_optimizations(dataset_dir):
    # Create TensorFlow data loaders without caching, shuffling, and prefetching
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        batch_size=32,
        image_size=(256, 256)
    )
    return train_ds

# Measure the time it takes to load the dataset with optimizations
start_time = time.time()
train_ds_with_optimizations = load_dataset_with_optimizations(dataset_dir)
end_time = time.time()
loading_time_with_optimizations = end_time - start_time

# Measure the time it takes to load the dataset without optimizations
start_time = time.time()
train_ds_without_optimizations = load_dataset_without_optimizations(dataset_dir)
end_time = time.time()
loading_time_without_optimizations = end_time - start_time

# Print the loading times
print("Loading time with optimizations:", loading_time_with_optimizations)
print("Loading time without optimizations:", loading_time_without_optimizations)

# Calculate the difference in loading times
time_difference = loading_time_without_optimizations - loading_time_with_optimizations
print("Difference in loading time:", time_difference)
