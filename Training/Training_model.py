import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
import cv2
import imghdr

#ALT 3 FOR COMMENTS
import numpy as np

print(tf.__version__)


IMG_SIZE = 256
BATCH_SIZE = 32
valid_types = ['jpg', 'jpeg', 'png','bmp']

# preprocessing data
project_dir = 'C:\\Users\\82109\\Desktop\\Deep Learning Project'
img_dir = os.path.join(project_dir, 'img')

##removing dodgy images
##target_color_space = cv2.COLOR_BGR2RGB
for img_class in os.listdir(img_dir):
   for img in os.listdir(os.path.join(img_dir, img_class)):
       img_path = os.path.join(img_dir, img_class, img)
       try:
           image = cv2.imread(img_path) #gets image
           tip = imghdr.what(img_path)  #img type
           if tip not in valid_types:
               print("image not of valid type")
               os.remove(img_path)
       except Exception as e:
           print('issue with image {}'.format(img_path))
        
#loading data
print("loading data")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    img_dir,
    shuffle = True,
    image_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE
    ) # loads the entire dataset
class_names = dataset.class_names # stores class names
print("display batch:")
for img_batch, label_batch in dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(12):
        print("done")
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(img_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    plt.show()

##for image_batch, label_batch in dataset.take(1):
##    for i in range(12):
##        plt.imshow(image_batch[i].numpy().astype("uint8"))
##        plt.title(class_names[label_batch[i]])
##        plt.show()

# splitting into training and validation sets

def split_dataset(ds, train_split, val_split, test_split, shuffle=True, shuffle_size = 10000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed = 12)
    
    train_size = int(ds_size * train_split)
    val_size = int(ds_size * val_split)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.2, 0.1)
print(len(train_ds))
print(len(test_ds))
print(len(val_ds))

# #caching and prefetching to improve data pipelin
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
# #RESIZEING
resize = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)])
# #AUGMENTATION
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
input_shape = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
n_classes = 4 # CHANGE THE VALUE LATER

# #CREATING A CNN.
model = tf.keras.models.Sequential([
    resize,
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape ),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

model.build(input_shape = input_shape)

model.compile(
    optimizer = 'adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
    )
EPOCHS = 55
print("fitting model")
history = model.fit(
    train_ds,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    verbose=1,
    validation_data = val_ds
    )
print(history)
scores = model.evaluate(test_ds)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
