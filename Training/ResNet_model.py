import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
import cv2
import imghdr
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from memory_profiler import profile

print(tf.__version__)


IMG_SIZE = 256
BATCH_SIZE = 32
valid_types = ['jpg', 'jpeg', 'png','bmp']
input_shape = (IMG_SIZE, IMG_SIZE, 3)
n_classes = 5 # CHANGE THE VALUE LATER
EPOCHS = 10
# preprocessing data
project_dir = 'C:\\Users\\82109\\Desktop\\Deep Learning Project'
img_dir = os.path.join(project_dir, 'img')

##removing dodgy images
##target_color_space = cv2.COLOR_BGR2RGB
def remove_invalid_images():
    for img_class in os.listdir(img_dir):
        for img in os.listdir(os.path.join(img_dir, img_class)):
            img_path = os.path.join(img_dir, img_class, img)
            try:
                image = cv2.imread(img_path) #gets image
                ##image_converted = cv2.cvtColor(img, target_color_space)
                ##cv2.imwrite(img_path, image_converted)
                tip = imghdr.what(img_path)  #img type
                if tip not in valid_types:
                    print("image not of valid type")
                    os.remove(img_path)
            except Exception as e:
                print('issue with image {}'.format(img_path))
        
#loading data
print("loading data")
def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir,
        shuffle = True,
        image_size = (IMG_SIZE, IMG_SIZE),
        batch_size = BATCH_SIZE
        ) # loads the entire dataset
    class_names = dataset.class_names # stores class names
    return dataset, class_names
# print("display batch:")
# for img_batch, label_batch in dataset.take(1):
#     plt.figure(figsize=(10, 10))
#     for i in range(12):
#         print("done")
#         ax = plt.subplot(3, 4, i + 1)
#         plt.imshow(img_batch[i].numpy().astype("uint8"))
#         plt.title(class_names[label_batch[i]])
#         plt.axis("off")
#     plt.show()

def dataset_splitter():
    train_ds = ImageDataGenerator(
        rescale = 1.0/255,
        horizontal_flip = True,
        rotation_range = 10
    )

    train_generator = train_ds.flow_from_directory(
        '../dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )

    valid_ds = ImageDataGenerator(
        rescale = 1.0/255,
        horizontal_flip = True,
        rotation_range = 10
    )

    valid_generator = valid_ds.flow_from_directory(
        '../dataset/val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    test_ds = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True)

    test_generator = test_ds.flow_from_directory(
        '../dataset/test',
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=32,
        class_mode="sparse"
    )

    return train_generator, valid_generator, test_generator


# splitting into training and validation sets
#@profile

# def img_data_augmentation():
#     # #RESIZEING
#     resize = tf.keras.Sequential([
#         layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
#         layers.experimental.preprocessing.Rescaling(1.0/255)])
#     # #AUGMENTATION
#     data_augmentation = tf.keras.Sequential([
#         layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#         layers.experimental.preprocessing.RandomRotation(0.2)
#         ])
#     return resize, data_augmentation
#@profile
def create_model():
# #CREATING A CNN.
    resnet_model = tf.keras.models.Sequential()

    pretrained_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape = input_shape,
        pooling="max",
        classes=n_classes,
        classifier_activation="softmax",
    )
    for layer in pretrained_model.layers:
        layer.trainable = False
    resnet_model.add(pretrained_model)
    resnet_model.add(tf.keras.layers.Flatten())
    resnet_model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    resnet_model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    resnet_model.build(input_shape = input_shape)

    resnet_model.compile(
        optimizer = 'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return resnet_model
#@profile
def train_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        epochs = EPOCHS,
        validation_data = val_ds
        )
    return history

def plot_results(history):
    accuracy = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize = (8,8))
    plt.subplot(1,2,1)
    plt.plot(range(EPOCHS), accuracy, label = 'Training Accuracy')
    plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1,2,2)
    plt.plot(range(EPOCHS),loss, label = 'Training loss')
    plt.plot(range(EPOCHS), val_loss, label = 'Validation loss')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation loss')

if __name__ == "__main__":
    remove_invalid_images()
    dataset, classnames = load_dataset()

    train_ds, val_ds,test_ds = dataset_splitter()
    print(len(train_ds))
    print(len(val_ds))

    # #caching and prefetching to improve data pipelin
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

    model = create_model()

    history = train_model(model, train_ds, val_ds)

    scores = model.evaluate(test_ds)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])

    try:
        model.save("../models/version1")
        print("Model saved successfully.")
    except Exception as e:
        print("Error occurred while saving the model:", str(e))

    plot_results(history)

# model_version = max([int (i) for i in os.listdir("../models") + [0]]) + 1



