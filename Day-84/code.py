import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from myUtils import ImageToArrayPreprocessor
from myUtils import SimplePreprocessor
from myUtils import SimpleDatasetLoader
from pathlib import Path
import numpy as np
import os

# Define the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Load and process the images
print("[INFO] loading images...")
p = Path(args["dataset"])
imagePaths = list(p.glob('./**/*.jpg'))
imagePaths = [str(names) for names in imagePaths]
classNames = [os.path.split(os.path.split((names))[0])[1] for names in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

sp = SimplePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float32") / 255.0

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=config.split, random_state=42)

# Convert labels to vectors
label_bin = LabelBinarizer()
trainY = label_bin.fit_transform(trainY)
testY = label_bin.transform(testY)

# Load the VGG16 network without the top FC layers
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Build the top layers for classification
headModel = Flatten()(baseModel.output)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dense(len(classNames), activation="softmax")(headModel)

# Combine the base and top models
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers in the base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the top layers
print("[INFO] training head...")
model.fit(aug.flow(trainX, trainY, batch_size=config.batch_size),
          validation_data=(testX, testY), epochs=config.warmUp_epochs,
          steps_per_epoch=len(trainX) // config.batch_size, verbose=1)

# Evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=config.batch_size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=classNames))

# Unfreeze the final set of CONV layers
for layer in baseModel.layers[config.unfreeze_layer:]:
    layer.trainable = True

# Recompile the model with SGD and a small learning rate
print("[INFO] re-compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Fine-tune the model
print("[INFO] fine-tuning model...")
model.fit(aug.flow(trainX, trainY, batch_size=config.batch_size),
          validation_data=(testX, testY), epochs=config.epochs,
          steps_per_epoch=len(trainX) // config.batch_size, verbose=1)

# Evaluate the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=config.batch_size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=classNames))

# Save the model to disk
print("[INFO] serializing model...")
model.save(args["model"])

