#UPDATE VERSION [24]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: Multi-Class Image Classifier AI Model
#Assignment: Semester Project
#==================================================

#Start Program
print("[SYSTEM MESSAGE] Train.py Program Start!")

#Import Dependencies
import os
import subprocess
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

#New Dependencies For Pre-Trained Models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

#Declare Variables
NEW_MODEL_VERSION = 3
imageSize = 224
batchSize = 32
currentDirectory = Path(__file__).resolve().parent
catsDirectory = currentDirectory / "DataSets" / "TrainingSet" / "Cats"
dogsDirectory = currentDirectory / "DataSets" / "TrainingSet" / "Dogs"
AIModelsDirectory = os.path.join(currentDirectory, "AIModels")

#Load Images Function
def loadImages(folder, label):
    allImagesArray = []
    allLabelsArray = []
    for file in os.listdir(folder):
        imagePath = folder / file
        image = tf.keras.utils.load_img(imagePath, target_size = (imageSize, imageSize))
        imageArray = tf.keras.utils.img_to_array(image)
        allImagesArray.append(imageArray)
        allLabelsArray.append(label)
    return allImagesArray, allLabelsArray

#Load Images
catImages, catLabels = loadImages(catsDirectory, 0)
dogImages, dogLabels = loadImages(dogsDirectory, 1)

#Combine & Convert To TensorFlow Dataset
xTrainData = tf.convert_to_tensor(catImages + dogImages, dtype = tf.float32)
yTrainData = tf.convert_to_tensor(catLabels + dogLabels)

#Shuffle Dataset
dataSet = tf.data.Dataset.from_tensor_slices((xTrainData, yTrainData)).shuffle(len(yTrainData))

#Split Dataset Into Training & Testing With (80/20) Split
testingSize = int(0.2 * len(yTrainData))
trainDataSet = dataSet.skip(testingSize)
testingDataSet = dataSet.take(testingSize)

#Data Augmentation (Improved Augmentation)
dataAugmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

#Preprocessing Function
def preprocess(x, y):
    x = dataAugmentation(x, training = True)
    x = preprocess_input(x)
    return x, y

#Prepare Datasets
trainDataSet = trainDataSet.map(preprocess).shuffle(1000).batch(batchSize).prefetch(tf.data.AUTOTUNE)
testingDataSet = testingDataSet.map(lambda x, y: (preprocess_input(x), y)).batch(batchSize).prefetch(tf.data.AUTOTUNE)

#New Base Model With EfficientNetB0 (Transfer Learning)
efficientNetB0BaseModel = EfficientNetB0(
    include_top = False,
    input_shape = (imageSize, imageSize, 3),
    weights = "imagenet"
)
efficientNetB0BaseModel.trainable = False

#Build Model
trainingModel = models.Sequential([
    layers.Input(shape = (imageSize, imageSize, 3)),
    efficientNetB0BaseModel,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation = "sigmoid")
])

#Compile Model
trainingModel.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss = "binary_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

#Callbacks
learingRateReductionCallback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 3,
    min_lr = 1e-6,
    verbose = 1
)
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
    monitor = "val_accuracy",
    patience = 10,
    restore_best_weights = True,
    verbose = 1
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(AIModelsDirectory, "Training_Model_" + str(NEW_MODEL_VERSION) + ".h5"),
    monitor = "val_accuracy",
    save_best_only = True,
    verbose = 1
)

#Stage 1: Train Frozen Base
trainingModel.fit(
    trainDataSet,
    validation_data = testingDataSet,
    epochs = 10,
    callbacks = [learingRateReductionCallback, earlyStoppingCallback, checkpoint]
)

#Stage 2: Fine-Tune Top Layers
efficientNetB0BaseModel.trainable = True
for layer in efficientNetB0BaseModel.layers[:-20]:
    layer.trainable = False

#Recompile with lower learning rate for fine-tuning
trainingModel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

#Fine-Tune Model
trainingModel.fit(
    trainDataSet,
    validation_data=testingDataSet,
    epochs = 20,
    callbacks = [learingRateReductionCallback, earlyStoppingCallback, checkpoint]
)

#Execute test.py To Test Model
subprocess.run([sys.executable, "test.py"])

#Terminate Program
print("[SYSTEM MESSAGE] Train.py Program Terminated...")