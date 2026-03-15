#UPDATE VERSION [28]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: Multi-Class Image Classifier AI Model
#Assignment: Semester Project
#==================================================

#==================================================
#Start Program
#==================================================
print("[SYSTEM MESSAGE] Train.py Program Start!")

#==================================================
#Import Dependencies
#==================================================
import os
import subprocess
import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

#==================================================
#Declare Variables
#==================================================
NEW_MODEL_VERSION = 4
imageSize = 224
batchSize = 32
currentDirectory = Path(__file__).resolve().parent
trainingSetDirectory = currentDirectory / "DataSets" / "TrainingSet"
AIModelsDirectory = os.path.join(currentDirectory, "AIModels")

#==================================================
#Detect & Display Classes
#==================================================
classNames = sorted(os.listdir(trainingSetDirectory))
classQuantity = len(classNames)
print("[SYSTEM MESSAGE] Detected Classes: ", classNames)
print("[SYSTEM MESSAGE] Number Of Classes: ", classQuantity)

#==================================================
#Load All Class Images Function
#==================================================
def loadImages():
    allImagesArray = []
    allLabelsArray = []
    for classLabel, className in enumerate(classNames):
        classFolder = trainingSetDirectory / className
        for file in os.listdir(classFolder):
            imagePath = classFolder / file
            try:
                image = tf.keras.utils.load_img(imagePath, target_size = (imageSize, imageSize))
                imageArray = tf.keras.utils.img_to_array(image)
            except Exception as error:
                print("[SYSTEM ERROR] Exception Error: ", error, " File Path: ", imagePath)
                continue
            allImagesArray.append(imageArray)
            allLabelsArray.append(classLabel)
    return allImagesArray, allLabelsArray

#==================================================
#Load All Class Images
#==================================================
print("[SYSTEM MESSAGE] Loading Images...")
xImages, yLabels = loadImages()
print("[SYSTEM MESSAGE] Images Loaded!")

#==================================================
#Prepare Datasets
#==================================================
xTrainData = tf.convert_to_tensor(xImages, dtype = tf.float32)
yTrainData = tf.convert_to_tensor(yLabels)
dataSet = tf.data.Dataset.from_tensor_slices((xTrainData, yTrainData)).shuffle(len(yTrainData))
testingSize = int(0.2 * len(yTrainData))
trainDataSet = dataSet.skip(testingSize)
testingDataSet = dataSet.take(testingSize)

#==================================================
#Data Augmentation
#==================================================
dataAugmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

#==================================================
#Preprocessing Function
#==================================================
def preprocess(x, y):
    x = dataAugmentation(x, training = True)
    x = preprocess_input(x)
    return x, y

#==================================================
#Preprocessing Datasets
#==================================================
trainDataSet = trainDataSet.map(preprocess).shuffle(1000).batch(batchSize).prefetch(tf.data.AUTOTUNE)
testingDataSet = testingDataSet.map(lambda x, y: (preprocess_input(x), y)).batch(batchSize).prefetch(tf.data.AUTOTUNE)

#==================================================
#Base Model w/Transfer Learning (EfficientNetB0)
#==================================================
efficientNetB0BaseModel = EfficientNetB0(
    include_top = False,
    input_shape = (imageSize, imageSize, 3),
    weights = "imagenet"
)
efficientNetB0BaseModel.trainable = False

#==================================================
#Build Model
#==================================================
trainingModel = models.Sequential([
    layers.Input(shape = (imageSize, imageSize, 3)),
    efficientNetB0BaseModel,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    #classQuantity should be the number of classes, so the output should equal to the number of classes for multi-class classification
    layers.Dense(classQuantity, activation = "softmax") #sigmoid => binary classificaiton, softmax => multi-class classification
])

#==================================================
#Compile Model
#==================================================
trainingModel.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
    loss = "sparse_categorical_crossentropy", #binary_crossentropy => binary classification, sparse_categorical_crossentropy => mutl-class classification
    metrics = ["accuracy"]
)
#==================================================
#Callbacks
#==================================================
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

#==================================================
#Train Frozen Base Model (Stage 1)
#==================================================
trainingModel.fit(
    trainDataSet,
    validation_data = testingDataSet,
    epochs = 10,
    callbacks = [learingRateReductionCallback, earlyStoppingCallback, checkpoint]
)
#==================================================
#Fine-Tune Top Layers (Stage 2)
#==================================================
efficientNetB0BaseModel.trainable = True
for layer in efficientNetB0BaseModel.layers[:-20]:
    layer.trainable = False

#==================================================
#Recompile For Fine-Tuning (With Lower LR)
#==================================================
trainingModel.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
    loss = "sparse_categorical_crossentropy", #binary_crossentropy => binary classification, sparse_categorical_crossentropy => mutl-class classification
    metrics = ["accuracy"]
)

#==================================================
#Train Fine-Tune Model
#==================================================
trainingModel.fit(
    trainDataSet,
    validation_data=testingDataSet,
    epochs = 20,
    callbacks = [learingRateReductionCallback, earlyStoppingCallback, checkpoint]
)

#==================================================
#Execute test.py To Test Model
#==================================================
subprocess.run([sys.executable, "test.py"])

#==================================================
#Terminate Program
#==================================================
print("[SYSTEM MESSAGE] Train.py Program Terminated...")