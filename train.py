#UPDATE VERSION [16]

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
NEW_MODEL_VERSION = 2
imageSize = 224
batchSize = 32 #Reduced batchSize: 64 -> 32
currentDirectory = Path(__file__).resolve().parent
catsDirectory = currentDirectory / "DataSets" / "TrainingSet" / "Cats"
dogsDirectory = currentDirectory / "DataSets" / "TrainingSet" / "Dogs"
trainingModelsDirectory = os.path.join(currentDirectory, "TrainingModels")

#Load Images Function
def loadImages(folder, label):
    allImagesArray = []
    allLabelsArray = []
    for file in os.listdir(folder):
        imagePath = folder / file
        image = tf.keras.utils.load_img(imagePath, target_size = (imageSize, imageSize))
        imageArray = tf.keras.utils.img_to_array(image) / 255.0
        allImagesArray.append(imageArray)
        allLabelsArray.append(label)
    return allImagesArray, allLabelsArray

#Load Images
catImages, catLabels = loadImages(catsDirectory, 0)
dogImages, dogLabels = loadImages(dogsDirectory, 1)

#Combine & Convert To TensorFlow Dataset
xTrainData = tf.convert_to_tensor(catImages + dogImages)
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
    layers.RandomRotation(0.2), #Increased: 0.1 -> 0.2
    layers.RandomZoom(0.2), #Increased: 01. -> 0.2
    layers.RandomConstrast(0.1), #New Constrast Adjustment
])

#Prepare Datasets
trainDataSet = trainDataSet.map(lambda x, y: (dataAugmentation(x, training = True), y)).shuffle(buffer_size = 1000).batch(batchSize).prefetch(tf.data.AUTOTUNE)
testingDataSet = testingDataSet.batch(batchSize).prefetch(tf.data.AUTOTUNE)

#New Base Model With EfficientNetB0 (Transfer Learning)
efficientNetB0BaseModel = EfficientNetB0(
    include_top = False, #New: Remove Original Classification Head
    input_shape = (imageSize, imageSize, 3),
    weights = "imagenet" #New: Load Pretrained ImageNet Weights
)
efficientNetB0BaseModel.trainable = False #New: Freeze Base Layers For Initial Training

#Build Model
trainingModel = models.Sequential([
    layers.Input(shape = (imageSize, imageSize, 3)),
    layers.Lambda(preprocess_input), #New: Preprocess Inputs For EfficientNet
    efficientNetB0BaseModel, #New: Add EfficientNet As Feature Extractor
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation = "sigmoid")
    #Old Code
    #layers.Input(shape = (imageSize, imageSize, 3)),
    #layers.Conv2D(32, (3, 3), activation = "relu"),
    #layers.BatchNormalization(),
    #layers.MaxPooling2D(2, 2),
    #layers.Dropout(0.1),
    #layers.Conv2D(64, (3, 3), activation = "relu"),
    #layers.BatchNormalization(),
    #layers.MaxPooling2D(2, 2),
    #layers.Dropout(0.1),
    #layers.Conv2D(128, (3, 3), activation = "relu"),
    #layers.BatchNormalization(),
    #layers.MaxPooling2D(2, 2),
    #layers.Dropout(0.2),
    #layers.GlobalAveragePooling2D(),
    #layers.Dense(128, activation = "relu"),
    #layers.Dropout(0.3),
    #layers.Dense(1, activation = "sigmoid")
])

#Compile Model
trainingModel.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), #New: Lower Learning Rate For Pretrained Network
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
    filepath = os.path.join(trainingModelsDirectory, "CNN_Model" + str(NEW_MODEL_VERSION) + ".h5"),
    monitor = "val_accuracy",
    save_best_only = True,
    verbose = 1
)

#Initial Training (New: Frozen Base)
trainingModel.fit(
    trainDataSet,
    validation_data = testingDataSet,
    epochs = 10, #New: Initial Training With Frozen Base (Changed Epochs: 50 -> 10)
    callbacks = [learingRateReductionCallback, earlyStoppingCallback, checkpoint]
)

#Fine-Tuning: UnFreeze Top Layers Of Base Model (New)
efficientNetB0BaseModel.trainable = True
for layer in efficientNetB0BaseModel.layers[:-20]: #New: Freeze First Layers, Unfreeze Top 20
    layer.trainable = False

#Recompile with lower learning rate for fine-tuning (NEW)
trainingModel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  #New: Very Low Learning Rate For Fine-Tuning
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

#Fine-Tune Model (New)
trainingModel.fit(
    trainDataSet,
    validation_data=testingDataSet,
    epochs=20,  #New: Fine-Tuning Epochs To 20
    callbacks=[learingRateReductionCallback, earlyStoppingCallback, checkpoint]
)

#Execute test.py To Test Model
subprocess.run([sys.executable, "test.py"])

#Terminate Program
print("[SYSTEM MESSAGE] Train.py Program Terminated...")