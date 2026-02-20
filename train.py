#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: Image Classifier AI Model
#Assignment: Semester Project
#==================================================

#Start Program
print("[SYSTEM MESSAGE] Program Start!")

#Import Dependencies
import os
import subprocess
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from pathlib import Path

#Declare Variables
imageSize = 180
batchSize = 32
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

#Data Augmentation
dataAugmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.15),
    layers.RandomTranslation(0.15, 0.15),
    layers.Resizing(imageSize, imageSize)
])

#Prepare Datasets
trainDataSet = trainDataSet.map(lambda x, y: (dataAugmentation(x, training = True), y)).shuffle(buffer_size = 1000).batch(batchSize).prefetch(tf.data.AUTOTUNE)
testingDataSet = testingDataSet.batch(batchSize).prefetch(tf.data.AUTOTUNE)

#Build CNN Model
trainingCNNModel = models.Sequential([
    layers.Input(shape = (imageSize, imageSize, 3)),
    layers.Conv2D(32, (3, 3), activation = "relu", kernel_regularizer = regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation = "relu", kernel_regularizer = regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), activation = "relu", kernel_regularizer = regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation = "relu", kernel_regularizer = regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation = "sigmoid")
])

#Compile CNN Model
trainingCNNModel.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
    loss = "binary_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

#Callbacks: Learning Rate Reduction & Early Stopping
learingRateReductionCallback = tf.keras.callbacks.ReduceLROnPlateau(
    mointor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-6, verbose = 1
)
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
    monitor = "val_accuracy", patience = 5, restore_best_weights = True, verbose = 1
)

#Train CNN Model
epochs = 50
print("[SYSTEM MESSAGE] Testing With Epochs:", epochs)

trainingCNNModel.fit(trainDataSet, validation_data = testingDataSet, epochs = epochs, callbacks = [learingRateReductionCallback, earlyStoppingCallback])

#Save CNN Model
saveCNNModelDirectory = os.path.join(trainingModelsDirectory, "CNN_Model_1.h5")
trainingCNNModel.save(saveCNNModelDirectory)

#Execute test.py To Test Model
subprocess.run([sys.executable, "test.py"])

#Terminate Program
print("[SYSTEM MESSAGE] Program Terminated...")