#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: Image Classifier AI Model
#Assignment: Semester Project
#==================================================

print("[SYSTEM MESSAGE] Program Start!")

#Import Dependencies
import os
import tensorflow as tf
from tensorflow.keras import layers, models

#Declare Variables
imageSize = 150
batchSize = 32
epochs = 5
currentDirectory = os.path.dirname(os.path.abspath(__file__))
trainingSetDirectory = os.path.join(currentDirectory, "DataSets", "TrainingSet")
trainingModelsDirectory = os.path.join(currentDirectory, "TrainingModels")

#Load Dataset
trainDataSet = tf.keras.preprocessing.image_dataset_from_directory(
    directory = trainingSetDirectory,
    labels = "inferred",
    label_mode = "binary",
    image_size = (imageSize, imageSize),
    batch_size = batchSize
)

#Normalize Images
normalizationLayer = layers.Rescaling(1./255)
trainDataSet = trainDataSet.map(lambda x, y: (normalizationLayer(x), y))

#Build CNN Model
trainingCNNModel = trainingCNNModel.Sequential([
    layers.Input(shape = (imageSize, imageSize, 3)),
    layers.Conv2D(32, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(1, activation = "sigmoid")
])

#Compile CNN Model
trainingCNNModel.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

#Train CNN Model
trainingCNNModel.fit(trainDataSet, epochs = epochs)

#Save CNN Model
saveCNNModelDirectory = os.path.join(trainingModelsDirectory, "CNN_Model_1.h5")
trainingCNNModel.save(saveCNNModelDirectory)

print("[SYSTEM MESSAGE] Program Terminated...")