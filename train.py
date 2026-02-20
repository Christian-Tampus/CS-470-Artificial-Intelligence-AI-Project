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
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

#Declare Variables
imageSize = 150
batchSize = 32
epochs = 5
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
trainDataSet = tf.data.Dataset.from_tensor_slices((xTrainData, yTrainData)).shuffle(len(y))

#Normalize Images
normalizationLayer = layers.Rescaling(1./255)
trainDataSet = trainDataSet.map(lambda x, y: (normalizationLayer(x), y))

#Build CNN Model
trainingCNNModel = models.Sequential([
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

#Terminate Program
print("[SYSTEM MESSAGE] Program Terminated...")