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
epochs = 5 #Try different epochs to improve accuracy, remember to record data
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
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.Resizing(imageSize, imageSize)
])
trainDataSet = trainDataSet.map(lambda x, y: (dataAugmentation(x, training = True), y)).batch(batchSize)
testingDataSet = testingDataSet.batch(batchSize)

#Build CNN Model
trainingCNNModel = models.Sequential([
    layers.Input(shape = (imageSize, imageSize, 3)),
    layers.Conv2D(32, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation = "relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation = "sigmoid")
])

#Compile CNN Model
trainingCNNModel.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

#Train CNN Model
trainingCNNModel.fit(trainDataSet, validation_data = testingDataSet, epochs = epochs)

#Save CNN Model
saveCNNModelDirectory = os.path.join(trainingModelsDirectory, "CNN_Model_1.h5")
trainingCNNModel.save(saveCNNModelDirectory)

#Terminate Program
print("[SYSTEM MESSAGE] Program Terminated...")