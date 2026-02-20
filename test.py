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
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

#Declare Variables
imageSize = 150
currentDirectory = Path(__file__).resolve().parent
testingSetDirectory = currentDirectory / "DataSets" / "TestingSet"
trainingCNNModelDirectory = currentDirectory / "TrainingModels" / "CNN_Model_1.h5"

#Load CNN Model
trainingCNNModel = tf.keras.models.load_model(trainingCNNModelDirectory)

#Predict Image Function
def predictImage(directory):
    image = Image.open(directory).resize((imageSize, imageSize))
    imageArray = np.array(image) / 255.0
    imageArray = np.expand_dims(imageArray, axis = 0)
    prediction = trainingCNNModel.predict(imageArray, verbose = 0)[0][0]
    if prediction > 0.5:
        return "Dog"
    return "Cat"

#Test CNN Model
correctResults = 0
testSize = 0
for file in os.listdir(testingSetDirectory):
    testSize += 1
    imagePath = testingSetDirectory / file
    result = predictImage(imagePath)
    print(f"[SYSTEM MESSAGE] Prediction For File: {file} Result: {result}")
    if "dog" in file.lower() and result.lower() == "dog" or "cat" in file.lower() and result.lower() == "cat":
        correctResults += 1

#Calculate & Print Out Accuracy
print("[SYSTEM MESSAGE] Accuracy: " + str(round((correctResults / testSize) * 100, 2)) + "%")

#Terminate Program
print("[SYSTEM MESSAGE] Program Terminated...")