#UPDATE VERSION [14]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: Multi-Class Image Classifier AI Model
#Assignment: Semester Project
#==================================================

#Start Program
print("[SYSTEM MESSAGE] Test.py Program Start!")

#Import Dependencies
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

#Declare Variables
CURRENT_CNN_MODEL_VERSION = 1
imageSize = 224
currentDirectory = Path(__file__).resolve().parent
testingSetDirectory = currentDirectory / "DataSets" / "TestingSet"
trainingCNNModelDirectory = currentDirectory / "TrainingModels" / ("CNN_Model" + str(CURRENT_CNN_MODEL_VERSION) + ".h5")

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

#Declare Result Variables
decimalPlaces = 4
correctResults = 0
testSize = 0
confusionMatrixDataArray = {
    "cats": {
        "TPR": 0,
        "TNR": 0,
        "FPR": 0,
        "FNR": 0,
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    },
    "dogs": {
        "TPR": 0,
        "TNR": 0,
        "FPR": 0,
        "FNR": 0,
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    },
}

#Test CNN Model
print("============================================================")
print("[SYSTEM MESSAGE] Prediction Start!")
for file in os.listdir(testingSetDirectory):
    testSize += 1
    imagePath = testingSetDirectory / file
    result = predictImage(imagePath).lower()
    fileName = file.lower().removesuffix(".jpg")
    print(f"[SYSTEM MESSAGE] Prediction For File: {fileName} Result: {result}")
    if "cat" in fileName:
        if result == "cat":
            confusionMatrixDataArray["cats"]["TP"] += 1
            confusionMatrixDataArray["dogs"]["TN"] += 1
        elif result == "dog":
            confusionMatrixDataArray["cats"]["FN"] += 1
            confusionMatrixDataArray["dogs"]["FP"] += 1
    if "dog" in fileName:
        if result == "dog":
            confusionMatrixDataArray["dogs"]["TP"] += 1
            confusionMatrixDataArray["cats"]["TN"] += 1
        elif result == "cat":
            confusionMatrixDataArray["dogs"]["FN"] += 1
            confusionMatrixDataArray["cats"]["FP"] += 1
    if "dog" in fileName and result == "dog" or "cat" in fileName and result == "cat":
        correctResults += 1

#Calculate Results
print("============================================================")
accuracy = str(round((correctResults / testSize) * 100, decimalPlaces))
for key, value in confusionMatrixDataArray.items():
    value["TPR"] = value["TP"] / (value["TP"] + value["FN"])
    value["TNR"] = value["TN"] / (value["TN"] + value["FP"])
    value["FPR"] = value["FP"] / (value["TN"] + value["FP"])
    value["FNR"] = value["FN"] / (value["TP"] + value["FN"])
    precision = value["TP"] / (value["TP"] + value["FP"])
    recall = value["TP"] / (value["TP"] + value["FN"])
    print("[SYSTEM MESSAGE] " + key.upper() + " Confusion Matrix Data:")
    print("[SYSTEM MESSAGE] True Positive Rate: " + str(round(value["TPR"], decimalPlaces)))
    print("[SYSTEM MESSAGE] True Negative Rate: " + str(round(value["TNR"], decimalPlaces)))
    print("[SYSTEM MESSAGE] False Positive Rate: " + str(round(value["FPR"], decimalPlaces)))
    print("[SYSTEM MESSAGE] False Negative Rate: " + str(round(value["FNR"], decimalPlaces)))
    print("[SYSTEM MESSAGE] Precision: " + str(round(precision, decimalPlaces)))
    print("[SYSTEM MESSAGE] Recall: " + str(round(recall, decimalPlaces)))
    print("[SYSTEM MESSAGE] F1 Score: " + str(round((2 * (precision * recall))/(precision + recall),decimalPlaces)))
    print("============================================================")
print("[SYSTEM MESSAGE] Accuracy: " + accuracy + "%")
print("============================================================")

#Terminate Program
print("[SYSTEM MESSAGE] Test.py Program Terminated...")