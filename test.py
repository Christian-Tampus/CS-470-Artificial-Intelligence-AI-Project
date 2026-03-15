#UPDATE VERSION [28]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: Multi-Class Image Classifier AI Model
#Assignment: Semester Project
#==================================================

#Start Program
print("[SYSTEM MESSAGE] Test.py Program Start!")

#==================================================
#Import Dependencies
#==================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#==================================================
#Declare Variables
#==================================================
CURRENT_MODEL_VERSION = 4
imageSize = 224
currentDirectory = Path(__file__).resolve().parent
testingSetDirectory = currentDirectory / "DataSets" / "TestingSet"
trainingSetDirectory = currentDirectory / "DataSets" / "TrainingSet"
AIModelsDirectory = currentDirectory / "AIModels" / ("Training_Model_" + str(CURRENT_MODEL_VERSION) + ".h5")

#==================================================
#Detect & Display Classes
#==================================================
classNames = sorted(os.listdir(trainingSetDirectory))
classQuantity = len(classNames)
print("[SYSTEM MESSAGE] Detected Classes: ", classNames)
print("[SYSTEM MESSAGE] Number Of Classes: ", classQuantity)

#==================================================
#Load Model
#==================================================
AIModel = tf.keras.models.load_model(AIModelsDirectory)

#==================================================
#Predict Image Function
#==================================================
def predictImage(directory):
    try:
        image = Image.open(directory).convert("RGB").resize((imageSize, imageSize))
    except Exception as error:
        print("[SYSTEM ERROR] Exception Error: ", error, " File Path: ", directory)
        return None, None
    imageArray = np.array(image, dtype = np.float32)
    imageArray = preprocess_input(imageArray)
    imageArray = np.expand_dims(imageArray, axis = 0)
    predictions = AIModel.predict(imageArray, verbos = 0)[0]
    predictedIndex = np.argmax(predictions)
    predictedClass = classNames[predictedIndex]
    confidence = predictions[predictedIndex]
    return predictedClass, confidence

#==================================================
#Prediction Variables
#==================================================
trueLabels = []
predictedLabels = []

#==================================================
#Predict Testing Set
#==================================================
print("============================================================")
print("[SYSTEM MESSAGE] Prediction Start!")
for classIndex, className in enumerate(classNames):
    classFolder = testingSetDirectory / className
    if not os.path.isdir(classFolder):
        continue
    for file in os.listdir(classFolder):
        imagePath = classFolder / file
        predictedClass, confidence = predictImage(imagePath)
        if predictedClass is None:
            continue
        predictedIndex = classNames.index(predictedClass)
        trueLabels.append(classIndex)
        predictedLabels.append(predictedIndex)
        print(f"[SYSTEM MESSAGE] File: {file} | Actual: {className} | Prediction: {predictedClass} | Confidence: {confidence:.4f}")

#==================================================
#Accuracy
#==================================================
print("============================================================")
accuracy = round(accuracy_score(trueLabels, predictedLabels), 2)
print("[SYSTEM MESSAGE] Accuracy: " + str(accuracy) + "%")
print("[SYSTEM MESSAGE] Classification Report: ",classification_report(trueLabels, predictedLabels, target_names = classNames))

#==================================================
#Confusion Matrix
#==================================================
confusionMatrix = confusion_matrix(trueLabels, predictedLabels)

#==================================================
#Display Confusion Matrix
#==================================================
plt.figure(figsize = (8, 6))
sns.heatmap(confusionMatrix, annot = True, fmt = "d", cmap = "Blues", xticklabels = classNames, yticklabels = classNames)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

#==================================================
#Terminate Program
#==================================================
print("[SYSTEM MESSAGE] Test.py Program Terminated...")