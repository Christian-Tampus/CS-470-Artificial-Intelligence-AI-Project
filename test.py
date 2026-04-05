#UPDATE VERSION [51]

#==================================================
#Class: CS-470 Artificial Intelligence
#Professor: Amit Das
#Name: Christian Tampus
#Description: General Purpose Image Classifier & Analyzer
#Assignment: Semester Project
#==================================================

#==================================================
#Start Program
#==================================================
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
#Model Names
#==================================================
MODEL_NAMES = {
    "MAIN_CLASSIFIER_MODEL": "MAIN_CLASSIFIER_MODEL_VERSION_",
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL" : "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL" : "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL" : "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL" : "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL" : "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
}

#==================================================
#Model Versions
#==================================================
MODEL_VERSIONS = {
    "MAIN_CLASSIFIER_MODEL": 6,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 2,
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL" : 1,
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 1,
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": 1,
}

#==================================================
#Model Training Set Directories
#==================================================
MODEL_TRAINING_SET_DIRECTORIES = {
    "MAIN_CLASSIFIER_MODEL": Path("DataSets") / "TrainingSet",
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Cars",
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Cats",
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Dogs",
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Humans",
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Characters",
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL" : Path("DataSets") / "AttributeTrainingSet" / "Planes",
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Fish",
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Food",
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "Tools",
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": Path("DataSets") / "AttributeTrainingSet" / "ComputerParts",
}

#==================================================
#Attribute Accuracy
#==================================================
ATTRIBUTE_ACCURACY = {
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Car Model",
        "Total": 0,
        "Predicted": 0,
    },
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Cat Breed",
        "Total": 0,
        "Predicted": 0,
    },
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Dog Breed",
        "Total": 0,
        "Predicted": 0,
    },
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Human Race",
        "Total": 0,
        "Predicted": 0,
    },
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Character Type",
        "Total": 0,
        "Predicted": 0,
    },
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Plane Model",
        "Total": 0,
        "Predicted": 0,
    },
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Fish Species",
        "Total": 0,
        "Predicted": 0,
    },
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Food Dish",
        "Total": 0,
        "Predicted": 0,
    },
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Tool Type",
        "Total": 0,
        "Predicted": 0,
    },
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": {
        "Attribute": "Computer Part",
        "Total": 0,
        "Predicted": 0,
    },
}

#==================================================
#Model Image Size
#==================================================
MODEL_IMAGE_SIZE = {
    "MAIN_CLASSIFIER_MODEL": 224,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 128,
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL" : 224,
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": 224,
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": 224,
}

#==================================================
#Global Variables
#==================================================
currentDirectory = Path(__file__).resolve().parent
testingSetDirectory = currentDirectory / "DataSets" / "TestingSet"
MAIN_CLASSIFIER_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["MAIN_CLASSIFIER_MODEL"]
CAR_MODEL_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]
CAT_BREED_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]
DOG_BREED_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]
HUMAN_RACE_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]
CHARACTER_TYPE_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]
PLANE_MODEL_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]
FISH_SPECIES_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]
FOOD_DISH_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]
TOOL_TYPE_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]
COMPUTERPART_PART_ANALYZER_MODEL_TESTING_SET_DIRECTORY = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]
MAIN_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["MAIN_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["MAIN_CLASSIFIER_MODEL"]) + ".h5")
CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")
COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY = currentDirectory / "AIModels" / (MODEL_NAMES["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"] + str(MODEL_VERSIONS["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]) + ".h5")

#==================================================
#Detect & Display Classes
#==================================================
MAIN_CLASSIFIER_CLASS_NAMES = sorted(os.listdir(MAIN_CLASSIFIER_TESTING_SET_DIRECTORY))
MAIN_CLASSIFIER_CLASS_QUANTITY = len(MAIN_CLASSIFIER_CLASS_NAMES)
CAR_MODEL_ANALYZER_CLASS_NAMES = sorted(os.listdir(CAR_MODEL_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
CAR_MODEL_ANALYZER_CLASS_QUANTITY = len(CAR_MODEL_ANALYZER_CLASS_NAMES)
CAT_BREED_ANALYZER_CLASS_NAMES = sorted(os.listdir(CAT_BREED_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
CAT_BREED_ANALYZER_CLASS_QUANTITY = len(CAT_BREED_ANALYZER_CLASS_NAMES)
DOG_BREED_ANALYZER_CLASS_NAMES = sorted(os.listdir(DOG_BREED_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
DOG_BREED_ANALYZER_CLASS_QUANTITY = len(DOG_BREED_ANALYZER_CLASS_NAMES)
HUMAN_RACE_ANALYZER_CLASS_NAMES = sorted(os.listdir(HUMAN_RACE_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
HUMAN_RACE_ANALYZER_CLASS_QUANTITY = len(HUMAN_RACE_ANALYZER_CLASS_NAMES)
CHARACTER_TYPE_ANALYZER_CLASS_NAMES = sorted(os.listdir(CHARACTER_TYPE_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
CHARACTER_TYPE_ANALYZER_CLASS_QUANTITY = len(CHARACTER_TYPE_ANALYZER_CLASS_NAMES)
PLANE_MODEL_ANALYZER_CLASS_NAMES = sorted(os.listdir(PLANE_MODEL_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
PLANE_MODEL_ANALYZER_CLASS_QUANTITY = len(PLANE_MODEL_ANALYZER_CLASS_NAMES)
FISH_SPECIES_ANALYZER_CLASS_NAMES = sorted(os.listdir(FISH_SPECIES_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
FISH_SPECIES_ANALYZER_CLASS_QUANTITY = len(FISH_SPECIES_ANALYZER_CLASS_NAMES)
FOOD_DISH_ANALYZER_CLASS_NAMES = sorted(os.listdir(FOOD_DISH_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
FOOD_DISH_ANALYZER_CLASS_QUANTITY = len(FOOD_DISH_ANALYZER_CLASS_NAMES)
TOOL_TYPE_ANALYZER_CLASS_NAMES = sorted(os.listdir(TOOL_TYPE_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
TOOL_TYPE_ANALYZER_CLASS_QUANTITY = len(TOOL_TYPE_ANALYZER_CLASS_NAMES)
COMPUTERPART_PART_ANALYZER_CLASS_NAMES = sorted(os.listdir(COMPUTERPART_PART_ANALYZER_MODEL_TESTING_SET_DIRECTORY))
COMPUTERPART_PART_ANALYZER_CLASS_QUANTITY = len(COMPUTERPART_PART_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] MAIN_CLASSIFIER_CLASS_NAMES Classes: ", MAIN_CLASSIFIER_CLASS_NAMES)
print("[SYSTEM MESSAGE] MAIN_CLASSIFIER_CLASS_QUANTITY Quantity: ", MAIN_CLASSIFIER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] CAR_MODEL_ANALYZER_CLASS_NAMES Classes: ", CAR_MODEL_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] CAR_MODEL_ANALYZER_CLASS_QUANTITY Quantity: ", CAR_MODEL_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] CAT_BREED_ANALYZER_CLASS_NAMES Classes: ", CAT_BREED_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] CAT_BREED_ANALYZER_CLASS_QUANTITY Quantity: ", CAT_BREED_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] DOG_BREED_ANALYZER_CLASS_NAMES Classes: ", DOG_BREED_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] DOG_BREED_ANALYZER_CLASS_QUANTITY Quantity: ", DOG_BREED_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] HUMAN_RACE_ANALYZER_CLASS_NAMES Classes: ", HUMAN_RACE_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] HUMAN_RACE_ANALYZER_CLASS_QUANTITY Quantity: ", HUMAN_RACE_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] CHARACTER_TYPE_ANALYZER_CLASS_NAMES Classes: ", CHARACTER_TYPE_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] CHARACTER_TYPE_ANALYZER_CLASS_QUANTITY Quantity: ", CHARACTER_TYPE_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] PLANE_MODEL_ANALYZER_CLASS_NAMES Classes: ", PLANE_MODEL_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] PLANE_MODEL_ANALYZER_CLASS_QUANTITY Quantity: ", PLANE_MODEL_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] FISH_SPECIES_ANALYZER_CLASS_NAMES Classes: ", FISH_SPECIES_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] FISH_SPECIES_ANALYZER_CLASS_QUANTITY Quantity: ", FISH_SPECIES_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] FOOD_DISH_ANALYZER_CLASS_NAMES Classes: ", FOOD_DISH_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] FOOD_DISH_ANALYZER_CLASS_QUANTITY Quantity: ", FOOD_DISH_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] TOOL_TYPE_ANALYZER_CLASS_NAMES Classes: ", TOOL_TYPE_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] TOOL_TYPE_ANALYZER_CLASS_QUANTITY Quantity: ", TOOL_TYPE_ANALYZER_CLASS_QUANTITY)
print("[SYSTEM MESSAGE] COMPUTERPART_PART_ANALYZER_CLASS_NAMES Classes: ", COMPUTERPART_PART_ANALYZER_CLASS_NAMES)
print("[SYSTEM MESSAGE] COMPUTERPART_PART_ANALYZER_CLASS_QUANTITY Quantity: ", COMPUTERPART_PART_ANALYZER_CLASS_QUANTITY)

#==================================================
#Load Models
#==================================================
CLASSIFICATION_MODEL = tf.keras.models.load_model(MAIN_CLASSIFIER_MODEL_DIRECTORY)
CAR_MODEL_ANALYZER_MODEL = tf.keras.models.load_model(CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
CAT_BREED_ANALYZER_MODEL = tf.keras.models.load_model(CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
DOG_BREED_ANALYZER_MODEL = tf.keras.models.load_model(DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
HUMAN_RACE_ANALYZER_MODEL = tf.keras.models.load_model(HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
CHARACTER_TYPE_ANALYZER_MODEL = tf.keras.models.load_model(CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
PLANE_MODEL_ANALYZER_MODEL = tf.keras.models.load_model(PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
FISH_SPECIES_ANALYZER_MODEL = tf.keras.models.load_model(FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
FOOD_DISH_ANALYZER_MODEL = tf.keras.models.load_model(FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
TOOL_TYPE_ANALYZER_MODEL = tf.keras.models.load_model(TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)
COMPUTERPART_PART_ANALYZER_MODEL = tf.keras.models.load_model(COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL_DIRECTORY)

#==================================================
#Classify Image Function
#==================================================
def classifyImage(directory):
    try:
        image = Image.open(directory).convert("RGB").resize((MODEL_IMAGE_SIZE["MAIN_CLASSIFIER_MODEL"], MODEL_IMAGE_SIZE["MAIN_CLASSIFIER_MODEL"]))
    except Exception as error:
        print("[SYSTEM ERROR] Exception Error: ", error, " File Path: ", directory)
        return None, None
    imageArray = np.array(image, dtype = np.float32)
    imageArray = preprocess_input(imageArray)
    imageArray = np.expand_dims(imageArray, axis = 0)
    classifications = CLASSIFICATION_MODEL.predict(imageArray, verbose = 0)[0]
    classificationIndex = np.argmax(classifications)
    classificationClass = MAIN_CLASSIFIER_CLASS_NAMES[classificationIndex]
    confidence = classifications[classificationIndex]
    return classificationClass, confidence

#==================================================
#Analyze Image Function
#==================================================
def analyzeImage(directory, analyzerModel, attributeNames, modelName):
    try:
        image = Image.open(directory).convert("RGB").resize((MODEL_IMAGE_SIZE[modelName], MODEL_IMAGE_SIZE[modelName]))
    except Exception as error:
        print("[SYSTEM ERROR] Exception Error: ", error, " File Path: ", directory)
        return None, None
    imageArray = np.array(image, dtype = np.float32)
    imageArray = preprocess_input(imageArray)
    imageArray = np.expand_dims(imageArray, axis = 0)
    analysis = analyzerModel.predict(imageArray, verbose = 0)[0]
    analysisIndex = np.argmax(analysis)
    analysisAttribute = attributeNames[analysisIndex]
    confidence = analysis[analysisIndex]
    return analysisAttribute, confidence

#==================================================
#Classification Variables
#==================================================
classificationTrueLabels = []
classificationPredictedLabels = []

#==================================================
#Classify Testing Set
#==================================================
print("============================================================")
print("[SYSTEM MESSAGE] Classification Start!")
for classIndex, className in enumerate(MAIN_CLASSIFIER_CLASS_NAMES):
    print("[SYSTEM MESSAGE] Classification Start For Class: ",className)
    classFolder = testingSetDirectory / className
    if not os.path.isdir(classFolder):
        continue
    for file in os.listdir(classFolder):
        print("[SYSTEM MESSAGE] Classifying Image: ", file)
        imagePath = classFolder / file
        classificationClass, classificationConfidence = classifyImage(imagePath)
        if classificationClass is None:
            continue
        classificationIndex = MAIN_CLASSIFIER_CLASS_NAMES.index(classificationClass)
        classificationTrueLabels.append(classIndex)
        classificationPredictedLabels.append(classificationIndex)
        print("[SYSTEM MESSAGE] Image: ", file, " Classified!")
        print("[SYSTEM MESSAGE] Now Analyzing Image: ", file)
        attributeLabel = ""
        analysisAttribute = ""
        analysisConfidence = ""
        classValue = file.split("_")[0]
        if classValue == "Car":
            ATTRIBUTE_ACCURACY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Cat":
            ATTRIBUTE_ACCURACY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Dog":
            ATTRIBUTE_ACCURACY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Human":
            ATTRIBUTE_ACCURACY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Character":
            ATTRIBUTE_ACCURACY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Plane":
            ATTRIBUTE_ACCURACY["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Fish":
            ATTRIBUTE_ACCURACY["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Food":
            ATTRIBUTE_ACCURACY["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "Tools":
            ATTRIBUTE_ACCURACY["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        elif classValue == "ComputerParts":
            ATTRIBUTE_ACCURACY["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]["Total"] += 1
        match classificationClass:
            case "Cars":
                print("[SYSTEM MESSAGE] Analyzing Car Model...")
                attributeLabel = " Attribute [Car Model]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, CAR_MODEL_ANALYZER_MODEL, CAR_MODEL_ANALYZER_CLASS_NAMES, "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Cats":
                print("[SYSTEM MESSAGE] Analyzing Cat Breed...")
                attributeLabel = " Attribute [Cat Breed]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, CAT_BREED_ANALYZER_MODEL, CAT_BREED_ANALYZER_CLASS_NAMES, "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Dogs":
                print("[SYSTEM MESSAGE] Analyzing Dog Breed...")
                attributeLabel = " Attribute [Dog Breed]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, DOG_BREED_ANALYZER_MODEL, DOG_BREED_ANALYZER_CLASS_NAMES, "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Humans":
                print("[SYSTEM MESSAGE] Analyzing Human Race...")
                attributeLabel = " Attribute [Human Race]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, HUMAN_RACE_ANALYZER_MODEL, HUMAN_RACE_ANALYZER_CLASS_NAMES, "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Characters":
                print("[SYSTEM MESSAGE] Analyzing Character Type...")
                attributeLabel = " Attribute [Character Type]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, CHARACTER_TYPE_ANALYZER_MODEL, CHARACTER_TYPE_ANALYZER_CLASS_NAMES, "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Planes":
                print("[SYSTEM MESSAGE] Analyzing Plane Model...")
                attributeLabel = " Attribute [Plane Model]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, PLANE_MODEL_ANALYZER_MODEL, PLANE_MODEL_ANALYZER_CLASS_NAMES, "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Fish":
                print("[SYSTEM MESSAGE] Analyzing Fish Species...")
                attributeLabel = " Attribute [Fish Species]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, FISH_SPECIES_ANALYZER_MODEL, FISH_SPECIES_ANALYZER_CLASS_NAMES, "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Food":
                print("[SYSTEM MESSAGE] Analyzing Food Dish...")
                attributeLabel = " Attribute [Food Dish]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, FOOD_DISH_ANALYZER_MODEL, FOOD_DISH_ANALYZER_CLASS_NAMES, "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "Tools":
                print("[SYSTEM MESSAGE] Analyzing Tool Type...")
                attributeLabel = " Attribute [Tool Type]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, TOOL_TYPE_ANALYZER_MODEL, TOOL_TYPE_ANALYZER_CLASS_NAMES, "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case "ComputerParts":
                print("[SYSTEM MESSAGE] Analyzing Computer Part...")
                attributeLabel = " Attribute [Computer Part]: "
                analysisAttribute, analysisConfidence = analyzeImage(imagePath, COMPUTERPART_PART_ANALYZER_MODEL, COMPUTERPART_PART_ANALYZER_CLASS_NAMES, "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL")
                if analysisAttribute is None:
                    continue
                attributeValue = file.split("_")[1]
                if attributeValue == analysisAttribute:
                    ATTRIBUTE_ACCURACY["COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL"]["Predicted"] += 1
            case _:
                print("[SYSTEM MESSAGE] Unknown Class")
        print("[SYSTEM MESSAGE] [CLASSIFICATION] Image: ", file," Predicted Classification: ", classificationClass, " Actual Classification: ", className, " Confidence: ", round(classificationConfidence * 100, 2)," %")
        print("[SYSTEM MESSAGE] [ANALYSIS] Image: ", file, attributeLabel, analysisAttribute, " Confidence: ", round(analysisConfidence * 100, 2), "%\n")
    print("============================================================")

#==================================================
#Classification Accuracy
#==================================================
classificationAccuracy = round(accuracy_score(classificationTrueLabels, classificationPredictedLabels) * 100, 2)
print("[SYSTEM MESSAGE] Classification Accuracy: " + str(classificationAccuracy) + "%")
print("[SYSTEM MESSAGE] Classification Report:\n",classification_report(classificationTrueLabels, classificationPredictedLabels, target_names = MAIN_CLASSIFIER_CLASS_NAMES))

#==================================================
#Classification Confusion Matrix
#==================================================
confusionMatrix = confusion_matrix(classificationTrueLabels, classificationPredictedLabels)

#==================================================
#Display Classification Confusion Matrix
#==================================================
plt.figure(figsize = (8, 6))
sns.heatmap(confusionMatrix, annot = True, fmt = "d", cmap = "Blues", xticklabels = MAIN_CLASSIFIER_CLASS_NAMES, yticklabels = MAIN_CLASSIFIER_CLASS_NAMES)
plt.title("Classification Confusion Matrix")
plt.xlabel("Classification Predicted Label")
plt.ylabel("Classification True Label")
plt.tight_layout()
plt.show()

#==================================================
#Analysis Accuracy
#==================================================
totalImagesPredicted = 0
totalImagesTested = 0
for attribute, value in ATTRIBUTE_ACCURACY.items():
    totalImagesPredicted += value["Predicted"]
    totalImagesTested += value["Total"]
    print("[SYSTEM MESSAGE] Analysis Accuracy For Attribute ", value["Attribute"], " Accuracy: ", str(round((value["Predicted"] / value["Total"]) * 100, 2)), "% [" + str(value["Predicted"]) + "/" + str(value["Total"]) + "]")
print("[SYSTEM MESSAGE] Total Analysis Accuracy: ", str(round((totalImagesPredicted / totalImagesTested) * 100, 2)),"% [" + str(totalImagesPredicted) + "/" + str(totalImagesTested) + "]\n")

#==================================================
#Terminate Program
#==================================================
print("[SYSTEM MESSAGE] Test.py Program Terminated...")