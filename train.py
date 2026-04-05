#UPDATE VERSION [50]

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
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL" : "FISH_DISH_ATTRIBUTE_CLASSIFIER_MODEL_VERSION_",
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
#Model Augmentation Config Table
#==================================================
MODEL_AUGMENTATION_CONFIG_TABLE = {
    "MAIN_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.1},
        {"Type": "RandomZoom", "Value": 0.1},
    ],
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": [
        {"Type": "RandomFlip", "Value": "horizontal"},
        {"Type": "RandomRotation", "Value": 0.2},
        {"Type": "RandomZoom", "Value": 0.2},
        {"Type": "RandomContrast", "Value": 0.1},
    ],
}

#==================================================
#Model Preprocess Grayscale
#==================================================
MODEL_PREPROCESS_GRAYSCALE = {
    "MAIN_CLASSIFIER_MODEL": False,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": True,
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL" : False,
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": False,
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
#Train Models
#==================================================
TRAIN_MODELS = {
    "MAIN_CLASSIFIER_MODEL": True,
    "CAR_MODEL_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "CAT_BREED_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "DOG_BREED_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "HUMAN_RACE_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "CHARACTER_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": False,
    "PLANE_MODEL_ATTRIBUTE_CLASSIFIER_MODEL" : True,
    "FISH_SPECIES_ATTRIBUTE_CLASSIFIER_MODEL": True,
    "FOOD_DISH_ATTRIBUTE_CLASSIFIER_MODEL": True,
    "TOOL_TYPE_ATTRIBUTE_CLASSIFIER_MODEL": True,
    "COMPUTERPART_PART_ATTRIBUTE_CLASSIFIER_MODEL": True,
}

#==================================================
#Global Variables
#==================================================
batchSize = 32
currentDirectory = Path(__file__).resolve().parent
CURRENT_MODEL_IN_TRAINING = ""

#==================================================
#Load All Class Images Function
#==================================================
def loadImages(trainingSetDirectory, classNames):
    allImagesArray = []
    allLabelsArray = []
    for classLabel, className in enumerate(classNames):
        classFolder = trainingSetDirectory / className
        for file in os.listdir(classFolder):
            imagePath = classFolder / file
            try:
                image = tf.keras.utils.load_img(imagePath, target_size = (MODEL_IMAGE_SIZE[CURRENT_MODEL_IN_TRAINING], MODEL_IMAGE_SIZE[CURRENT_MODEL_IN_TRAINING]))
                imageArray = tf.keras.utils.img_to_array(image)
            except Exception as error:
                print("[SYSTEM ERROR] Exception Error: ", error, " File Path: ", imagePath)
                continue
            allImagesArray.append(imageArray)
            allLabelsArray.append(classLabel)
    return allImagesArray, allLabelsArray

#==================================================
#Train Models
#==================================================
for MODEL_NAME, SHOULD_TRAIN_MODEL in TRAIN_MODELS.items():
    if SHOULD_TRAIN_MODEL == True:
        print("============================================================")
        print("[SYSTEM MESSAGE] Now Training Model: ", MODEL_NAME)
        CURRENT_MODEL_IN_TRAINING = MODEL_NAME

        #==================================================
        #Local Variables
        #==================================================
        trainingSetDirectory = currentDirectory / MODEL_TRAINING_SET_DIRECTORIES[MODEL_NAME]
        AIModelsDirectory = os.path.join(currentDirectory, "AIModels")

        #==================================================
        #Detect & Display Classes
        #==================================================
        classNames = sorted(os.listdir(trainingSetDirectory))
        classQuantity = len(classNames)
        print("[SYSTEM MESSAGE] Detected Classes: ", classNames)
        print("[SYSTEM MESSAGE] Number Of Classes: ", classQuantity)

        #==================================================
        #Load All Class Images
        #==================================================
        print("[SYSTEM MESSAGE] Loading Images...")
        xImages, yLabels = loadImages(trainingSetDirectory, classNames)
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
        dataAugmentationArray = []
        for augmentation in MODEL_AUGMENTATION_CONFIG_TABLE[MODEL_NAME]:
            type = augmentation.get("Type")
            if type == "RandomFlip":
                dataAugmentationArray.append(layers.RandomFlip(augmentation.get("Value")))
            elif type == "RandomRotation":
                dataAugmentationArray.append(layers.RandomRotation(augmentation.get("Value")))
            elif type == "RandomZoom":
                dataAugmentationArray.append(layers.RandomZoom(augmentation.get("Value")))
            elif type == "RandomContrast":
                dataAugmentationArray.append(layers.RandomContrast(augmentation.get("Value")))
        dataAugmentation = tf.keras.Sequential(dataAugmentationArray)

        #==================================================
        #Preprocessing Function
        #==================================================
        def preprocess(x, y):
            if MODEL_PREPROCESS_GRAYSCALE[CURRENT_MODEL_IN_TRAINING] == True:
                x = dataAugmentation(x, training = True)
                x = preprocess_input(x)
            else:
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
            input_shape = (MODEL_IMAGE_SIZE[CURRENT_MODEL_IN_TRAINING], MODEL_IMAGE_SIZE[CURRENT_MODEL_IN_TRAINING], 3),
            weights = "imagenet"
        )
        efficientNetB0BaseModel.trainable = False

        #==================================================
        #Build Model
        #==================================================
        trainingModel = models.Sequential([
            layers.Input(shape = (MODEL_IMAGE_SIZE[CURRENT_MODEL_IN_TRAINING], MODEL_IMAGE_SIZE[CURRENT_MODEL_IN_TRAINING], 3)),
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
            filepath = os.path.join(AIModelsDirectory, MODEL_NAMES[MODEL_NAME] + str(MODEL_VERSIONS[MODEL_NAME]) + ".h5"),
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

        print("[SYSTEM MESSAGE] Finished Training Model: ", MODEL_NAME)
        print("============================================================")

#==================================================
#Execute test.py To Test Model
#==================================================
subprocess.run([sys.executable, "test.py"])

#==================================================
#Terminate Program
#==================================================
print("[SYSTEM MESSAGE] Train.py Program Terminated...")