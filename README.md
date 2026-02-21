CS-470 Artificial Intelligence Project
Multi-Class Image Classifier
Project Goals
1.	Create Project File Structure [COMPLETED]
2.	Implement Bare Minimum Working Code [COMPLETED]
3.	Complete Source Code Simple Comment Documentation [COMPLETED]
4.	Improve Accuracy Up To 75+% Without Using Pre-Trained Models [COMPLETED]
5.	First Update Of Detailed Documentation Below Of Every Single Line Of Code In Microsoft Word [COMPLETED]
6.	Calculate True Positive Rate & Display In Standard Output (Print)
7.	Calculate True Negative Rate & Display In Standard Output (Print)
8.	Calculate False Positive Rate & Display In Standard Output (Print)
9.	Calculate False Negative Rate & Display In Standard Output (Print)
10.	Calculate Accuracy & Display In Standard Output (Print)
11.	Calculate Precision & Display In Standard Output (Print)
12.	Calculate Recall & Display In Standard Output (Print)
13.	Calculate F1 Score & Display In Standard Output (Print)
14.	Display Confusion Matrix Using Matplotlib & Seaborn
15.	Display Histogram Of TPR, TNR, FPR, FNR Using Matplotlib
16.	Improve Accuracy Up To 90+% Using Pre-Trained Models
17.	Second Update Of Detailed Documentation Below Of Every Single Line Of Code In Microsoft Word [CHECKPOINT: MUST BE COMPLETED BEFORE FIRST PRESENTATION IN MARCH]
18.	Implement Image Classification To 10 Different Image Types (Cats & Dogs Already Included) [IMPORANT: Make Sure That Each Image Type Has 2000 Total Images Inside A Folder & 1 Folder Of 400 Images Of Random Images, 80% Of Which Is Divided By 10 For Each Image Type, 20% Of Which Is None Of The Above Thus The Model Must Predict “Other”, Thus 32 Images Per Type, 80 “Other” Images]
19.	Final Update Of Detailed Documentation Below Of Every Single Line Of Code In Microsoft Word
20.	Create Full-Stack Web Application

Analysis 1
Date: 2/21/2026

•	CNN_Model1.h5 scored a 78.12% accuracy on unseen image data from the testing set folder.

•	The testing set folder has unseen image data that has a total of 64 images comprised of 32 cat and 32 dog images.

•	The training set folder has two folders one cat folder of 2000 cat images and dog folder of 2000 dog images. Increasing the value may improve quantity and diversity thus could lead to better performance but the improvements can plateau quickly when the dataset is already very large.

•	The imageSize variable is set to 224 to set the image height & width to 224 pixels the tradeoff is if we have larger images the AI model can capture more details at the expense of it being more computationally expensive and thus requiring significantly more training time and memory usage compared to lower resolutions.

•	batchSize variable is set to 64 which affects training time, larger batches requires more GPU memory but improves training time while smaller batches increases training time but requires less memory.

•	imageArray = tf.keras.utils.img_to_array(image) / 255.0 is used for Normalization which is a crucial step for model convergence and thus skipping this can make the learning unstable for the AI model.

•	dataset = tf.dataset.Dataset.from_tensor_slices((xTrainData, yTrainData)).shuffle(len(yTrainData)) Shuffling ensures that the batches are mixed and thus failing to shuffle can lead to poor generalization.

•	testingSize = int(0.2 * len(yTrainData)) Used standard (80/20) split for training and testing, 80% training and 20% testing, having too small of a testing set gives unreliable validation and having too small of a training set produces a lower quality model and accuracy in general.

•	dataAugmentation = tf.keras.Sequential() Data Augmentation helps with generalization by creating varied data from the existing dataset without requiring more data/images. layers.RandomFlip(), layers.RandomRotation(0.1), layers.RandomZoom(0.1), can directly influence the AI model’s performance, having too aggressive values can confuse the AI model but having too weak may underfit it.

•	layers.Conv2D(), layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout() all have affects on the AI Model’s learning and overfitting.

•	Number of filters (32, 64, 128) More filers offer more capacity which can lead to potentially higher accuracy, but can overfit if dataset provided is small.

•	Dropout rates (0.1, 0.2, 0.3) Helps prevent overfitting however if the rate is too high will lead to underfitting but too low will lead to overfitting.

•	layers.Dense(1, activation = “sigmoid”) Number of neurons is 128 for the AI model and affects the model’s ability to learn complex features on the images.

•	Optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005) This is important for convergence if too high can lead to unstable AI training and too low can lead to slow learning.

•	loss = “binary_crossentropy” This is for binary classification wrong loss can lead to poor model quality.

•	Callbacks [learningRateReductionCallback, earlyStoppingCallback, checkpoint] The ReduceLROnPlateau helps refine the learning rate when loss plateaus and improves convergence. EarlyStopping prevents overfitting by stopping training at the right time. ModelCheckpoint saves the best model based on validation accuracy and ensures the final mode is high quality.

•	epochs(50) The epochs determine the number of times the AI model trains over the entire data set, setting it to 50 means that it will train up to 50 times. Having too low of a epoch number will lead to underfitting and too high will lead to overfitting.

•	Considerations for future AI model training improvements for better accuracy and model quality? I plan on using pre-trained models like MobileNet/MobileNetV2/MobileNetV3, ResNet etc.

•	Considerations for future improvements on training time efficiency? Make tensorflow use GPU by downloading drivers that allow it to use my NVIDIA GPU to train much quicker than on my CPU.
