# import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



"""
converts the label to np aray
"""
def process_label(labels_list):
	labels_list = np.array(labels_list)
	# make label list binary, required by tensor flow
	lb = LabelBinarizer()
	labels_list = lb.fit_transform(labels_list)
	labels_list = to_categorical(labels_list)
	return(labels_list,lb)
"""
Gets head model from base model
"""
def get_head_model(baseModel):
	# construct the head of the model that will be placed on top of the
	# the base model
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)
	return(headModel)


##########################   Main Program  #################################

"""
File and directory names
"""
dataset_dir = "dataset2"
#generates this plot at the end of training
plot_file = "train_plot.png"
model_file = "model\\mask_detector.model"
#get full dir name from current file
dataset_dir_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,dataset_dir)
model_file_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) ,model_file)

"""
Initial values
"""
#Epoch: number of times of error correction
epochs = 20
#each epoch is broken to baches for faster calculation
batch_size = 32
#learning rate:how much to change the model in response to the estimated error each time the model weights are updated
#.Choosing the learning rate is challenging as a value too small may result in a long training process that could get stuck,
# whereas a value too large may result in learning a sub-optimal set of weights too fast or an unstable training process
initial_learning_rate = 1e-4

"""
Load images from dataset
"""
# Get the list of images in our dataset directory
print("loading images...")
imagepaths_list = list(paths.list_images(dataset_dir_full_path))


image_list = []
labels_list = []

"""
Step 1: Normalize the images (make them all 224x224 pixels, normalize the RGB values
"""
for imagePath in imagepaths_list: #load each image in a loop and append image_list and labels_list


	# extract the label (eg "mask" or "without-mask") from the filename
	label = imagePath.split(os.path.sep)[-2]  #imagePath = C:\\code\\mask-detector\\dataset2\\without_mask\\0.jpg
	# use tensorflow functions load_img() img_to_array() and  preprocess_input() to normalize data
	# load the input image as (224x224) pixels - make all images same pixel size
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)  #normalization of each [R,G,B] of images

	# update the image_list and labels_list lists, respectively
	image_list.append(image)
	labels_list.append(label)

# convert the image_list and labels_list to NumPy arrays for faster processing
image_list = np.array(image_list, dtype="float32")
(labels_list,lb) = process_label(labels_list)

"""
Step 2: Divide images for training (80%) and testing (20%)
"""
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing.
(trainX, testX, trainY, testY) = train_test_split(image_list, labels_list,
	test_size=0.20, stratify=labels_list, random_state=42)

"""
Step 3: Perform image augumentation
"""
# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

"""
Step 4: Load untrained MobilNetV2 model
"""
# load the MobileNetV2 model
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
model = Model(inputs=baseModel.input, outputs=get_head_model(baseModel))
# loop over all layers in the base model and freeze them so they will
for layer in baseModel.layers:
	layer.trainable = False

"""
Step 5: Compile the model
"""
# compile our model
print("compiling model...")
#adam is the optimizer for the model
opt = Adam(lr=initial_learning_rate, decay=initial_learning_rate / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

"""
Step 6: Train and fit the model
"""

# train the model
print("fitting/training the model...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=batch_size),
	steps_per_epoch=len(trainX) // batch_size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch_size,
	epochs=epochs)

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=batch_size)
#convert to numpi array
predIdxs = np.argmax(predIdxs, axis=1)

# print classification report
print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

"""
Step 7: Save the model file 
"""
# save the model to disk
print("saving mask detector model...")
model.save(model_file_full_path, save_format="h5")

"""
Step 8: Plot error graphs using matplotlib
"""
# plot the training loss and accuracy
#https://stackoverflow.com/questions/51344839/what-is-the-difference-between-the-terms-accuracy-and-validation-accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation acccuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_file)