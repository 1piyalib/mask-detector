from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import time
import winsound


def get_face_list_from_facenet(frame, facenet_model):

	#This function gets one frame, and then using facenet it gets a lists of faces with mask no-mask and their locations
	minimum_confidence = 0.5
	"""
	through facenet to get the coordinates of the faces
	"""
	(height, width,cell_len) = frame.shape

	#blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None)
	blob = cv2.dnn.blobFromImage(frame, scalefactor = 1.0, size = (300, 300),mean = (104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	facenet_model.setInput(blob)
	detections = facenet_model.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces_list = []
	locations_list = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > minimum_confidence:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(width - 1, endX), min(height - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces_list.append(face)
			locations_list.append((startX, startY, endX, endY))

	return(faces_list,locations_list)


def detect_and_predict_mask(frame, facenet_model, mask_model):

	predictions_list = []

	"""
	Step 3, 4: This function 3)Detect face region in each frame 4)Extracts each face
	"""
	#Eextradct faces list and location from facenet and runs it through the model
	(faces_list,locations_list) = get_face_list_from_facenet(frame, facenet_model)

	# only make a predictions if at least one face was detected
	#debug
	#facelist[0].shape
	#
	"""
	Step 5: This function Applies our model to each face 
	"""

	if len(faces_list) > 0:
		#Get the prediction from model
		faces_list = np.array(faces_list, dtype="float32")
		predictions_list = mask_model.predict(faces_list, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locations_list, predictions_list)



####################   Main Program #################
"""
Variables
"""
model_dir = "model"
#our trained model file
model_name = "mask_detector.model"
#OpenCV pre-trained model files for face(s) detection from a images
prototxt_name = "deploy.prototxt"
weights_file = "res10_300x300_ssd_iter_140000.caffemodel"

model_file = os.path.sep.join([model_dir,model_name])
prototxtPath = os.path.sep.join([model_dir, prototxt_name])
weightsPath = os.path.sep.join([model_dir,weights_file])

"""
Step 1: load face detector trained model and Facenet model from disk
"""
#Load OpenCV  facenet model that detects each face from a picture
facenet_model = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
print(" loading face mask detector model...")
#load our model
mask_model = load_model(model_file)  #keras load model

"""
initialize the video stream and allow the camera sensor to warm up
"""

print(" starting video stream...")
#use imutils to start video stream. src =0 capture first source of video stream one (pi camera)
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

#variable for warning to wear mask. everyone_wearing_mask = False meas someone not wearing mask
everyone_wearing_mask = True

"""
loop over the frames from the video stream
"""

while True: #looping on each frame

	"""
	Step 2: Get each Frame from the vidoe stream
	"""

	# grab the frame from the threaded video stream and resize it
	frame = video_stream.read() #using imutil get the frame
	frame = cv2.rotate(frame, cv2.ROTATE_180) #rotateq frame, because it's coming upside down
	frame = imutils.resize(frame, width=1200)

    #debug: check image with this function
    #cv2.imwrite("test.jpg",frame)

	"""
	Step 3, 4, 5: This function 3)Detect face region in each frame 4)Extracts each frame 5) Applies our model to each face 
	"""
	# detect faces (using the opencv facenet model) in the frame and determine if they are wearing a
	# face mask or not (using our model)
	(location_list, prediction_list) = detect_and_predict_mask(frame, facenet_model, mask_model)

	"""
	Step 6: Put boxes and probability in the image
	"""

	# loop over the detected face locations and their corresponding
	# locations
	for (location_coordinate, pred) in zip(location_list, prediction_list):
		# unpack the bounding location_coordinate and predictions
		(startX, startY, endX, endY) = location_coordinate  #coordinates of face
		(mask, withoutMask) = pred  #probability of mask and no mask

		# determine the class label and color we'll use to draw
		# the bounding location_coordinate and text
		if mask > withoutMask:
			label = "Saving lives"
			color = (0, 255, 0) #green
		else:
			label = "Wear your mask"
			color = (0, 0, 255)  #red

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding location_coordinate rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.7, color = color, thickness= 4)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness = 4)
		#cv2.imwrite("test.jpg", frame)
	# show the output frame
	cv2.imshow("Frame", frame)

	"""
	If no mask warn with wav file
	"""
	if len(prediction_list) > 0:
		(mask, withoutMask) = prediction_list[0]
		if withoutMask > mask:
			if everyone_wearing_mask == True:
				winsound.PlaySound("wear_mask.wav", winsound.SND_ASYNC | winsound.SND_ALIAS)
			everyone_wearing_mask = False
		else:
			everyone_wearing_mask = True

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
video_stream.stop()