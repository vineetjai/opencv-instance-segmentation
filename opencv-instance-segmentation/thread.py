

# t1=threading.Thread(target=_threadi,args=(5,))
# t1.start()
# t2=threading.Thread(target=_threadi,args=(5,))
# t2.start()
# z=t2.isAlive()
# print(z)
# USAGE
# python instance_segmentation.py --mask-rcnn mask-rcnn-coco --kernel 41

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import _thread
import threading

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-k", "--kernel", type=int, default=41,
	help="size of gaussian blur kernel")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# construct the kernel for the Gaussian blur and initialize whether
# or not we are in "privacy mode"
K = (args["kernel"], args["kernel"])
privacy = False


def _threadi(c):
	for i in range(c):
		print(i)

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
cap = cv2.VideoCapture("cde.mp4")
time.sleep(2.0)
min_=1000
max_=-1000
# loop over frames from the video file stream
cc = 0
outputs=[]
def _thread(frame):
	global privacy,outputs
	frame = imutils.resize(frame, width=600)
	(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final",
		"detection_masks"])
	idxs = np.argsort(boxes[0, 0, :, 2])[::-1]
	mask = None
	roi = None
	coords = None
	# loop over the indexes
	for i in idxs:
		# extract the class ID of the detection along with the
		# confidence (i.e., probability) associated with the
		# prediction
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]

		# if the detection is not the 'person' class, ignore it
		if LABELS[classID] != "person":
			continue

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image and then compute the width and the
			# height of the bounding box
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			coords = (startX, startY, endX, endY)
			boxW = endX - startX
			boxH = endY - startY

			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > args["threshold"])

			# extract the ROI and break from the loop (since we make
			# the assumption there is only *one* person in the frame
			# who is also the person with the highest prediction
			# confidence)
			roi = frame[startY:endY, startX:endX][mask]
			break
	# initialize our output frame
	output = frame.copy()

	# if the mask is not None *and* we are in privacy mode, then we
	# know we can apply the mask and ROI to the output image
	if mask is not None and privacy:
		# blur the output frame
		output = cv2.GaussianBlur(output, K, 0)

		# add the ROI to the output frame for only the masked region
		(startX, startY, endX, endY) = coords
		output[startY:endY, startX:endX][mask] = roi

	outputs.append(output)
while True:
	cc+=1
	start = time.time()
	
	ret, img = cap.read()
	if(cc==1):
		t1=threading.Thread(target=_thread,args=(img,))
		t1.start()
	# elif(cc==2):
	# 	t2=threading.Thread(target=_thread,args=(img,))
	# 	t2.start()
	# elif(cc==3):
	# 	t3=threading.Thread(target=_thread,args=(img,))
	# 	t3.start()
	# elif(cc==4):
	# 	t4=threading.Thread(target=_thread,args=(img,))
	# 	t4.start()
	# elif(cc==5):
	# 	t5=threading.Thread(target=_thread,args=(img,))
	# 	t5.start()
	# elif(cc==6):
	# 	t6=threading.Thread(target=_thread,args=(img,))
	# 	t6.start()
	# elif(cc==7):
	# 	t7=threading.Thread(target=_thread,args=(img,))
	# 	t7.start()
	# elif(cc==8):
	# 	t8=threading.Thread(target=_thread,args=(img,))
	# 	t8.start()
	# elif(cc==9):
	# 	t9=threading.Thread(target=_thread,args=(img,))
	# 	t9.start()
	else:
		l=[t1]#,t2,t3,t4,t5,t6,t7,t8,t9]
		while(1):
			try:
				m=[x.isAlive() for x in l]
				idx=m.index(False)
				l[idx]._stop()
				l[idx] = threading.Thread(target=_thread,args=(img,))
				l[idx].start()
				break
			except ValueError:
				continue		
	# for img in outputs:
	# 	cv2.imshow("input", img)
	# 	key = cv2.waitKey(1) & 0xFF

	# # if the `p` key was pressed, toggle privacy mode
	# 	if key == ord("p"):
	# 		privacy = not privacy

	# # if the `q` key was pressed, break from the loop
	# 	elif key == ord("q"):
	# 		break

cap.release()
cv2.destroyAllWindows()