from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import _thread
import threading
from multiprocessing import Process
import os

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
net= cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# construct the kernel for the Gaussian blur and initialize whether
# or not we are in "privacy mode"
K = (args["kernel"], args["kernel"])
privacy = False


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()


def multiprocess_():
	cap = cv2.VideoCapture("cde.mp4")
	time.sleep(2.0)
	cc=0
	info('function multiprocess_')
	start=time.time()
	while True:
		_, frame = cap.read()
		# cv2.imshow(frame)
		cc+=1
		if(cc%3!=0):
			continue
		print(str(cc)+" frame undergoing masking")
		global img,privacy
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
			classID = int(boxes[0, 0, i, 1])
			confidence = boxes[0, 0, i, 2]

			# if the detection is not the 'person' class, ignore it
			if LABELS[classID] != "person":
				continue
			if confidence > args["confidence"]:
				box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				coords = (startX, startY, endX, endY)
				boxW = endX - startX
				boxH = endY - startY

				mask = masks[i, classID]
				mask = cv2.resize(mask, (boxW, boxH),
					interpolation=cv2.INTER_NEAREST)
				mask = (mask > args["threshold"])

				roi = frame[startY:endY, startX:endX][mask]
				break
		# initialize our output frame
		output = frame.copy()

		if mask is not None and privacy:
			# blur the output frame
			output = cv2.GaussianBlur(output, K, 0)

			# add the ROI to the output frame for only the masked region
			(startX, startY, endX, endY) = coords
			output[startY:endY, startX:endX][mask] = roi
		img=output
		# for img in outputs:
		cv2.imshow("input", img)
		key = cv2.waitKey(1) & 0xFF

	# if the `p` key was pressed, toggle privacy mode
		if key == ord("p"):
			privacy = not privacy

	# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			break
		print(time.time()-start)
	cap.release()
	cv2.destroyAllWindows()
	return 
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

info('main line')
p = Process(target=multiprocess_, args=())
p.start()
p.join()
