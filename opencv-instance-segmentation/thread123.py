

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
net1 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net2 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net3 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net4 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net5 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net6 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net7 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net8 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net9 = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
net_=[net1,net2,net3,net4,net5,net6,net7,net8,net9]
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
start=time.time()
cap = cv2.VideoCapture("cde.mp4")
time.sleep(2.0)
min_=1000
max_=-1000
# loop over frames from the video file stream
cc = -1
img=cap.read()[1]

def _thread1(frame,net):
	print("Thread for "+str(cc)+" frame")
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

while True:
	cc+=1
	_, frame = cap.read()
	if(cc%3!=0):
		continue
	# _thread1(frame)
	# flag = _thread1()
	# if flag:
	# 	break

	# ret, img = cap.read()
	if(cc==1):
		# t1=threading.Thread(target=_thread1,args=(frame,net_[0],))
		# t1.start()
		threading.Thread(target=_thread1,args=(frame,net_[0],)).start()
	# else:
	# 	img=frame
	elif(cc==2):
		# t2=threading.Thread(target=_thread1,args=(frame,net_[1],))
		# t2.start()
		threading.Thread(target=_thread1,args=(frame,net_[1],)).start()
	elif(cc==3):
		# t3=threading.Thread(target=_thread1,args=(frame,net_[2],))
		# t3.start()
		threading.Thread(target=_thread1,args=(frame,net_[2],)).start()
	elif(cc==4):
		threading.Thread(target=_thread1,args=(frame,net_[3],)).start()
		# t4=threading.Thread(target=_thread1,args=(img,))
		# t4.start()
	elif(cc==5):
		threading.Thread(target=_thread1,args=(frame,net_[4],)).start()
		# t5=threading.Thread(target=_thread1,args=(img,))
		# t5.start()
	elif(cc==6):
		threading.Thread(target=_thread1,args=(frame,net_[5],)).start()
		# t6=threading.Thread(target=_thread1,args=(img,))
		# t6.start()
	elif(cc==7):
		threading.Thread(target=_thread1,args=(frame,net_[6],)).start()
		# t7=threading.Thread(target=_thread1,args=(img,))
		# t7.start()
	elif(cc==8):
		threading.Thread(target=_thread1,args=(frame,net_[7],)).start()
		# t8=threading.Thread(target=_thread1,args=(img,))
		# t8.start()
	elif(cc==9):
		threading.Thread(target=_thread1,args=(frame,net_[8],)).start()
		# t9=threading.Thread(target=_thread1,args=(img,))
		# t9.start()
	# else:
	# 	l=[t1,t2,t3]#,,t4,t5,t6,t7,t8,t9]
	# 	while(1):
	# 		try:
	# 			m=[x.isAlive() for x in l]
	# 			idx=m.index(False)
	# 			l[idx]._stop()
	# 			x = threading.Thread(target=_thread,args=(frame,net_[idx],))
	# 			l[idx]=x
	# 			x.start()
	# 			break
	# 		except ValueError:
	# 			continue
	while( threading.active_count()>=9):
		continue
	else:
		threading.Thread(target=_thread1,args=(frame,cv2.dnn.readNetFromTensorflow(weightsPath, configPath),)).start()
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
