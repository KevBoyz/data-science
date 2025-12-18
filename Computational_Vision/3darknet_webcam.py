import cv2 as cv
from time import time

fp = 'trained-files/darknet/'
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# load files
# vid = cv.VideoCapture('assets/walking-pp.mp4')  # To capture a video
vid = cv.VideoCapture(0)
net = cv.dnn.readNet(fp+'yolov4-tiny.weights', fp+'yolov4-tiny.cfg.txt')
with open(fp + 'coco.names.txt', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]  # coll short syntax
# Set a detectiion model
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)  # Save config as config file
# Reading Frames
while True:
    _, frame = vid.read()
    start = time()  # fps system
    classid, score, boxes = model.detect(frame, 0.1, 0.2)  # Detecting objects 0.1, 0.2 = confidence
    end = time()
    # Running the detection
    for (classid, scrore, box) in zip(classid, score, boxes):  # Nice for loop
        color = colors[int(classid % len(colors))]  # Generate a custom color for each class id
        label = f'{class_names[classid]} : {str(score[0])[:4]}'  # Set the box label
        cv.rectangle(frame, box, color, 2)  # Applying the rect     --> box[0, 1] = x, y
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Applying box label
        cv.imshow('detections', frame)
    if cv.waitKey(1) == 'q':
        break


vid.release()
cv.destroyAllWindows()
