import cvzone as cz
import cv2
import os


filename = 'video.avi'
frames_per_second = 24.0
res = '480p'


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def get_dims(cap, res='480p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


vid = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(vid, res))

algorithm = cv2.CascadeClassifier('trained-files/haarcascades/haarcascade_frontalface_default.xml')
ah = cv2.imread('assets/ah1.png', cv2.IMREAD_UNCHANGED)
while True:
    _, frame = vid.read()  # cv2.imwrite('assets/st.png', frame)
    faces = algorithm.detectMultiScale(frame)
    for x, y, l, a in faces:
        try:
            img = cz.overlayPNG(frame, ah, [x - 50, y - 60])  # frame[y_offset:y_end, x_offset:x_end] = ah
            out.write(img)
            cv2.imshow('Display', img)
        except ValueError:
            pass

    if cv2.waitKey(1) == 'q':
        break
