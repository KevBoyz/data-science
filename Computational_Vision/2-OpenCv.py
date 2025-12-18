import cv2


def show(im):
    cv2.namedWindow('Image')
    cv2.imshow('Image', im)
    cv2.waitKey()


def find_faces():
    algorithm = cv2.CascadeClassifier('trained-files/haarcascades/haarcascade_frontalface_default.xml')
    im = cv2.imread('assets/person.jpg', cv2.IMREAD_COLOR)
    faces = algorithm.detectMultiScale(im)
    return faces


def output(faces):
    im = cv2.imread('assets/person.jpg', cv2.IMREAD_COLOR)
    for x, y, l, a in faces:
        cv2.rectangle(im, (x, y), (x + l, y + a), (0, 255, 0), 2)
    show(im)


output(find_faces())
