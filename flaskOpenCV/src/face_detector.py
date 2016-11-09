import cv2

def detect_face(filename):
    face_count = 0


    #face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt_tree.xml')
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')

    img = cv2.imread('images/' + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE'''
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_count = face_count + 1


    cv2.imwrite('images/' + filename, img)
    return (face_count)