import cv2

cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture('elon_ted_Trim.mp4')

cam.set(3, 648)
cam.set(4, 488)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    _, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.17,
                                          minNeighbors=13, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('face detector', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()