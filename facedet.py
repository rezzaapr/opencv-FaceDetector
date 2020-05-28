import cv2

# Directory Your Image
image = '/home/rezzaapr/Desktop/gambar.jpg'
# Directory haarcascade File
cascadefile = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascadefile)
# Reading Image
image = cv2.imread(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecion Face 
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Face Detected", image)
cv2.waitKey(0)
