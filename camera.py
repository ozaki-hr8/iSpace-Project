import cv2

cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()

    cv2.imshow('camera' , frame)

    key =cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()