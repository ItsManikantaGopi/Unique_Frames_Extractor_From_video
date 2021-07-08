import time
import cv2
file="3. Convolutions & Image Features.mp4"
capture=cv2.VideoCapture(file)
total= int( int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
temp=100
while temp < total:
    capture.set(cv2.CAP_PROP_POS_FRAMES, temp)
    print('Position:', int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    _, frame = capture.read()
    cv2.imshow('frame100', frame)
    temp+=100
    time.sleep(3)
cv2.destroyAllWindows()