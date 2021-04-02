import cv2 

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('sample-videos/manuel_pedestrian_tracker.mp4', fourcc, 20.0, (1280,720))

while(True):
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('frame', frame)
    c = cv2.waitKey(33)
    if c & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()