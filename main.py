from cv2 import cv2
video_capture = cv2.VideoCapture(0)

cv2.namedWindow("Window")
cv2.namedWindow("Frame Delta")
cv2.namedWindow("Thresh Frame")

firstFrame = None

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        
    frameDelta = cv2.absdiff(firstFrame, gray)

    threshFrame = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
    threshFrame = cv2.dilate(threshFrame, None, iterations=2)

    cnts,_ = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        # making green rectangle arround the moving object 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

    cv2.imshow("Window", gray)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Thresh Frame", threshFrame)

    firstFrame = gray
    #This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()