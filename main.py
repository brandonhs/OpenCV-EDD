from cv2 import cv2
video_capture = cv2.VideoCapture(0)

cv2.namedWindow("Window")
cv2.namedWindow("Frame Delta")
cv2.namedWindow("Thresh Frame")
cv2.namedWindow("Color")

firstFrame = None
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to grayscale
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # blur the grayscale image

    if firstFrame is None: # check if this is the first frame
        firstFrame = gray # if so, set the first frame to the current frame
        
    frameDelta = cv2.absdiff(firstFrame, gray) # get the difference between the past and current frame

    threshFrame = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1] # using a threshold value, create a new image to snap colors to white
    threshFrame = cv2.dilate(threshFrame, None, iterations=2) # widens the whitened parts of the image
    threshFrame = cv2.GaussianBlur(threshFrame, (21, 21), 0) # blur the threshold frame

    cnts,_ = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # calculate contours in the threshold frame

    for contour in cnts: # loop through each contour
        if cv2.contourArea(contour) < 10000: 
            continue
        (x, y, w, h) = cv2.boundingRect(contour) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) # draw green rectangle around each contour
    
    cv2.imshow("Window", gray)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Thresh Frame", threshFrame)
    cv2.imshow("Color", frame)

    out.write(frame)

    firstFrame = gray
    #This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()