import cv2 as cv

capture = cv.VideoCapture(0)
while True:
    check,frame = capture.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    casacade = cv.CascadeClassifier("face_front.xml")
    face_detection = casacade.detectMultiScale(gray,1.1,6)
    for x,y,w,h in face_detection:
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        cv.putText(frame,"Face",(x,y-40),cv.FONT_HERSHEY_TRIPLEX,2,(0,0,255),2)
    cv.imshow("video",frame)
    if cv.waitKey(20) and 0xFF == ord("d") : break

capture.release()
cv.destroyAllWindows()
cv.waitKey(0)