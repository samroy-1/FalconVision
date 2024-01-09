import cv2
import imutils
algo = "haarcascade_frontalface_default.xml"
haar_cas=cv2.CascadeClassifier(algo)

cam = cv2.VideoCapture(0)
firstframe=None
area=300

#count = 0

while True:
    _,img=cam.read()
    flip = cv2.flip(img, 1)
    text="Normal"
    grayimg=cv2.cvtColor(flip,cv2.COLOR_BGR2GRAY)
    gaussimg=cv2.GaussianBlur(grayimg,(21,21),0)
    if firstframe is None:
        firstframe=gaussimg
        continue
    imgdiff=cv2.absdiff(firstframe,gaussimg)
    thresh=cv2.threshold(imgdiff, 15, 255,cv2.THRESH_BINARY)[1]
    thresh=cv2.dilate(thresh, kernel=None, iterations=2)
    cont=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = imutils.grab_contours(cont)
    for _ in cont:
        if cv2.contourArea(_)<area:
            continue
        # (a,c,b,d) = cv2.boundingRect(_)
        # cv2.rectangle(flip, (a,c), (a+b,c+d), (0,255,0), 2)
        text="Motion detected"
    firstframe=None
    #count+=1
    #print(count)
    #print(text)
    cv2.putText(flip, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255), 3 )


    gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)

    face = haar_cas.detectMultiScale(gray, 1.3, 4)
    count = 0
    text_= None
    for (x, y, w, h) in face:
        cv2.rectangle(flip, (x, y), (x + w, y + h), (0, 0, 255), 3)
        count+=1
        text_ = (f"face detected: {count}")
    cv2.putText(flip, text_, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imshow("livefeed", flip)
    key=cv2.waitKey(1) &0xff
    if key==ord("o"):
        break
cam.release()
cv2.destroyAllWindows()
