import os
from datetime import datetime
import face_recognition
import cv2
import numpy as np

def img_capture():
    #load the har cascade classifier for face recognition
    face_cascade =cv2.CascadeClassifier(r"provide the path for calling harcaascade")
    # start the video capture
    #cap =cv2.VideoCapture(0)
    max_faces =1
    #set the maximum number of faces to detect
    num_faces =0

    while True:
        #read a from the camera
        ret,frame = cap.read()
        #convert the frame ti gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect faces in the grayscale frame.
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        #save each detected face to a file
        for (x,y,w,h) in faces:
            #crop the detected face from the frame
            face_img = frame[y:y+h ,x:x+w]
            #save the detected face to a file
            cv2.imwrite(f"face{num_faces}.jpg",face_img)
            #increment the number of the detected faces
            num_faces+=1
            #stop capturing when the maximum number of faces have been detected
            if num_faces >= max_faces:
                break
        
        #show thw frame with the detected face
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Webcam",frame)
            
        #stop the program when the  enter kwy is pressed
        if cv2.waitKey(10)==13 or num_faces>=max_faces:
            break
    cap.release()
    cv2.destroyAllWindows()

def send_mail():
#     #send an HTML email with an embedded image and a plain text message for email 
#     #clients that don't want to dosplay the HTML.
    import smtplib
#     #simple mail transfer protocol.
#     #SMTP(hoost,port):The SMPT class is used to create an instance of an SMTP object which can be used to 
#     # connect to an SMTP server.
#     #the host parameter is the host name of the SMTP server and the port parameter is the port number on which the 
#     #server is listening(usually port 25)
    from email.mime.text import MIMEText
#     #MIME = Multipurpose  internet mail extension
#     #the email.mime.text module is apython part of the email pacakage and is used to create a 
#     # MIME text message object,which can be sent vai SMTP using smtplib module.
    from email.mime.multipart import MIMEMultipart
#     #the email.mime.multipart module is apython part of the email pacakage and is used to create a
#     #  MIMEultipart message object, which can be  used to send email messages with attachments
#     # via SMTP using smtplib module. 
    from email.mime.image import MIMEImage
#     #which is used to attach image file to emil messages via SMTP using the smtplib module

#     # define these one,use the twice
    strFrom ="xxxxx@gmail.com"
    strTo ="xxxxx@gmail.com"

#     #create the root message an fill in the form ,to, and subject headers
    msgRoot = MIMEMultipart("related")
    msgRoot ["Subject"]="test mesaage"
    msgRoot["From"]=strFrom
    msgRoot ["To"] = strTo
# msgRoot.preamble = 'This is a multi-part message in MIME format.'
# Encapsulate the plain and HTML versions of the message body in an
# 'alternative' part, so message agents can decide which they want to display.
    msgAlternative = MIMEMultipart('alternative')
#The 'alternative' subtype is used to indicate that the message contains multiple
# versions of the same content, each with different levels of fidelity or formatting.
    msgRoot .attach(msgAlternative)
    msgText = MIMEText('New person detected, please verify and add to database.')
    msgAlternative.attach(msgText)

# We reference the image in the IMG SRC attribute by the ID we give it below
# msgText = MIMEText('<b>New person detected</b> with image.<br><img src="cid:imagel"><br><b>Please verify</b
# msgAlternative.attach(msgText)  

    # This example assumes the image is in the current directory
    fp = open('face.jpg', 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()

# Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<imagel>')
    msgRoot .attach(msgImage)

#Send the email (this example assumes SMTP authentication is required)
#import smtplib
#smtp = smtplib.SMTP()
#smtp. connect (smtp. example. com’)
# #smtp. login('exampleuser', 'examplepass')

    try:
        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.ehlo()
        smtp_server. login (strFrom, 'app password')
        smtp_server.sendmail(strFrom, strTo, msgRoot.as_string())
        smtp_server.close()
        print ("Email sent successfully!")
    except Exception as ex:
        print ("Something went wrong...",ex)

# smtp.sendmail(strFrom, strTo, msgRoot.as_string())
# smtp.quit()


path =r'E:\bda project1\images'
images=[]
classNames =[]
classDates =[]
date_now =datetime.now() #yyyy-mm-dd
date = date_now.strftime("%m/%d/%y")
# for dates in date
classDates.append(date)
print(classDates)
myList = os.listdir(path)
print(myList)
for cl in (myList):
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencodings(images):
    encodeList=[]

    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]# o implies first images ownwards
        encodeList.append(encode)
    return encodeList

def markAttendence(date):
     with open(r"E:\bda project1\attendance.csv",'r+') as f:
         myDataList = f.readlines()
         nameList =[]
         for line in myDataList:
             entry = line.split(',')
             nameList.append(entry[0])
             
         if date not in nameList:
             now=datetime.now()
             dtString = now.strftime("%d/%m/%Y,%H:%M:%S")
             f.writelines(f'\n{name},{dtString}')
             print("ok")
         
encodeListknown =findencodings(images)
print("Encoding Complete")

cap =cv2.VideoCapture(0)

while True:
    success,img =cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame =face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        print("facedis",faceDis)

        matchIndex =np.argmin(faceDis)
        print("matchindex",matchIndex)
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)
            
        else:
            name ="Unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # incresing the size of the frame
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2) 
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)#fixing rectangle frame on face
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #writting name of the image in the frame
            img_capture()

            markAttendence(name)
           
            

    cv2.imshow("Webcam",img)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()





