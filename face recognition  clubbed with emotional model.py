import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from keras.models import load_model

model = load_model(r'C:\Users\91799\Downloads\sona project\ER_FACE_ATTENDANCE\emotion_model_updated.h5')

classDates=[]
date_now = datetime.now()
date = date_now.strftime("%d/%m/%Y")
classDates.append(date)


def img_capture():
    # import cv2

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # # Start the video capture
    # cap = cv2.VideoCapture(0)

    # Set the maximum number of faces to detect
    max_faces = 1

    # Initialize the number of detected faces
    num_faces = 0

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Save each detected face to a file
        for (x, y, w, h) in faces:
            # Crop the detected face from the frame
            face_img = frame[y:y+h, x:x+w]

            # Save the detected face to a file
            cv2.imwrite(f'face{num_faces}.jpg', face_img)

            # Increment the number of detected faces
            num_faces += 1

            # Stop capturing when the maximum number of faces have been detected
            if num_faces >= max_faces:
                break

        # Show the frame with the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        # Stop the program when the 'enter' key is pressed
        if cv2.waitKey(10) == 13  or num_faces >= max_faces:
            break


    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

def send_mail():
    # Send an HTML email with an embedded image and a plain text message for
    # email clients that don't want to display the HTML.

    import smtplib   #simple mail transfer protocol
    #SMTP(host, port): The SMTP class is used to create an 
    # instance of an SMTP object, which can be used to connect to an SMTP server. 
    # The host parameter is the hostname of the SMTP server and the port parameter 
    # is the port number on which the server is listening (usually port 25).
    from email.mime.text import MIMEText   #MIME stands for Multipurpose Internet Mail Extensions.
    #The email.mime.text module in Python is part of the email package and is used to 
    # create a MIMEText message object, which can be sent via SMTP 
    # using the smtplib module.
    from email.mime.multipart import MIMEMultipart
    #The email.mime.multipart module in Python is part of the email package and 
    # is used to create a MIMEMultipart message object, which can be
    #  used to send email messages with attachments via SMTP using the smtplib module.
    from email.mime.image import MIMEImage   
    #which can be used to attach image files to
    #email messages via SMTP using the smtplib module.

    # Define these once; use them twice!
    strFrom = 'lekshmichinnu68@gmail.com'
    strTo = 'lekshmichinnu68@gmail.com'

    # Create the root message and fill in the from, to, and subject headers
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'test message'
    msgRoot['From'] = strFrom
    msgRoot['To'] = strTo
    # msgRoot.preamble = 'This is a multi-part message in MIME format.'

    # Encapsulate the plain and HTML versions of the message body in an
    # 'alternative' part, so message agents can decide which they want to display.
    msgAlternative = MIMEMultipart('alternative')
    #The 'alternative' subtype is used to indicate that the message contains multiple 
    # versions of the same content, each with different levels of fidelity or formatting.
    msgRoot.attach(msgAlternative)

    msgText = MIMEText('New person detected, please verify and add to database.')
    msgAlternative.attach(msgText)

    # # We reference the image in the IMG SRC attribute by the ID we give it below
    # msgText = MIMEText('<b>New person detected</b> with image.<br><img src="cid:image1"><br><b>Please verify</b>',)
    # msgAlternative.attach(msgText)

    # This example assumes the image is in the current directory
    fp = open('face0.jpg', 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)

    # Send the email (this example assumes SMTP authentication is required)
    # import smtplib
    # smtp = smtplib.SMTP()
    # smtp.connect('smtp.example.com')
    # smtp.login('exampleuser', 'examplepass')
    try:
        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.ehlo()
        smtp_server.login(strFrom, 'jyhqagezjitjbokp')
        smtp_server.sendmail(strFrom, strTo, msgRoot.as_string())
        smtp_server.close()
        print ("Email sent successfully!")
    except Exception as ex:
        print ("Something went wrong….",ex)
    # smtp.sendmail(strFrom, strTo, msgRoot.as_string())
    # smtp.quit()


path = r'C:\Users\91799\Downloads\sona project\ER_FACE_ATTENDANCE\Images'
images = []
classNames = []

classDates = []
date_now = datetime.now()
date = date_now.strftime("%m/%d/%Y")
# for DATES in date:
classDates.append(date)
print(classDates)    
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open(r'C:\Users\91799\Downloads\sona project\ER_FACE_ATTENDANCE\attendance.csv','r+') as f:
        myDataList = f.readlines()
        print('myDataList',myDataList)
        reve = myDataList[::-1]
        print('reverse', reve)
        for line in reve:      
            entry = line.split(',')
            entry = entry[::-1]
            print(entry)

            print("entry[-1]",entry[-1])
            print("entry[1]",entry[1])

            if name == entry[-1] and dayss == entry[1]:
                pass
            
            else:
                now = datetime.now()
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f'\n{name},{dayss},{dtString}')
                print("written successfully..!!")
            break
        


encodeListKnown = findEncodings(images)

print('Encoding Complete')

cap = cv2.VideoCapture(0)

emotion_dict = {0: "Happy", 1: "Neutral", 2: "Sad"}
while True:
    success, img = cap.read()


    facecasc = cv2.CascadeClassifier(r'C:\Users\91799\Downloads\sona project\ER_FACE_ATTENDANCE\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
   
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)
            # print("matchindex",matchIndex)
            MatchDates = classDates

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                dayss = MatchDates[0]
                print("name",name)
                print("date",dayss)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            
                # # roi_gray = gray[x2:x2 + x1, y1:y1 + y2] #
                # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0) #
                # prediction = model.predict(cropped_img) #
                # maxindex = int(np.argmax(prediction)) #
                # cv2.putText(img, f"{emotion_dict[maxindex]}", (y1+20, x2-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) #

                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

            else:
                name='Unknown'
                y1, x2, y2, x1 = faceLoc

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img,name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                img_capture()
                send_mail()
                markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(10) == 13: #enterkey ascii
        break
cap.release()
cv2.destroyAllWindows()
