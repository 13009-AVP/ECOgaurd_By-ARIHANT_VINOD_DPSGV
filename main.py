	
#BEFORE USING THE PROGRAM , MAKE SURE THAT YOU ARE WELL LIGHTEN AND DON"T MIX WITH THE BACKGROUND ( PREFERABLY TO BE DARK IN COLOT)
import cv2
from deepface import DeepFace
import pygame
from pygame import mixer
emotion='neutral'
counter=''
count=0
pygame.init()
min_detection_time = 5
max_consistent_frames = 10
consistent_emotion_frames = 0
consistent_emotion = None
import random
sF = False
aD = False


def music():
    x = random.randint(1, 6)
    if x == 1:
        mixer.music.load('anger_1.wav')    # song 1
    elif x == 2:
        mixer.music.load('anger_2.wav')    # song 2
    elif x == 3:
        mixer.music.load('anger_3.wav')    # song 3
    elif x == 4:
        mixer.music.load('sad_fear_1.wav') # song 4
    elif x == 5:
        mixer.music.load('sad_fear_2.wav') # song 5
    elif x == 6:
        mixer.music.load('sad_fear_3.wav') # song 6

    mixer.music.play(-1)
    mixer.music.pause()

music()

#Vid_C = cv2.VideoCapture('http://192.168.29.66:8080/video')
Vid_C = cv2.VideoCapture(0)
Fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haarcascade for the detection of the front face

while True:
    _, vid = Vid_C.read()
    vid=cv2.resize(vid,(600,400))
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    face = Fc.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)



    for (x, y, w, h) in face:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = vid[y:y + h, x:x + w]
        try:
            analyze = DeepFace.analyze(roi_color,actions=['emotion'])
            if analyze[0]['dominant_emotion'] == 'sad':
                analyze[0]['dominant_emotion'] = 'fear'
            if analyze[0]['dominant_emotion'] == emotion:
                count = count+1
                cv2.putText(vid,analyze[0]['dominant_emotion'], (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                2)

            else:
                count=0
                if analyze[0]['dominant_emotion'] == counter:
                    emotion = counter
                    cv2.putText(vid, analyze[0]['dominant_emotion'], (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                            2)
                else:
                    counter = analyze[0]['dominant_emotion']




        except:
            cv2.putText(vid, emotion, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                        2)

        print(count)

        if (count>=10):


            if  analyze[0]['dominant_emotion'] == 'angry' or analyze[0]['dominant_emotion'] == 'fear' or analyze[0]['dominant_emotion'] == 'sad':

                print('inA')
                if aD == False:
                    music()

                mixer.music.unpause()
                aD=True

            else:
               aD = False
               mixer.music.pause()

    cv2.imshow('Video', vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Vid_C.release()
cv2.destroyAllWindows()


