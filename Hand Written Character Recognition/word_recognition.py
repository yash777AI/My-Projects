#import libraries
import cv2
from tensorflow.keras.models import load_model
import numpy as np

print("[INFO] loading handwriting model...")
model = load_model('./model/model_handwritting_recognition.h5')

# map labels with the caracter
word_map = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


print("[INFO] Load image...")
I = cv2.imread('./test_data/word/bhavisha_n.jpg')
#I = cv2.resize(I, (640,480))
print("[INFO] preprocessing...")
I_blur = cv2.GaussianBlur(I, (3,3), 0)
I_gray = cv2.cvtColor(I_blur, cv2.COLOR_BGR2GRAY)
_, I_thresh = cv2.threshold(I_gray, 100, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
I_dilate = cv2.dilate(I_thresh, kernel, iterations = 3) 

word=""
cntrs,her=cv2.findContours(I_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(cntrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
for cnt in sorted_ctrs:    
    if cv2.contourArea(cnt) > 500:
        #print(cv2.contourArea(cnt))
        (x1, y1, w1, h1) = cv2.boundingRect(cnt)
        #print(x1,y1,w1,h1)
        cv2.rectangle(I, (x1, y1), (x1 + w1, y1 + h1), (255, 80, 0), 1)
        roi = I_thresh[y1:y1 + h1, x1:x1 + w1]
        I_final = cv2.resize(roi, (28,28))
        I_final =np.reshape(I_final, (1,28,28,1))
        pred_char = word_map[np.argmax(model.predict(I_final))]
        cv2.putText(I,pred_char, (x1,y1-10), cv2.FONT_HERSHEY_DUPLEX, 1.2, color = (255,80,0))
        word+=pred_char
        cv2.imshow('handwritten word recognition', I)
        cv2.waitKey(0)
        
print("predicted word is:",word)