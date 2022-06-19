#import libraries
import cv2
from tensorflow.keras.models import load_model
import numpy as np

print("[INFO] loading handwriting model...")
model = load_model('./model/model_handwritting_recognition.h5')

# map labels with the caracter
word_map = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


print("[INFO] Load image...")
I = cv2.imread('./test_data/character/J_m.jpg')
print("[INFO] preprocessing...")
I_blur = cv2.GaussianBlur(I, (3,3), 0)
I_gray = cv2.cvtColor(I_blur, cv2.COLOR_BGR2GRAY)
_, I_thresh = cv2.threshold(I_gray, 100, 255, cv2.THRESH_BINARY_INV)

I_final = cv2.resize(I_thresh, (28,28))
I_final =np.reshape(I_final, (1,28,28,1))

print("[INFO] predict character...")
pred_char = word_map[np.argmax(model.predict(I_final))]
print("predicted character is: ",pred_char)

cv2.putText(I, "Pred. char.: " + pred_char, (40,60), cv2.FONT_HERSHEY_DUPLEX, 2, color = (255,0,30))
cv2.imshow('handwritten character recognition', I)
cv2.waitKey(0)