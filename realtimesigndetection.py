from keras.models import model_from_json
import cv2
import numpy as np

import json

# Load model architecture from JSON file
with open("ASLsign-main\ISLModel.json", "r") as json_file:
    model_json = json_file.read()

# Load model from JSON
model = model_from_json(model_json)

# Load weights into the model
model.load_weights("ASLsign-main\ISLModel.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

cap = cv2.VideoCapture(0)
# label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]

label = ['0', '1', '2', '3' ]
# label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

while True:
    _,frame = cap.read()
    cv2.rectangle(frame,(0,40),(300,300),(0, 165, 255),1)
    cropframe=frame[40:300,0:300]
    cropframe=cv2.cvtColor(cropframe,cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe,(48,48))
    cropframe = extract_features(cropframe)

    # Predict the class
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]
    
    # Display the prediction
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    accu = "{:.2f}".format(np.max(pred) * 100)
    cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("output", frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()