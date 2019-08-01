import numpy as np
import cv2
from keras.models import load_model

# Loading model
model = load_model('smile_dection_model_cnn.h5')
cap = cv2.VideoCapture(0)
cap.set(3,400)
cap.set(4,300)
cap.set(12, 0.3)  
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (64,64))
    normalized_frame = (resized_frame - np.min(resized_frame)) / np.max(resized_frame)
    prediction = model.predict(np.expand_dims(resized_frame, axis=0)).round()
    if (prediction[0][0] == 0):
        cv2.putText(img=frame,
            text= u"Smiling",
            org=(int(frameWidth/2 - 20),
            int(frameHeight/2)),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1, color = (255, 255, 255))
    else:
        cv2.putText(img=frame,
            text=u"Not Smiling",
            org=(int(frameWidth/2 - 20),
            int(frameHeight/2)),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = 1, color = (255, 255, 255))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()