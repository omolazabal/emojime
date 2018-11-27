
import numpy as np
import cv2
import dlib
from .utils import shape_to_np, rect_to_bb, distances
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


emotions = ['neutral', 'happy', 'sad', 'fear', 'angry']

class FaceDetector:
    def __init__(self, debug=False, data_gen=False):
        self.debug = debug
        self.data_gen = data_gen
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

        self.img_size = (224, 224)
        self.vs = cv2.VideoCapture(0)
        self.model = pickle.load(open('models/svm_emotion_classifier', 'rb'))
        self.scaler = pickle.load(open('models/emotion_scaler', 'rb'))

    def __del__(self):
        cv2.destroyAllWindows()
        self.vs.release()

    def predict(self):
        width = 400

        _, frame = self.vs.read()
        (h, w) = frame.shape[:2]
        new_size = (width, int(h * width/float(w)))

        frame = cv2.resize(frame, new_size)

        rects = self.detector(frame, 0)
        if len(rects) > 0:
            rect = rects[0]
            shape = self.predictor(frame, rect)
            shape = shape_to_np(shape)
            input_ = distances(shape)
            input_ = self.scaler.transform(input_)
            return self.model.predict(input_)

        return None

    def frame(self):
        width = 400
        _, frame = self.vs.read()
        (h, w) = frame.shape[:2]
        new_size = (width, int(h * width/float(w)))
        frame = cv2.resize(frame, new_size)
        return frame
    
    def landscape_frame(self):
        width = 400

        _, frame = self.vs.read()
        (h, w) = frame.shape[:2]
        new_size = (width, int(h * width/float(w)))

        frame = cv2.resize(frame, new_size)

        if self.debug or self.data_gen:
            rects = self.detector(frame, 0)
            if len(rects) > 0:
                rect = rects[0]
                shape = self.predictor(frame, rect)
                shape = shape_to_np(shape)
                (x, y, w, h) = rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                if self.debug:
                    pred = self.predict()
                    if pred is not None:
                        text = emotions[int(pred[0])]
                    else:
                        text = 'None'
                elif self.data_gen:
                    if w > 120 and h > 120:
                        text = 'Too close'
                    elif w < 75 and h < 75:
                        text = 'Too far'
                    else:
                        text = 'Perfect'
                else:
                    text = 'Undefined'
                text += ' ({}, {})'.format(w, h)
                cv2.putText(frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        _, jpeg = cv2.imencode('.jpg', frame)
        jpeg = jpeg.tobytes()

        return jpeg
