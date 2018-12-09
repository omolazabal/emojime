
import numpy as np
import cv2
import dlib
from . import utils
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
        self.width = 400
        self.extraction_boundary_padding = 0.3
        self.extracted_face = None

    def __del__(self):
        cv2.destroyAllWindows()
        self.vs.release()

    def extract_face(self, frame):
        rects = self.detector(frame, 0)
        if len(rects) > 0:
            rect = rects[0]
            (x, y, w, h) = rect_to_bb(rect)
            (x, y, w, h) = (x-int(w*self.extraction_boundary_padding),
                            y-int(h*self.extraction_boundary_padding),
                            w+(int(w*self.extraction_boundary_padding))*2,
                            h+(int(h*self.extraction_boundary_padding))*2)
            if y > 0 and y+h < frame.shape[0] and x > 0 and x+w < frame.shape[1]:
                frame = frame[y:y+h, x:x+w]
                frame = cv2.resize(frame, utils.new_size(self.width//2, frame))
                return frame
        return None

    def predict(self):
        if self.extracted_face is not None:
            frame = self.extracted_face
        else:
            _, frame = self.vs.read()
            frame = cv2.resize(frame, utils.new_size(self.width, frame))
            frame = self.extract_face(frame)

        if frame is not None:
            rects = self.detector(frame, 0)
            if len(rects) > 0:
                rect = rects[0]
                shape = self.predictor(frame, rect)
                shape = shape_to_np(shape)
                input_ = distances(shape)
                input_ = self.scaler.transform(input_)
                probas = self.model.predict_proba(input_)[0, :]
                max_index = np.argmax(probas)
                return emotions[max_index], round(probas[max_index], 2)

        return None, None

    def save_face(self, path, count):
        if self.extracted_face is not None:
            cv2.imwrite(path, self.extracted_face)
            print('Image {} saved'.format(count))
        else:
            print('Error saving image: face not found')
    
    def landscape_frame(self):
        _, frame = self.vs.read()
        frame = cv2.resize(frame, utils.new_size(self.width, frame))

        if self.debug or self.data_gen:
            self.extracted_face = self.extract_face(frame)
            if self.extracted_face is not None:
                frame = np.copy(self.extracted_face)
                rects = self.detector(frame, 0)
                if len(rects) > 0:
                    rect = rects[0]
                    (x, y, w, h) = rect_to_bb(rect)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    shape = self.predictor(frame, rect)
                    shape = shape_to_np(shape)

                    if self.debug:
                        label, proba = self.predict()
                        if label is not None:
                            text = '{} ({})'.format(label, proba)
                        else:
                            text = 'None'
                    else:
                        text = 'Face detected'

                    cv2.putText(frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
        _, jpeg = cv2.imencode('.jpg', frame)
        jpeg = jpeg.tobytes()
        return jpeg
