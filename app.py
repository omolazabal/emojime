
from flask import Flask, render_template, Response, jsonify
from emojime import FaceDetector
from emojime import emoji
import numpy as np
import sys
import subprocess
import argparse
import cv2


app = Flask(__name__)
emotions = ['neutral', 'happy', 'sad', 'fear', 'angry']

def write_to_clipboard(output):
    process = subprocess.Popen(
        'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
    process.communicate(output.encode('utf-8'))

def gen_video_feed(detector):
    while True:
        landscape_frame = detector.landscape_frame()
        yield (b'--landscape_frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + landscape_frame + b'\r\n\r\n')

@app.route('/')
def index():
    global args
    if args.debug:
        return render_template('debug.html')
    elif args.data:
        return render_template('data.html')
    else:
        return render_template('demo.html')

@app.route('/video_feed')
def video_feed():
    global face_detector
    return Response(gen_video_feed(face_detector), mimetype='multipart/x-mixed-replace; boundary=landscape_frame')

@app.route('/action')
def background_process_test():
    global args
    global face_detector
    if args.data:
        cv2.imwrite('data/emotion-images/{}/{}.png'.format(args.type, str(args.start)), face_detector.frame())
        return jsonify(result='nothing')
    else:
        pred = face_detector.predict()
        if pred is not None:
            emotion = emoji[emotions[int(pred[0])]]
            write_to_clipboard(emoji[emotions[int(pred[0])]])
        else:
            emotion = emoji[emotions[0]]
        return jsonify(result=emotion)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug', action='store_true', help='Enter debug mode.')
    ap.add_argument('--data', action='store_true', help='Enter data generation mode.')
    ap.add_argument('--type', help='Specify which type of data to generate [angry, fear, happy, sad, neutral].')
    ap.add_argument('--start', help='Specify which index to start at.', type=int)
    global args
    global face_detector
    args = ap.parse_args()
    face_detector = FaceDetector(args.debug, args.data)
    app.run(host='0.0.0.0', threaded=True)

