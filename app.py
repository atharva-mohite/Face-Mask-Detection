from flask import Flask, request, render_template
import keras
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

json_file = open('model.json', 'r')
model_json = json_file.read()
model = keras.models.model_from_json(model_json)
model.load_weights("weights.h5")

color_dict = {0: (255, 0, 0), 1: (0, 255, 0)}

def predict(im):
    im = cv2.resize(im, (64, 64))
    im = np.reshape(im, (1, 64, 64, 3))
    im = im / 255.0
    result = model.predict(im)
    result = np.argmax(result, axis=1)[0]
    if result:
        label = 'Mask'
    else:
        label = 'No Mask'
    return label

def detect_mask(img_file):
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:
        im2 = img[y:y + h, x:x + w]
        p = predict(im2)
    return p


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    result = detect_mask(file)
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
