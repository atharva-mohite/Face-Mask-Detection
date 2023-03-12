from flask import Flask, request, render_template
import tensorflow
import cv2
import numpy as np
import os
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# json_file = open('model.json', 'r')
# model_json = json_file.read()
# model = tensorflow.keras.models.model_from_json(model_json)
# model.load_weights("weights.h5")

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

color_dict = {0: (255, 0, 0), 1: (0, 255, 0)}

def predict(im):
    im = cv2.resize(im, (64, 64))
    im = np.reshape(im, (1, 64, 64, 3))
    im = im / 255.0
    result = model.predict(im)
    # confidence = result[0][class_index]
    result = np.argmax(result, axis=1)[0]
    if result:
        label = 'Mask'
    else:
        label = 'No Mask'
    return label, result

def detect_mask(img_file):
    filename = img_file.filename
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:
        im2 = img[y:y + h, x:x + w]
        p = predict(im2)
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[p[1]], 1)
        text_size, _ = cv2.getTextSize(p[0], cv2.FONT_HERSHEY_SIMPLEX, min(img.shape[:2]) / 250.0, 1)
        cv2.rectangle(img, (0, 0), (text_size[0], text_size[1]), (255, 255, 255), -1)
        cv2.putText(img, p[0], (0, text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, min(img.shape[:2]) / 250.0, (0, 0, 0), 1)
        detected_filename = f'detected_{filename}'
        detected_filepath = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
        cv2.imwrite(detected_filepath, img)
    # _, jpeg = cv2.imencode('.jpg', img)
    # result = jpeg.tobytes()
    return p[0], detected_filepath


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    result = detect_mask(file)
    # return render_template('index.html', image=result)
    # image_path = result[1].replace('\\', '/')
    # print(image_path)
    return render_template("index.html", prediction_text=result[0])

if __name__ == "__main__":
    app.run(debug=True)