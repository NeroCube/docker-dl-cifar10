import numpy as np
import pandas as pd
import gunicorn, cv2, io
from flask import Flask, jsonify, request



app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to ML Service!"

@app.route('/predict', methods = ['POST'])
def upload_file():
    photo = request.files['file']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    return '{}'.format(img)
