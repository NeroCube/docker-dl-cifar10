import gunicorn, io
from flask import Flask, request
from ml.predict import *


app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to ML Service!"

@app.route('/predict', methods = ['POST'])
def upload_file():
    photo = request.files['file']
    result = predict(photo)
    return '{}'.format(result)
