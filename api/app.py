import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import gunicorn

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to ML Service!"
